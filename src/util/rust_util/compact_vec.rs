//! A `Vec`-like growable array that uses `u16` instead of `usize` for its
//! length and capacity fields, in order to reduce memory overhead when a very
//! large number of small vectors need to be stored (e.g. one per stubbed
//! object in the Compressor's stub table).
//!
//! Because the length and capacity are stored as `u16`, a [`CompactVec`] can
//! hold at most [`u16::MAX`] elements.

use std::alloc::{self, Layout};
use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};

/// See the [module-level documentation](self) for details.
pub struct CompactVec<T> {
    ptr: NonNull<T>,
    len: u16,
    cap: u16,
}

unsafe impl<T: Send> Send for CompactVec<T> {}
unsafe impl<T: Sync> Sync for CompactVec<T> {}

impl<T> CompactVec<T> {
    /// Create a new, empty `CompactVec`. This does not allocate.
    pub const fn new() -> Self {
        CompactVec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// Create a new, empty `CompactVec` with at least the given capacity.
    ///
    /// # Panics
    /// Panics if `cap` is greater than `u16::MAX`.
    pub fn with_capacity(cap: usize) -> Self {
        assert!(
            cap <= u16::MAX as usize,
            "CompactVec cannot hold more than u16::MAX elements"
        );
        if cap == 0 {
            return Self::new();
        }
        let layout = Self::layout_for(cap);
        let ptr = unsafe { alloc::alloc(layout) } as *mut T;
        let ptr = match NonNull::new(ptr) {
            Some(ptr) => ptr,
            None => alloc::handle_alloc_error(layout),
        };
        CompactVec {
            ptr,
            len: 0,
            cap: cap as u16,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap as usize
    }

    fn layout_for(cap: usize) -> Layout {
        Layout::array::<T>(cap).expect("CompactVec capacity overflow")
    }

    fn grow(&mut self) {
        assert_ne!(
            std::mem::size_of::<T>(),
            0,
            "CompactVec does not support zero-sized types"
        );

        let new_cap = if self.cap == 0 {
            1
        } else {
            let doubled = self.cap as usize * 2;
            assert!(
                doubled <= u16::MAX as usize,
                "CompactVec cannot grow beyond u16::MAX elements"
            );
            doubled
        };
        let new_layout = Self::layout_for(new_cap);

        let new_ptr = if self.cap == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Self::layout_for(self.cap as usize);
            unsafe { alloc::realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size()) }
        };

        self.ptr = match NonNull::new(new_ptr as *mut T) {
            Some(ptr) => ptr,
            None => alloc::handle_alloc_error(new_layout),
        };
        self.cap = new_cap as u16;
    }

    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.grow();
        }
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len as usize), value);
        }
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(ptr::read(self.ptr.as_ptr().add(self.len as usize))) }
        }
    }

    pub fn clear(&mut self) {
        while self.pop().is_some() {}
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize) }
    }
}

impl<T> Default for CompactVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for CompactVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for CompactVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Drop for CompactVec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            unsafe {
                ptr::drop_in_place(self.as_mut_slice());
                alloc::dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Self::layout_for(self.cap as usize),
                );
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for CompactVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T> FromIterator<T> for CompactVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = CompactVec::new();
        for item in iter {
            vec.push(item);
        }
        vec
    }
}

impl<'a, T> IntoIterator for &'a CompactVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut CompactVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// Owning iterator for [`CompactVec`], created by [`CompactVec::into_iter`].
pub struct IntoIter<T> {
    buf: NonNull<T>,
    cap: u16,
    ptr: *mut T,
    end: *mut T,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                let old = self.ptr;
                self.ptr = self.ptr.add(1);
                Some(ptr::read(old))
            }
        }
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}
        if self.cap != 0 {
            unsafe {
                alloc::dealloc(
                    self.buf.as_ptr() as *mut u8,
                    CompactVec::<T>::layout_for(self.cap as usize),
                );
            }
        }
    }
}

impl<T> IntoIterator for CompactVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> IntoIter<T> {
        let me = ManuallyDrop::new(self);
        let ptr = me.ptr.as_ptr();
        let len = me.len as usize;
        IntoIter {
            buf: me.ptr,
            cap: me.cap,
            ptr,
            end: unsafe { ptr.add(len) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn test_push_pop_len() {
        let mut v: CompactVec<u32> = CompactVec::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        for i in 0..100 {
            v.push(i);
        }
        assert_eq!(v.len(), 100);
        assert_eq!(&v[..], &(0..100).collect::<Vec<_>>()[..]);
        for i in (0..100).rev() {
            assert_eq!(v.pop(), Some(i));
        }
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_iter() {
        let v: CompactVec<u32> = (0..10).collect();
        let sum: u32 = v.iter().sum();
        assert_eq!(sum, 45);
        let collected: Vec<u32> = v.into_iter().collect();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_drop_runs_for_elements() {
        struct DropCounter<'a>(&'a RefCell<usize>);
        impl Drop for DropCounter<'_> {
            fn drop(&mut self) {
                *self.0.borrow_mut() += 1;
            }
        }

        let count = RefCell::new(0);
        {
            let mut v: CompactVec<DropCounter> = CompactVec::new();
            for _ in 0..5 {
                v.push(DropCounter(&count));
            }
        }
        assert_eq!(*count.borrow(), 5);
    }

    #[test]
    #[should_panic]
    fn test_capacity_overflow() {
        let mut v: CompactVec<u8> = CompactVec::with_capacity(u16::MAX as usize);
        for _ in 0..=u16::MAX {
            v.push(0);
        }
    }
}
