use crate::util::ObjectReference;
use crate::vm::VMBinding;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(C)]
pub struct RemSetEntry<VM: VMBinding>(VM::VMSlot, ObjectReference);

impl<VM: VMBinding> RemSetEntry<VM> {
    fn encode(slot: VM::VMSlot, o: ObjectReference) -> Self {
        Self(slot, o)
    }
    pub fn decode(&self) -> (VM::VMSlot, ObjectReference) {
        (self.0, self.1)
    }
}

pub struct RemSet<VM: VMBinding> {
    pub(super) gc_buffers: Vec<UnsafeCell<Vec<RemSetEntry<VM>>>>,
    saved_buffers: Vec<UnsafeCell<Vec<Vec<RemSetEntry<VM>>>>>,
    _p: PhantomData<VM>,
    size: AtomicUsize,
}

unsafe impl<VM: VMBinding> Sync for RemSet<VM> { }

const BUFFER_CAPACITY: usize = 1024;

impl<VM: VMBinding> RemSet<VM> {
    pub fn new(workers: usize) -> Self {
        let mut rs = Self {
            gc_buffers: vec![],
            saved_buffers: vec![],
            _p: PhantomData,
            size: AtomicUsize::new(0),
        };
        rs.gc_buffers
            .resize_with(workers, || UnsafeCell::new(vec![]));
        rs.saved_buffers
            .resize_with(workers, || UnsafeCell::new(vec![]));
        rs
    }

    #[allow(clippy::mut_from_ref)]
    fn gc_buffer(&self, id: usize) -> &mut Vec<RemSetEntry<VM>> {
        unsafe { &mut *self.gc_buffers[id].get() }
    }

    pub fn flush_all(&self, buffer_consumer: &mut impl FnMut(Vec<RemSetEntry<VM>>)) {
        self.size.store(0, Ordering::SeqCst);
        for id in 0..self.gc_buffers.len() {
            if !self.gc_buffer(id).is_empty() {
                let remset = std::mem::take(self.gc_buffer(id));
                buffer_consumer(remset);
            }
        }
        for id in 0..self.saved_buffers.len() {
            let buf = unsafe { &mut *self.saved_buffers[id].get() };
            if !buf.is_empty() {
                let packets = std::mem::take(buf);
                for p in packets {
                    buffer_consumer(p);
                }
            }
        }
    }

    #[cold]
    fn flush(&self, id: usize) {
        if !self.gc_buffer(id).is_empty() {
            let remset = std::mem::take(self.gc_buffer(id));
            self.size.fetch_add(remset.len(), Ordering::SeqCst);
            let packet_buffer = unsafe { &mut *self.saved_buffers[id].get() };
            packet_buffer.push(remset);
        }
    }

    pub fn record(&self, s: VM::VMSlot, o: ObjectReference) {
        let id = crate::scheduler::current_worker_ordinal();
        self.gc_buffer(id).push(RemSetEntry::encode(s, o));
        if self.gc_buffer(id).len() >= BUFFER_CAPACITY {
            self.flush(id)
        }
    }
}
