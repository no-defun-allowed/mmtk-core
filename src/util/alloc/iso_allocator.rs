// n.b. this whole file is only compiled in when #[cfg(feature = "thread_local_gc")].
// So that feature's for free.
use super::allocator::{Allocator, AllocatorContext};
use super::immix_allocator::ImmixAllocSemantics;
use crate::policy::space::Space;
use crate::policy::immix::ImmixSpace;
use crate::util::{Address, VMThread};
use crate::vm::VMBinding;
use crate::AllocationSemantics;
use super::immix_allocator::ImmixAllocator;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;

/// A fancier allocator for Iso.
/// Each mutator has separate allocators for objects
/// which the mutator predicts will be private or public.
#[repr(C)]
pub struct IsoAllocator<VM: VMBinding> {
    likely_private: ImmixAllocator<VM>,
    likely_public: ImmixAllocator<VM>,
    predictor_table: *const AtomicU8,
}

unsafe impl<VM: VMBinding> Send for IsoAllocator<VM> {}

impl<VM: VMBinding> IsoAllocator<VM> {
    pub(crate) fn new(
        tls: VMThread,
        mutator_id: u32,
        space: Option<&'static dyn Space<VM>>,
        context: Arc<AllocatorContext<VM>>,
        copy: bool,
        semantic: Option<ImmixAllocSemantics>) -> Self {
        assert_eq!(semantic, None, "IsoAllocator is only for mutators");
        Self {
            likely_private: ImmixAllocator::new(tls, mutator_id, space, context.clone(), copy, None),
            likely_public: ImmixAllocator::new(tls, mutator_id, space, context, copy, None),
            predictor_table: &super::allocator::PREDICTOR_TABLE as *const AtomicU8,
        }
    }

    fn is_likely_public_allocation(&self, s: AllocationSemantics) -> bool {
        match s {
            AllocationSemantics::Default => false,
            AllocationSemantics::LikelyPublic => true,
            _ => unreachable!("{s:?} should be semantics for an IsoAllocator"),
        }
    }

    // These all get forwarded to both sub-allocators.
    pub fn prepare(&mut self) {
        self.likely_private.prepare();
        self.likely_public.prepare();
    }
    pub fn release(&mut self) {
        self.likely_private.release();
        self.likely_public.release();
    }
    pub fn reset(&mut self) {
        self.likely_private.reset();
        self.likely_public.reset();
    }
    pub fn thread_local_prepare(&mut self) {
        self.likely_private.thread_local_prepare();
        self.likely_public.thread_local_prepare();
    }
    pub fn thread_local_defrag_prepare(&mut self) {
        self.likely_private.thread_local_defrag_prepare();
        self.likely_public.thread_local_defrag_prepare();
    }
    pub fn thread_local_release(&mut self) {
        self.likely_private.thread_local_release();
        self.likely_public.thread_local_release();
    }
    pub(crate) fn immix_space(&self) -> &'static ImmixSpace<VM> {
        self.likely_private.immix_space()
    }

    // These pertain to private objects, so I forward them to the
    // likely_private sub-allocator.
    #[cfg(feature = "thread_local_gc_copying")]
    pub fn thread_local_copy_reserve_exhausted(&mut self, size: usize) -> bool {
        self.likely_private.thread_local_copy_reserve_exhausted(size)
    }

    pub fn alloc_copy(&mut self, size: usize, align: usize, offset: usize) -> Address {
        self.likely_private.alloc_copy(size, align, offset)
    }

    #[cfg(feature = "thread_local_gc_copying")]
    pub fn thread_local_gc_reserve_pages(&mut self) {
        self.likely_private.thread_local_gc_reserve_pages()
    }
}
    

impl<VM: VMBinding> Allocator<VM> for IsoAllocator<VM> {
    fn get_tls(&self) -> VMThread { self.likely_private.get_tls() }
    fn get_space(&self) -> &'static dyn Space<VM> { self.likely_private.get_space() }
    fn get_context(&self) -> &AllocatorContext<VM> { self.likely_private.get_context() }
    fn does_thread_local_allocation(&self) -> bool { true }
    fn get_thread_local_buffer_granularity(&self) -> usize {
        self.likely_private.get_thread_local_buffer_granularity()
    }
    #[cfg(feature = "thread_local_gc_copying")]
    fn local_heap_in_pages(&self) -> usize {
        self.likely_private.local_heap_in_pages() + self.likely_public.local_heap_in_pages()
    }
    fn alloc(&mut self, size: usize, align: usize, offset: usize) -> Address {
        self.likely_private.alloc(size, align, offset)
    }
    // The following methods dispatch to either sub-allocator
    // depending on if the allocation is likely private or public.
    fn alloc_with_semantics(&mut self, size: usize, align: usize, offset: usize,
                            semantics: AllocationSemantics) -> Address {
        if self.is_likely_public_allocation(semantics) {
            self.likely_public.alloc(size, align, offset)
        } else {
            self.likely_private.alloc(size, align, offset)
        }
    }
    // XXX: Are these reachable in practise? I understand that alloc() can
    // ping-pong between default methods on Allocator and these methods,
    // but the implementation of alloc() dispatches to the right sub-allocator,
    // and all the ping-pong happens on that allocator. Oh well.
    fn alloc_slow_once(&mut self, size: usize, align: usize, offset: usize) -> Address {
        self.likely_private.alloc_slow_once(size, align, offset)
    }
     fn alloc_slow_once_precise_stress(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        need_poll: bool,
     ) -> Address {
         self.likely_private.alloc_slow_once_precise_stress(size, align, offset, need_poll)
     }
}
