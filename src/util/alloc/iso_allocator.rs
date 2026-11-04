// n.b. this whole file is only compiled in when #[cfg(feature = "thread_local_gc")].
// So that feature's for free.
use super::allocator::{Allocator, AllocatorContext};
use super::immix_allocator::ImmixAllocSemantics;
use crate::policy::space::Space;
use crate::util::{Address, VMThread};
use crate::vm::VMBinding;
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
        Self {
            likely_private: ImmixAllocator::new(tls, mutator_id, space, context.clone(), copy, semantic),
            likely_public: ImmixAllocator::new(tls, mutator_id, space, context, copy, semantic),
            predictor_table: &super::allocator::PREDICTOR_TABLE as *const AtomicU8,
        }
    }

    fn is_likely_public_allocation(&self) -> bool {
        self.get_context().get_alloc_options().likely_public
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
    // The following methods dispatch to either sub-allocator
    // depending on if the allocation is likely private or public.
    fn alloc(&mut self, size: usize, align: usize, offset: usize) -> Address {
        if self.is_likely_public_allocation() {
            self.likely_public.alloc(size, align, offset)
        } else {
            self.likely_private.alloc(size, align, offset)
        }
    }
    fn alloc_slow_once(&mut self, size: usize, align: usize, offset: usize) -> Address {
        if self.is_likely_public_allocation() {
            self.likely_public.alloc_slow_once(size, align, offset)
        } else {
            self.likely_private.alloc_slow_once(size, align, offset)
        }
    }
     fn alloc_slow_once_precise_stress(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        need_poll: bool,
     ) -> Address {
         // XXX: I don't think this will be all too precise? We will still hit the slow path
         // every N bytes on average, yes, but we can get more variance from an
         // unlucky sequence of predictions. e.g. if we allocate N-1 bytes from either
         // buffer, the next allocation will hit the slow path after 2(N-1) bytes of
         // allocations. But this would emerge whenever we are partitioning allocations
         // between multiple allocators, so let's not think about it too hard.
         if self.is_likely_public_allocation() {
            self.likely_public.alloc_slow_once_precise_stress(size, align, offset, need_poll)
        } else {
            self.likely_private.alloc_slow_once_precise_stress(size, align, offset, need_poll)
        }
     }
}
