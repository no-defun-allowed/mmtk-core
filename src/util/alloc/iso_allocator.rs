use crate::util::alloc::allocator::AllocatorContext;
use crate::util::alloc::immix_allocator::ImmixAllocSemantics;
use crate::policy::space::Space;
use crate::util::VMThread;
use crate::vm::VMBinding;
use super::immix_allocator::ImmixAllocator;
use std::sync::Arc;

/// A fancier allocator for Iso.
/// Each mutator has separate allocators for objects
/// which the mutator predicts will be private or public.
#[repr(C)]
pub struct IsoAllocator<VM: VMBinding> {
    likely_private: ImmixAllocator<VM>,
    likely_public: ImmixAllocator<VM>,
    predictor_table: *mut u8,
}

impl<VM: VMBinding> IsoAllocator<VM> {
    pub(crate) fn new(
        tls: VMThread,
        #[cfg(feature = "thread_local_gc")] mutator_id: u32,
        space: Option<&'static dyn Space<VM>>,
        context: Arc<AllocatorContext<VM>>,
        copy: bool,
        semantic: Option<ImmixAllocSemantics>) -> Self {
        Self {
            likely_private: ImmixAllocator::new(tls, mutator_id, space, context.clone(), copy, semantic),
            likely_public: ImmixAllocator::new(tls, mutator_id, space, context, copy, semantic),
            predictor_table: unimplemented!(),
        }
    }
}
    
