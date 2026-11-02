use super::immix_allocator::ImmixAllocator;
use crate::vm::VMBinding;

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
        #[cfg(feature = "thread_local_gc")] _mutator_id: u32,
        space: Option<&'static dyn Space<VM>>,
        context: Arc<AllocatorContext<VM>>,
        copy: bool,
        _semantic: Option<ImmixAllocSemantics>) {
        Self {
            likely_private: ImmixAllocator::new(tls, mutator_id, space, context, copy, semantic),
            likely_private: ImmixAllocator::new(tls, mutator_id, space, context, copy, semantic),
            predictor_table: unimplemented!(),
        }
    }
}
    
