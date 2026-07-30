use std::sync::Arc;

use crate::policy::marksweepspace::malloc_ms::MallocSpace;
use crate::policy::space::Space;
use crate::util::alloc::Allocator;
use crate::util::opaque_pointer::*;
use crate::util::Address;
use crate::vm::VMBinding;
use crate::AllocationSemantics;

use super::allocator::AllocatorContext;

/// The allocator that internally uses malloc for all the allocation requests.
/// This allocator is only intended for experimental uses.
#[repr(C)]
pub struct MallocAllocator<VM: VMBinding> {
    /// [`VMThread`] associated with this allocator instance
    pub tls: VMThread,
    /// [`Space`](src/policy/space/Space) instance associated with this allocator instance.
    space: &'static MallocSpace<VM>,
    context: Arc<AllocatorContext<VM>>,
}

impl<VM: VMBinding> Allocator<VM> for MallocAllocator<VM> {
    fn get_space(&self) -> &'static dyn Space<VM> {
        self.space as &'static dyn Space<VM>
    }

    fn get_context(&self) -> &AllocatorContext<VM> {
        &self.context
    }

    fn alloc(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        semantics: AllocationSemantics,
    ) -> Address {
        assert!(matches!(
            semantics,
            AllocationSemantics::Default
                | AllocationSemantics::NonMoving
                | AllocationSemantics::PrimitiveArray
                | AllocationSemantics::ReferenceArray
        ));
        self.alloc_slow(size, align, offset, semantics)
    }

    fn get_tls(&self) -> VMThread {
        self.tls
    }

    fn does_thread_local_allocation(&self) -> bool {
        false
    }

    fn alloc_slow_once(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        _semantics: AllocationSemantics,
    ) -> Address {
        self.space.alloc(self.tls, size, align, offset)
    }
}

impl<VM: VMBinding> MallocAllocator<VM> {
    pub(crate) fn new(
        tls: VMThread,
        space: &'static MallocSpace<VM>,
        context: Arc<AllocatorContext<VM>>,
    ) -> Self {
        MallocAllocator {
            tls,
            space,
            context,
        }
    }
}
