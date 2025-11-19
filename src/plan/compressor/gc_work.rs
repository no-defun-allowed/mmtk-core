use super::global::Compressor;
use super::process_edges::{CompressorCondition, PlanProcessEdges};
use crate::policy::compressor::{CompressorSpace, TRACE_KIND_FORWARD_ROOT, TRACE_KIND_MARK};
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::vm::{ActivePlan, Scanning, VMBinding};
use crate::MMTK;
use std::marker::{PhantomData, Send};

/// Generate more packets by calling a method on [`CompressorSpace`].
pub struct GenerateWork<VM: VMBinding, F: Fn() + Send + 'static> {
    f: F,
    _p: PhantomData<VM>,
}

impl<VM: VMBinding, F: Fn() + Send + 'static> GCWork<VM>
    for GenerateWork<VM, F>
{
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        (self.f)();
    }
}

impl<VM: VMBinding, F: Fn() + Send + 'static> GenerateWork<VM, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _p: PhantomData
        }
    }
}

/// Create another round of root scanning work packets
/// to update object references.
pub struct UpdateReferences<VM: VMBinding> {
    p: PhantomData<VM>,
}

unsafe impl<VM: VMBinding> Send for UpdateReferences<VM> {}

impl<VM: VMBinding> GCWork<VM> for UpdateReferences<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        // The following needs to be done right before the second round of root scanning
        VM::VMScanning::prepare_for_roots_re_scanning();
        mmtk.state.prepare_for_stack_scanning();
        #[cfg(feature = "extreme_assertions")]
        mmtk.slot_logger.reset();

        for mutator in VM::VMActivePlan::mutators() {
            mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots].add(ScanMutatorRoots::<
                CompressorForwardingWorkContext<VM>,
            >(mutator));
        }

        mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots]
            .add(ScanVMSpecificRoots::<CompressorForwardingWorkContext<VM>>::new());
    }
}

impl<VM: VMBinding> UpdateReferences<VM> {
    pub fn new() -> Self {
        Self { p: PhantomData }
    }
}

/// Reset the allocator and update references in large object space.
pub struct AfterCompact<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for AfterCompact<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.after_compact();
    }
}

impl<VM: VMBinding> AfterCompact<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
    ) -> Self {
        Self {
            compressor_space,
        }
    }
}

/// Marking trace
pub type MarkingProcessEdges<VM> =
    PlanProcessEdges<VM, Compressor<VM>, CompressorCondition<VM>, TRACE_KIND_MARK>;
/// Forwarding trace
pub type ForwardingProcessEdges<VM> =
    PlanProcessEdges<VM, Compressor<VM>, CompressorCondition<VM>, TRACE_KIND_FORWARD_ROOT>;

pub struct CompressorWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorWorkContext<VM> {
    type VM = VM;
    type PlanType = Compressor<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

pub struct CompressorForwardingWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorForwardingWorkContext<VM> {
    type VM = VM;
    type PlanType = Compressor<VM>;
    type DefaultProcessEdges = ForwardingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}
