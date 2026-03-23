use crate::plan::concurrent::compressor::global::ConcurrentCompressor;
use crate::policy::compressor::{CompressorSpace, TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::scheduler::gc_work::*;
use crate::scheduler::ProcessEdgesWork;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::util::object_enum::ClosureObjectEnumerator;
use crate::util::ObjectReference;
use crate::vm::{ActivePlan, Scanning, VMBinding};
use crate::MMTK;
use std::marker::PhantomData;

pub(super) type MarkingProcessEdges<VM> =
    PlanProcessEdges<VM, ConcurrentCompressor<VM>, TRACE_KIND_MARK>;

pub(super) type ForwardingProcessEdges<VM> =
    PlanProcessEdges<VM, ConcurrentCompressor<VM>, TRACE_KIND_FORWARD>;

/// Create another round of root scanning work packets
/// to update root references.
pub(super) struct UpdateRoots<VM: VMBinding>(PhantomData<VM>);

unsafe impl<VM: VMBinding> Send for UpdateRoots<VM> {}

impl<VM: VMBinding> GCWork<VM> for UpdateRoots<VM> {
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

impl<VM: VMBinding> UpdateRoots<VM> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// Fix references from a LOS.
pub(super) struct UpdateLOS<VM: VMBinding> {
    space: &'static CompressorSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
}

impl<VM: VMBinding> UpdateLOS<VM> {
    pub fn new(space: &'static CompressorSpace<VM>, los: &'static LargeObjectSpace<VM>) -> Self {
        Self { space, los }
    }
}

impl<VM: VMBinding> GCWork<VM> for UpdateLOS<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.los
            .enumerate_to_space_objects(&mut ClosureObjectEnumerator::<_, VM>::new(
                &mut |o: ObjectReference| {
                    self.space.update_references::<false>(worker, o);
                },
            ));
    }
}

/// The STW trace.
pub(super) struct ConcurrentCompressorSTWGCWorkContext<VM: VMBinding>(PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for ConcurrentCompressorSTWGCWorkContext<VM> {
    type VM = VM;
    type PlanType = ConcurrentCompressor<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}
pub(super) struct ConcurrentCompressorGCWorkContext<E: ProcessEdgesWork>(
    std::marker::PhantomData<E>,
);

/// The root fixing "trace".
pub struct CompressorForwardingWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorForwardingWorkContext<VM> {
    type VM = VM;
    type PlanType = ConcurrentCompressor<VM>;
    type DefaultProcessEdges = ForwardingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// The actually concurrent trace!
impl<E: ProcessEdgesWork> crate::scheduler::GCWorkContext for ConcurrentCompressorGCWorkContext<E> {
    type VM = E::VM;
    type PlanType = ConcurrentCompressor<E::VM>;
    type DefaultProcessEdges = E;
    type PinningProcessEdges = UnsupportedProcessEdges<Self::VM>;
}
