use super::global::OnePass;
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::policy::one_pass::{Counters, OnePassSpace};
use crate::policy::one_pass::{TRACE_KIND_FORWARD_ROOT, TRACE_KIND_MARK};
use crate::scheduler::gc_work::PlanProcessEdges;
use crate::scheduler::gc_work::*;
use crate::scheduler::GCWork;
use crate::scheduler::GCWorker;
use crate::scheduler::WorkBucketStage;
use crate::vm::ActivePlan;
use crate::vm::Scanning;
use crate::vm::VMBinding;
use crate::MMTK;
use std::marker::PhantomData;

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

        // We do two passes of transitive closures. We clear the live bytes from the first pass.
        mmtk.scheduler
            .worker_group
            .get_and_clear_worker_live_bytes();

        for mutator in VM::VMActivePlan::mutators() {
            mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots].add(ScanMutatorRoots::<
                OnePassForwardingWorkContext<VM>,
            >(mutator));
        }

        mmtk.scheduler.work_buckets[WorkBucketStage::SecondRoots]
            .add(ScanVMSpecificRoots::<OnePassForwardingWorkContext<VM>>::new());
    }
}

impl<VM: VMBinding> UpdateReferences<VM> {
    pub fn new() -> Self {
        Self { p: PhantomData }
    }
}

/// Compact live objects based on the previously-calculated forwarding pointers.
pub struct Compact<VM: VMBinding> {
    op_space: &'static OnePassSpace<VM>,
    los: &'static LargeObjectSpace<VM>,
    counters: &'static Counters,
}

impl<VM: VMBinding> GCWork<VM> for Compact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.op_space.compact(worker, self.los, self.counters);
    }
}

impl<VM: VMBinding> Compact<VM> {
    pub fn new(
        op_space: &'static OnePassSpace<VM>,
        los: &'static LargeObjectSpace<VM>,
        counters: &'static Counters,
    ) -> Self {
        Self {
            op_space,
            los,
            counters,
        }
    }
}

/// Marking trace
pub type MarkingProcessEdges<VM> = PlanProcessEdges<VM, OnePass<VM>, TRACE_KIND_MARK>;
/// Forwarding trace
pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, OnePass<VM>, TRACE_KIND_FORWARD_ROOT>;

pub struct OnePassWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for OnePassWorkContext<VM> {
    type VM = VM;
    type PlanType = OnePass<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

pub struct OnePassForwardingWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for OnePassForwardingWorkContext<VM> {
    type VM = VM;
    type PlanType = OnePass<VM>;
    type DefaultProcessEdges = ForwardingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}
