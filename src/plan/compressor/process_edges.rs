use crate::mmtk::MMTK;
use crate::plan::{Plan, PlanTraceObject, VectorObjectQueue};
use crate::policy::gc_work::TraceKind;
use crate::scheduler::gc_work::{PlanScanObjects, ProcessEdgesBase, SlotOf};
use crate::scheduler::{GCWorker, ProcessEdgesWork, WorkBucketStage};
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::VMBinding;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub trait RemsetCondition<P: Plan<VM = VM>, VM: VMBinding>: Send {
    fn relevant(plan: &P, source: VM::VMSlot, target: ObjectReference) -> bool;
}

pub trait PlanRemember<VM: VMBinding> {
    fn record(&self, source: VM::VMSlot, target: ObjectReference, worker: &GCWorker<VM>);
}

/// This provides an implementation of [`crate::scheduler::gc_work::ProcessEdgesWork`]. A plan that implements
/// `PlanTraceObject` can use this work packet for tracing objects.
pub struct PlanProcessEdgesRemset<
    VM: VMBinding,
    P: Plan<VM = VM> + PlanTraceObject<VM> + PlanRemember<VM>,
    C: RemsetCondition<P, VM>,
    const KIND: TraceKind,
> {
    plan: &'static P,
    base: ProcessEdgesBase<VM>,
    _c: PhantomData<C>,
}

impl<
        VM: VMBinding,
        P: PlanTraceObject<VM> + Plan<VM = VM> + PlanRemember<VM>,
        C: RemsetCondition<P, VM> + 'static,
        const KIND: TraceKind,
    > ProcessEdgesWork for PlanProcessEdgesRemset<VM, P, C, KIND>
{
    type VM = VM;
    type ScanObjectsWorkType = PlanScanObjects<Self, P>;

    fn new(
        slots: Vec<SlotOf<Self>>,
        roots: bool,
        mmtk: &'static MMTK<VM>,
        bucket: WorkBucketStage,
    ) -> Self {
        let base = ProcessEdgesBase::new(slots, roots, mmtk, bucket);
        let plan = base.plan().downcast_ref::<P>().unwrap();
        Self {
            plan,
            base,
            _c: PhantomData,
        }
    }

    fn create_scan_work(&self, nodes: Vec<ObjectReference>) -> Option<Self::ScanObjectsWorkType> {
        Some(PlanScanObjects::<Self, P>::new(
            self.plan,
            nodes,
            false,
            self.bucket,
        ))
    }

    fn trace_object(&mut self, object: ObjectReference) -> ObjectReference {
        // We cannot borrow `self` twice in a call, so we extract `worker` as a local variable.
        let worker = self.worker();
        self.plan
            .trace_object::<VectorObjectQueue, KIND>(&mut self.base.nodes, object, worker)
    }

    fn process_slot(&mut self, slot: SlotOf<Self>) {
        let Some(object) = slot.load() else {
            // Skip slots that are not holding an object reference.
            return;
        };
        if C::relevant(self.plan, slot, object) {
            trace!("Recording {:x} -> {:x}", slot.as_address(), object);
            let worker = self.worker();
            self.plan.record(slot, object, worker);
        }
        let new_object = self.trace_object(object);
        if P::may_move_objects::<KIND>() && new_object != object {
            slot.store(new_object);
        }
    }
}

// Impl Deref/DerefMut to ProcessEdgesBase for PlanProcessEdgesRemset
impl<
        VM: VMBinding,
        P: PlanTraceObject<VM> + Plan<VM = VM> + PlanRemember<VM>,
        C: RemsetCondition<P, VM>,
        const KIND: TraceKind,
    > Deref for PlanProcessEdgesRemset<VM, P, C, KIND>
{
    type Target = ProcessEdgesBase<VM>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<
        VM: VMBinding,
        P: PlanTraceObject<VM> + Plan<VM = VM> + PlanRemember<VM>,
        C: RemsetCondition<P, VM>,
        const KIND: TraceKind,
    > DerefMut for PlanProcessEdgesRemset<VM, P, C, KIND>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
