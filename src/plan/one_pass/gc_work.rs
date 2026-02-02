use super::global::OnePass;
use crate::plan::compressor::process_edges::{PlanProcessEdgesRemset, RemsetCondition};
use crate::policy::one_pass::{OnePassSpace, TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWork, GCWorker};
use crate::util::ObjectReference;
use crate::vm::VMBinding;
use crate::vm::slot::Slot;
use crate::MMTK;
use std::marker::PhantomData;

/// Reset the allocator and update references in large object space.
pub struct AfterCompact<VM: VMBinding> {
    one_pass_space: &'static OnePassSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for AfterCompact<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.one_pass_space.after_compact();
    }
}

impl<VM: VMBinding> AfterCompact<VM> {
    pub fn new(one_pass_space: &'static OnePassSpace<VM>) -> Self {
        Self { one_pass_space }
    }
}

/// Marking trace
pub struct OnePassCondition<VM: VMBinding> {
    _p: PhantomData<VM>,
}

impl<VM: VMBinding> RemsetCondition<OnePass<VM>, VM> for OnePassCondition<VM> {
    fn relevant(plan: &OnePass<VM>, source: VM::VMSlot, target: ObjectReference) -> bool {
        !plan.op_space.address_in_space(source.as_address())
            && plan.op_space.in_space(target)
    }
}

pub type MarkingProcessEdges<VM> =
    PlanProcessEdgesRemset<VM, OnePass<VM>, OnePassCondition<VM>, TRACE_KIND_MARK>;

pub struct OnePassWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for OnePassWorkContext<VM> {
    type VM = VM;
    type PlanType = OnePass<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Forwarding trace
pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, OnePass<VM>, TRACE_KIND_FORWARD>;
