use super::global::OldPass;
use crate::plan::compressor::process_edges::{PlanProcessEdgesRemset, RemsetCondition};
use crate::policy::old_pass::{TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::VMBinding;
use std::marker::PhantomData;

/// Marking trace
pub struct OldPassCondition<VM: VMBinding> {
    _p: PhantomData<VM>,
}

impl<VM: VMBinding> RemsetCondition<OldPass<VM>, VM> for OldPassCondition<VM> {
    fn relevant(plan: &OldPass<VM>, source: VM::VMSlot, target: ObjectReference) -> bool {
        !plan.op_space.address_in_space(source.as_address()) && plan.op_space.in_space(target)
    }
}

pub type MarkingProcessEdges<VM> =
    PlanProcessEdgesRemset<VM, OldPass<VM>, OldPassCondition<VM>, TRACE_KIND_MARK>;

pub struct OldPassWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for OldPassWorkContext<VM> {
    type VM = VM;
    type PlanType = OldPass<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Forwarding trace
pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, OldPass<VM>, TRACE_KIND_FORWARD>;
