use crate::plan::concurrent::compressor::global::ConcurrentCompressor;
use crate::plan::compressor::process_edges::{PlanProcessEdgesRemset, RemsetCondition};
use crate::policy::compressor::{TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::space::Space;
use crate::scheduler::gc_work::{PlanProcessEdges, UnsupportedProcessEdges};
use crate::scheduler::ProcessEdgesWork;
use crate::util::ObjectReference;
use crate::vm::VMBinding;
use crate::vm::slot::Slot;
use std::marker::PhantomData;

/// Remset tracing fluff
pub(super) type MarkingProcessEdges<VM> =
    PlanProcessEdgesRemset<VM, ConcurrentCompressor<VM>, CompressorCondition<VM>, TRACE_KIND_MARK>;

pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, ConcurrentCompressor<VM>, TRACE_KIND_FORWARD>;

pub(super) struct CompressorCondition<VM: VMBinding>(PhantomData<VM>);

impl<VM: VMBinding> RemsetCondition<ConcurrentCompressor<VM>, VM> for CompressorCondition<VM> {
    fn relevant(plan: &ConcurrentCompressor<VM>, source: VM::VMSlot, target: ObjectReference) -> bool {
        !plan.compressor_space.address_in_space(source.as_address())
            && plan.compressor_space.in_space(target)
    }
}

pub(super) struct ConcurrentCompressorSTWGCWorkContext<VM: VMBinding>(
    PhantomData<VM>,
);
impl<VM: VMBinding> crate::scheduler::GCWorkContext
    for ConcurrentCompressorSTWGCWorkContext<VM>
{
    type VM = VM;
    type PlanType = ConcurrentCompressor<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}
pub(super) struct ConcurrentCompressorGCWorkContext<E: ProcessEdgesWork>(std::marker::PhantomData<E>);

impl<E: ProcessEdgesWork> crate::scheduler::GCWorkContext for ConcurrentCompressorGCWorkContext<E> {
    type VM = E::VM;
    type PlanType = ConcurrentCompressor<E::VM>;
    type DefaultProcessEdges = E;
    type PinningProcessEdges = UnsupportedProcessEdges<Self::VM>;
}
