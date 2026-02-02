use super::global::Compressor;
use super::process_edges::{PlanProcessEdgesRemset, RemsetCondition};
use crate::policy::compressor::{CompressorSpace, TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWork, GCWorker};
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::VMBinding;
use crate::MMTK;
use std::marker::{PhantomData, Send};

/// Generate more packets by calling a method on [`CompressorSpace`].
pub struct GenerateWork<VM: VMBinding, F: Fn() + Send + 'static> {
    f: F,
    _p: PhantomData<VM>,
}

impl<VM: VMBinding, F: Fn() + Send + 'static> GCWork<VM> for GenerateWork<VM, F> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        (self.f)();
    }
}

impl<VM: VMBinding, F: Fn() + Send + 'static> GenerateWork<VM, F> {
    pub fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
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
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

/// Marking trace
pub struct CompressorCondition<VM: VMBinding> {
    _p: PhantomData<VM>,
}

impl<VM: VMBinding> RemsetCondition<Compressor<VM>, VM> for CompressorCondition<VM> {
    fn relevant(plan: &Compressor<VM>, source: VM::VMSlot, target: ObjectReference) -> bool {
        !plan.compressor_space.address_in_space(source.as_address())
            && plan.compressor_space.in_space(target)
    }
}

pub type MarkingProcessEdges<VM> =
    PlanProcessEdgesRemset<VM, Compressor<VM>, CompressorCondition<VM>, TRACE_KIND_MARK>;

pub struct CompressorWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for CompressorWorkContext<VM> {
    type VM = VM;
    type PlanType = Compressor<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Forwarding trace
pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, Compressor<VM>, TRACE_KIND_FORWARD>;
