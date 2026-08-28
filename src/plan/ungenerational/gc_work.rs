use super::global::Ungenerational;
use crate::plan::tracing::{PlanTrace, UnsupportedTrace};
use crate::policy::gc_work::DEFAULT_TRACE;
use crate::vm::VMBinding;

pub struct UGGCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for UGGCWorkContext<VM> {
    type VM = VM;
    type PlanType = Ungenerational<VM>;
    type DefaultTrace = PlanTrace<Ungenerational<VM>, DEFAULT_TRACE>;
    type PinningTrace = UnsupportedTrace<VM>;
}
