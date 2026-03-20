use crate::plan::concurrent::Pause;
use crate::plan::Plan;
use crate::scheduler::GCWork;
use crate::util::ObjectReference;

/// Trait for a concurrent plan.
pub trait ConcurrentPlan: Plan {
    /// Return `true`` if concurrent work (such as concurrent marking) is in progress.
    fn concurrent_work_in_progress(&self) -> bool;
    /// Return the current pause kind.  `None` if not in a pause.
    fn current_pause(&self) -> Option<Pause>;
    /// Produce a work packet for flushing a SATB buffer.
    fn satb_packet(&self, satb: Vec<ObjectReference>) -> Box<dyn GCWork<Self::VM>>;
}
