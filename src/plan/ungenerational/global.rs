use super::gc_work::UGGCWorkContext;
use super::mutator::ALLOCATOR_MAPPING;
use crate::plan::global::CommonPlan;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::AllocationSemantics;
use crate::plan::Plan;
use crate::plan::PlanConstraints;
use crate::policy::copyspace::CopySpace;
use crate::policy::space::Space;
use crate::scheduler::*;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::copy::*;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::heap::VMRequest;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::opaque_pointer::VMWorkerThread;
use crate::{plan::global::BasePlan, vm::VMBinding};

use mmtk_macros::{HasSpaces, PlanTraceObject};

use enum_map::EnumMap;

#[cfg(feature = "malloc_mark_sweep")]
pub type MarkSweepSpace<VM> = crate::policy::marksweepspace::malloc_ms::MallocSpace<VM>;
#[cfg(feature = "malloc_mark_sweep")]
use crate::policy::marksweepspace::malloc_ms::MAX_OBJECT_SIZE;

#[cfg(not(feature = "malloc_mark_sweep"))]
pub type MarkSweepSpace<VM> = crate::policy::marksweepspace::native_ms::MarkSweepSpace<VM>;
#[cfg(not(feature = "malloc_mark_sweep"))]
use crate::policy::marksweepspace::native_ms::MAX_OBJECT_SIZE;

#[derive(HasSpaces, PlanTraceObject)]
pub struct Ungenerational<VM: VMBinding> {
    #[space]
    #[copy_semantics(CopySemantics::DefaultCopy)]
    pub copy: CopySpace<VM>,
    #[space]
    pub ms: MarkSweepSpace<VM>,
    #[parent]
    pub common: CommonPlan<VM>,
}

/// The plan constraints for the ungenerational plan.
pub const UG_CONSTRAINTS: PlanConstraints = PlanConstraints {
    moves_objects: true,
    max_non_los_default_alloc_bytes: MAX_OBJECT_SIZE,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for Ungenerational<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &UG_CONSTRAINTS
    }

    fn create_copy_config(&'static self) -> CopyConfig<Self::VM> {
        use enum_map::enum_map;
        CopyConfig {
            // XXX: need to copy into ms
            copy_mapping: enum_map! {
                CopySemantics::DefaultCopy => CopySelector::MarkSweep(0),
                _ => CopySelector::Unused,
            },
            space_mapping: vec![
                (CopySelector::MarkSweep(0), &self.ms),
            ],
            constraints: &UG_CONSTRAINTS,
        }
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        scheduler.schedule_common_work::<UGGCWorkContext<VM>>(self);
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        self.common.prepare(tls, true);
        self.copy.prepare(true); // copyspace is always from-space
        self.ms.prepare(true);
        self.copy.set_copy_for_sft_trace(Some(CopySemantics::DefaultCopy));
        info!("prepare: {} pages in copy, {} pages in ms", self.copy.reserved_pages(), self.ms.reserved_pages())
    }

    fn prepare_worker(&self, _worker: &mut GCWorker<VM>) {}

    fn release(&mut self, tls: VMWorkerThread) {
        self.common.release(tls, true);
        self.copy.release();
        self.ms.release();
        info!("release: {} pages in copy, {} pages in ms", self.copy.reserved_pages(), self.ms.reserved_pages())
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        self.base().collection_required(self, space_full)
    }

    fn current_gc_may_move_object(&self) -> bool {
        true
    }

    fn get_collection_reserved_pages(&self) -> usize {
        self.copy.reserved_pages()
    }
    
    fn get_used_pages(&self) -> usize {
        self.copy.reserved_pages() + self.ms.reserved_pages() + self.common.get_used_pages()
    }

    fn get_available_pages(&self) -> usize {
        (self
            .get_total_pages()
            .saturating_sub(self.get_reserved_pages()))
            >> 1
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.common.base
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.common
    }

    fn common_mut(&mut self) -> &mut CommonPlan<VM> {
        &mut self.common
    }
}

impl<VM: VMBinding> Ungenerational<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &UG_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&[]),
        };

        Self {
            copy: CopySpace::new(
                plan_args.get_normal_space_args(
                    "copy",
                    true,
                    false,
                    VMRequest::discontiguous(),
                ),
                false,
            ),
            ms: MarkSweepSpace::new(
                plan_args.get_normal_space_args(
                    "ms",
                    true,
                    false,
                    VMRequest::discontiguous(),
                ),
            ),
            common: CommonPlan::new(plan_args),
        }
    }
}
