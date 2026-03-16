use crate::plan::concurrent::concurrent_marking_work::ProcessRootSlots;
use crate::plan::concurrent::global::ConcurrentPlan;
use crate::plan::concurrent::Pause;
use crate::plan::global::{BasePlan, CommonPlan, CreateGeneralPlanArgs, CreateSpecificPlanArgs};
use crate::plan::compressor::mutator::ALLOCATOR_MAPPING;
use crate::plan::{AllocationSemantics, Plan, PlanConstraints};
use crate::policy::space::Space;
use crate::scheduler::gc_work::{Release, StopMutators, UnsupportedProcessEdges, VMProcessWeakRefs};
use crate::scheduler::*;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::copy::*;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::heap::VMRequest;
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::vm::{ObjectModel, VMBinding};
use crate::{policy::compressor::CompressorSpace, util::opaque_pointer::VMWorkerThread};
use std::sync::atomic::AtomicBool;
use super::gc_work::{ConcurrentImmixGCWorkContext, ConcurrentImmixSTWGCWorkContext};

use atomic::Atomic;
use atomic::Ordering;
use enum_map::EnumMap;

use mmtk_macros::{HasSpaces, PlanTraceObject};

/// A concurrent Compressor plan. The plan supports concurrent marking but pauses for compaction.
/// The concurrent GC consists of two STW pauses (initial mark and final mark+compaction) with concurrent marking in between.
#[derive(HasSpaces, PlanTraceObject)]
pub struct ConcurrentCompressor<VM: VMBinding> {
    #[parent]
    pub common: CommonPlan<VM>,
    #[space]
    pub compressor_space: CompressorSpace<VM>,
    remset: Remset<VM>,
    current_pause: Atomic<Option<Pause>>,
    previous_pause: Atomic<Option<Pause>>,
    should_do_full_gc: AtomicBool,
    concurrent_marking_active: AtomicBool,
}

pub const CONCURRENT_COMPRESSOR_CONSTRAINTS: PlanConstraints = PlanConstraints {
    max_non_los_default_alloc_bytes: MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
    moves_objects: true,
    needs_forward_after_liveness: true,
    needs_prepare_mutator: true,
    barrier: crate::BarrierSelector::SATBBarrier,
    needs_log_bit: true,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for ConcurrentCompressor<VM> {
    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        if self.base().collection_required(self, space_full) {
            self.should_do_full_gc.store(true, Ordering::Release);
            info!("Triggering full GC");
            return true;
        }

        let concurrent_marking_in_progress = self.concurrent_marking_in_progress();

        if concurrent_marking_in_progress
            && self.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent].is_drained()
        {
            // After the Concurrent bucket is drained during concurrent marking,
            // we trigger the FinalMark pause at the next poll() site (here).
            // FIXME: Immediately trigger FinalMark when the Concurrent bucket is drained.
            return true;
        }

        let threshold = self.get_total_pages() >> 1;
        let used_pages_after_last_gc = self.common.base.global_state.get_used_pages_after_last_gc();
        let used_pages_now = self.get_used_pages();
        let allocated = used_pages_now.saturating_sub(used_pages_after_last_gc);
        if !concurrent_marking_in_progress && allocated > threshold {
            info!("Allocated {allocated} pages since last GC ({used_pages_now} - {used_pages_after_last_gc} > {threshold}): Do concurrent marking");
            debug_assert!(
                self.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent].is_empty()
            );
            debug_assert!(!self.concurrent_marking_in_progress());
            debug_assert_ne!(self.previous_pause(), Some(Pause::InitialMark));
            true
        } else {
            false
        }
    }

    fn constraints(&self) -> &'static PlanConstraints {
        &CONCURRENT_COMPRESSOR_CONSTRAINTS
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        let pause = if self.concurrent_marking_in_progress() {
            // FIXME: Currently it is unsafe to bypass `FinalMark` and go directly from `InitialMark` to `Full`.
            // It is related to defragmentation.  See https://github.com/mmtk/mmtk-core/issues/1357 for more details.
            // We currently force `FinalMark` to happen if the last pause is `InitialMark`.
            Pause::FinalMark
        } else if self.should_do_full_gc.load(Ordering::SeqCst) {
            Pause::Full
        } else {
            Pause::InitialMark
        };

        self.current_pause.store(Some(pause), Ordering::SeqCst);

        probe!(mmtk, concurrent_pause_determined, pause as usize);

        match pause {
            Pause::Full => {
                // Ref closure buckets is disabled by initial mark, and needs to be re-enabled for full GC before
                // we reuse the normal Immix scheduling.
                self.set_ref_closure_buckets_enabled(true);
                crate::plan::immix::global::Immix::schedule_immix_full_heap_collection::<
                    ConcurrentImmix<VM>,
                    ConcurrentImmixSTWGCWorkContext<VM, TRACE_KIND_FAST>,
                    ConcurrentImmixSTWGCWorkContext<VM, TRACE_KIND_DEFRAG>,
                >(self, &self.immix_space, scheduler);
            }
            Pause::InitialMark => self.schedule_concurrent_marking_initial_pause(scheduler),
            Pause::FinalMark => self.schedule_concurrent_marking_final_pause(scheduler),
        }
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();
        match pause {
            Pause::Full => {
                self.common.prepare(tls, true);
                self.compressor_space.prepare();
            }
            Pause::InitialMark => {
                self.compressor_space.prepare();
                unimplemented!("bulk set log bits");
                self.common.prepare(tls, true);
                // Bulk set log bits so SATB barrier will be triggered on the existing objects.
                self.common
                    .schedule_unlog_bits_op(UnlogBitsOperation::BulkSet);
            }
            Pause::FinalMark => (),
        }
    }

    fn release(&mut self, tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();
        match pause {
            Pause::InitialMark => (),
            Pause::Full | Pause::FinalMark => {
                self.compressor_space.release();
                unimplemented!("bulk clear log bits");

                self.common.release(tls, true);

                if pause == Pause::FinalMark {
                    // Bulk clear log bits so SATB barrier will not be triggered.
                    self.common
                        .schedule_unlog_bits_op(UnlogBitsOperation::BulkClear);
                } else {
                    // Full pauses didn't set unlog bits in the first place,
                    // so there is no need to clear them.
                    // TODO: Currently InitialMark must be followed by a FinalMark.
                    // If we allow upgrading a concurrent GC to a full STW GC,
                    // we will need to clear the unlog bits at an appropriate place.
                }
            }
        }
    }

    fn end_of_gc(&mut self, _tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();
        if pause == Pause::InitialMark {
            self.set_concurrent_marking_state(true);
        }
        self.previous_pause.store(Some(pause), Ordering::SeqCst);
        self.current_pause.store(None, Ordering::SeqCst);
        if pause != Pause::FinalMark {
            self.should_do_full_gc.store(false, Ordering::SeqCst);
        } else {
            // FIXME: Currently it is unsafe to trigger full GC during concurrent marking.
            // See `Self::schedule_collection`.
            // We keep the value of `self.should_do_full_gc` so that if full GC is triggered,
            // the next GC will be full GC.
        }
        info!("{:?} end", pause);
    }

    fn current_gc_may_move_object(&self) -> bool {
        true
    }

    fn get_used_pages(&self) -> usize {
        self.compressor_space.reserved_pages() + self.common.get_used_pages()
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

    fn notify_mutators_paused(&self, _scheduler: &GCWorkScheduler<VM>) {
        use crate::vm::ActivePlan;
        let pause = self.current_pause().unwrap();
        match pause {
            Pause::Full => {
                self.set_concurrent_marking_state(false);
            }
            Pause::InitialMark => {
                debug_assert!(
                    !self.concurrent_marking_in_progress(),
                    "prev pause: {:?}",
                    self.previous_pause().unwrap()
                );
            }
            Pause::FinalMark => {
                debug_assert!(self.concurrent_marking_in_progress());
                // Flush barrier buffers
                for mutator in <VM as VMBinding>::VMActivePlan::mutators() {
                    mutator.barrier.flush();
                }
                self.set_concurrent_marking_state(false);
            }
        }
        info!("{:?} start", pause);
    }

    fn concurrent(&self) -> Option<&dyn ConcurrentPlan<VM = VM>> {
        Some(self)
    }
}

impl<VM: VMBinding> ConcurrentCompressor<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let spec = crate::util::metadata::extract_side_metadata(&[
            *VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC,
        ]);

        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &CONCURRENT_IMMIX_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&spec),
        };

        // These buckets are not used in an Immix plan. We can simply disable them.
        // TODO: We should be more systmatic on this, and disable unnecessary buckets for other plans as well.
        let scheduler = &plan_args.global_args.scheduler;
        /*
        scheduler.work_buckets[WorkBucketStage::VMRefForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::CalculateForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::SecondRoots].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::RefForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::FinalizableForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::Compact].set_enabled(false);
        */

        let res = ConcurrentCompressor {
            CompressorSpace::new(plan_args.get_normal_space_args(
                "compressor_space",
                true,
                false,
                VMRequest::discontiguous(),
            )),
            common: CommonPlan::new(plan_args),
            remset: RemSet::new(scheduler.num_workers()),
            last_gc_was_defrag: AtomicBool::new(false),
            current_pause: Atomic::new(None),
            previous_pause: Atomic::new(None),
            should_do_full_gc: AtomicBool::new(false),
            concurrent_marking_active: AtomicBool::new(false),
        };

        res.verify_side_metadata_sanity();

        res
    }

    fn set_ref_closure_buckets_enabled(&self, do_closure: bool) {
        let scheduler = &self.common.base.scheduler;
        scheduler.work_buckets[WorkBucketStage::VMRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::WeakRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::FinalRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::SoftRefClosure].set_enabled(do_closure);
        scheduler.work_buckets[WorkBucketStage::PhantomRefClosure].set_enabled(do_closure);
    }

    pub(crate) fn schedule_concurrent_marking_initial_pause(
        &'static self,
        scheduler: &GCWorkScheduler<VM>,
    ) {
        use crate::scheduler::gc_work::Prepare;

        self.set_ref_closure_buckets_enabled(false);

        scheduler.work_buckets[WorkBucketStage::Unconstrained].add(StopMutators::<
            ConcurrentImmixGCWorkContext<ProcessRootSlots<VM, Self, TRACE_KIND_FAST>>,
        >::new());
        scheduler.work_buckets[WorkBucketStage::Prepare].add(Prepare::<
            ConcurrentImmixGCWorkContext<UnsupportedProcessEdges<VM>>,
        >::new(self));
    }

    fn schedule_concurrent_marking_final_pause(&'static self, scheduler: &GCWorkScheduler<VM>) {
        self.set_ref_closure_buckets_enabled(true);

        // Skip root scanning in the final mark
        scheduler.work_buckets[WorkBucketStage::Unconstrained].add(StopMutators::<
            ConcurrentImmixGCWorkContext<ProcessRootSlots<VM, Self, TRACE_KIND_FAST>>,
        >::new_no_scan_roots());

        scheduler.work_buckets[WorkBucketStage::Release].add(Release::<
            ConcurrentImmixGCWorkContext<UnsupportedProcessEdges<VM>>,
        >::new(self));

        // Deal with weak ref and finalizers
        // TODO: Check against schedule_common_work and see if we are still missing any work packet
        type RefProcessingEdges<VM> =
            crate::scheduler::gc_work::PlanProcessEdges<VM, ConcurrentImmix<VM>, TRACE_KIND_FAST>;
        // Reference processing
        if !*self.base().options.no_reference_types {
            use crate::util::reference_processor::{
                PhantomRefProcessing, SoftRefProcessing, WeakRefProcessing,
            };
            scheduler.work_buckets[WorkBucketStage::SoftRefClosure]
                .add(SoftRefProcessing::<RefProcessingEdges<VM>>::new());
            scheduler.work_buckets[WorkBucketStage::WeakRefClosure]
                .add(WeakRefProcessing::<VM>::new());
            scheduler.work_buckets[WorkBucketStage::PhantomRefClosure]
                .add(PhantomRefProcessing::<VM>::new());

            use crate::util::reference_processor::RefEnqueue;
            scheduler.work_buckets[WorkBucketStage::Release].add(RefEnqueue::<VM>::new());
        }

        // Finalization
        if !*self.base().options.no_finalizer {
            use crate::util::finalizable_processor::Finalization;
            // finalization
            scheduler.work_buckets[WorkBucketStage::FinalRefClosure]
                .add(Finalization::<RefProcessingEdges<VM>>::new());
        }

        // VM-specific weak ref processing
        // Note that ConcurrentImmix does not have a separate forwarding stage,
        // so we don't schedule the `VMForwardWeakRefs` work packet.
        scheduler.work_buckets[WorkBucketStage::VMRefClosure]
            .set_sentinel(Box::new(VMProcessWeakRefs::<RefProcessingEdges<VM>>::new()));
    }

    pub fn concurrent_marking_in_progress(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::Acquire)
    }

    fn set_concurrent_marking_state(&self, active: bool) {
        use crate::plan::global::HasSpaces;

        // Tell the spaces to allocate new objects as live
        let allocate_object_as_live = active;
        self.for_each_space(&mut |space: &dyn Space<VM>| {
            space.set_allocate_as_live(allocate_object_as_live);
        });

        // Store the state.
        self.concurrent_marking_active
            .store(active, Ordering::SeqCst);

        // We also set SATB barrier as active -- this is done in Mutator prepare/release.
    }

    pub(super) fn is_concurrent_marking_active(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::SeqCst)
    }

    fn previous_pause(&self) -> Option<Pause> {
        self.previous_pause.load(Ordering::SeqCst)
    }
}

impl<VM: VMBinding> ConcurrentPlan for ConcurrentImmix<VM> {
    fn current_pause(&self) -> Option<Pause> {
        self.current_pause.load(Ordering::SeqCst)
    }

    fn concurrent_work_in_progress(&self) -> bool {
        self.concurrent_marking_in_progress()
    }
}
