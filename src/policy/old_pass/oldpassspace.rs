use super::forwarding;
use crate::plan::VectorObjectQueue;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::sft::GCWorkerMutRef;
use crate::policy::sft::SFT;
use crate::policy::space::{CommonSpace, Space};
use crate::scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage};
use crate::util::copy::CopySemantics;
use crate::util::heap::{MonotonePageResource, PageResource};
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::{self, ObjectEnumerator};
use crate::util::statistics::counter::EventCounter;
use crate::util::statistics::stats::Stats;
use crate::util::{Address, ObjectReference};
use crate::vm::slot::Slot;
use crate::MMTK;
use crate::{vm::*, ObjectQueue};
use atomic::Ordering;
use std::sync::{Arc, Mutex};

pub(crate) const TRACE_KIND_MARK: TraceKind = 0;
pub(crate) const TRACE_KIND_FORWARD: TraceKind = 1;

/// OldPassSpace is a stop-the-world and serial implementation of
/// the One-Pass Compactor, as described in Cory and Petrank,
/// [The One Pass (OP) Compactor: An Intellectual Abstract](https://dl.acm.org/doi/pdf/10.1145/3652024.3665513).
pub struct OldPassSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: MonotonePageResource<VM>,
    forwarding: forwarding::ForwardingMetadata<VM>,
    scheduler: Arc<GCWorkScheduler<VM>>,
}

impl<VM: VMBinding> SFT for OldPassSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        if self.forwarding.has_calculated_forwarding_addresses() {
            Some(self.forward::<false>(object, false))
        } else {
            None
        }
    }

    fn is_live(&self, object: ObjectReference) -> bool {
        Self::is_marked(object)
    }

    #[cfg(feature = "object_pinning")]
    fn pin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of OldPassSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of OldPassSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, _object: ObjectReference) -> bool {
        false
    }

    fn is_movable(&self) -> bool {
        true
    }

    fn initialize_object_metadata(&self, _object: ObjectReference) {
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(_object);
    }

    #[cfg(feature = "sanity")]
    fn is_sane(&self) -> bool {
        true
    }

    #[cfg(feature = "is_mmtk_object")]
    fn is_mmtk_object(&self, addr: Address) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::is_vo_bit_set_for_addr(addr)
    }

    #[cfg(feature = "is_mmtk_object")]
    fn find_object_from_internal_pointer(
        &self,
        ptr: Address,
        max_search_bytes: usize,
    ) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::find_object_from_internal_pointer::<VM>(
            ptr,
            max_search_bytes,
        )
    }

    fn sft_trace_object(
        &self,
        _queue: &mut VectorObjectQueue,
        _object: ObjectReference,
        _worker: GCWorkerMutRef,
    ) -> ObjectReference {
        // We should not use trace_object for onepass space.
        // Depending on which trace it is, we should manually call either trace_mark or trace_forward.
        panic!("sft_trace_object() cannot be used with one-pass space")
    }

    fn debug_print_object_info(&self, object: ObjectReference) {
        println!("marked = {}", OldPassSpace::<VM>::is_marked(object));
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for OldPassSpace<VM> {
    fn as_space(&self) -> &dyn Space<VM> {
        self
    }

    fn as_sft(&self) -> &(dyn SFT + Sync + 'static) {
        self
    }

    fn get_page_resource(&self) -> &dyn PageResource<VM> {
        &self.pr
    }

    fn maybe_get_page_resource_mut(&mut self) -> Option<&mut dyn PageResource<VM>> {
        Some(&mut self.pr)
    }

    fn common(&self) -> &CommonSpace<VM> {
        &self.common
    }

    fn initialize_sft(&self, sft_map: &mut dyn crate::policy::sft_map::SFTMap) {
        self.common().initialize_sft(self.as_sft(), sft_map)
    }

    fn release_multiple_pages(&mut self, _start: Address) {
        panic!("onepassspace only releases pages enmasse")
    }

    fn enumerate_objects(&self, enumerator: &mut dyn ObjectEnumerator) {
        object_enum::enumerate_blocks_from_monotonic_page_resource(enumerator, &self.pr);
    }

    fn clear_side_log_bits(&self) {
        unimplemented!()
    }

    fn set_side_log_bits(&self) {
        unimplemented!()
    }
}

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for OldPassSpace<VM> {
    fn trace_object<Q: ObjectQueue, const KIND: crate::policy::gc_work::TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        _copy: Option<CopySemantics>,
        _worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        debug_assert!(
            KIND != TRACE_KIND_TRANSITIVE_PIN,
            "Compressor does not support transitive pin trace."
        );
        if KIND == TRACE_KIND_MARK {
            self.trace_mark_object(queue, object)
        } else if KIND == TRACE_KIND_FORWARD {
            self.forward::<false>(object, true)
        } else {
            unreachable!()
        }
    }
    fn may_move_objects<const KIND: crate::policy::gc_work::TraceKind>() -> bool {
        if KIND == TRACE_KIND_MARK {
            false
        } else if KIND == TRACE_KIND_FORWARD {
            true
        } else {
            unreachable!()
        }
    }
}

pub struct Counters {
    threaded: Arc<Mutex<EventCounter>>,
    seen: Arc<Mutex<EventCounter>>,
}
impl Counters {
    pub fn new(stats: &Stats) -> Self {
        Self {
            threaded: stats.new_event_counter("threaded", true, true),
            seen: stats.new_event_counter("seen", true, true),
        }
    }
}

impl<VM: VMBinding> OldPassSpace<VM> {
    pub fn new(args: crate::policy::space::PlanCreateSpaceArgs<VM>) -> Self {
        let vm_map = args.vm_map;
        assert!(
            !args.vmrequest.is_discontiguous(),
            "The Compressor requires a contiguous heap"
        );
        assert!(
            VM::VMObjectModel::UNIFIED_OBJECT_REFERENCE_ADDRESS,
            "The Compressor requires a unified object reference address model"
        );
        let local_specs = extract_side_metadata(&[
            MetadataSpec::OnSide(forwarding::MARK_SPEC),
            MetadataSpec::OnSide(forwarding::OFFSET_VECTOR_SPEC),
        ]);
        let scheduler = args.scheduler.clone();
        let common = CommonSpace::new(args.into_policy_args(true, false, local_specs));
        let use_clmul = *common.options.compressor_use_clmul;
        assert!(
            *common.options.no_finalizer,
            "Finalizers should be disabled with MMTK_NO_FINALIZER=true"
        );
        assert!(
            *common.options.no_reference_types,
            "Reference types should be disabled with MMTK_NO_REFERENCE_TYPES=true"
        );
        OldPassSpace {
            pr: MonotonePageResource::new_contiguous(common.start, common.extent, vm_map),
            forwarding: forwarding::ForwardingMetadata::new(common.start, use_clmul),
            common,
            scheduler,
        }
    }

    pub fn prepare(&self) {
        for (from_start, size) in self.pr.iterate_allocated_regions() {
            forwarding::MARK_SPEC.bzero_metadata(from_start, size);
        }
    }

    pub fn release(&self) {
        self.forwarding.release();
    }

    pub fn trace_mark_object<Q: ObjectQueue>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
    ) -> ObjectReference {
        #[cfg(feature = "vo_bit")]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "{:x}: VO bit not set",
            object
        );
        if OldPassSpace::<VM>::test_and_mark(object) {
            queue.enqueue(object);
            self.forwarding.mark_last_word_of_object(object);
        }
        object
    }

    pub fn test_and_mark(object: ObjectReference) -> bool {
        forwarding::MARK_SPEC
            .fetch_update_atomic::<u8, _>(
                object.to_raw_address(),
                Ordering::SeqCst,
                Ordering::Relaxed,
                |v| {
                    if v == 0 {
                        Some(1)
                    } else {
                        None
                    }
                },
            )
            .is_ok()
    }

    pub fn is_marked(object: ObjectReference) -> bool {
        let mark_bit =
            forwarding::MARK_SPEC.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst);
        mark_bit == 1
    }

    pub fn forward<const CAN_CLMUL: bool>(
        &self,
        object: ObjectReference,
        _vo_bit_valid: bool,
    ) -> ObjectReference {
        if !self.in_space(object) {
            return object;
        }
        let addr = self
            .forwarding
            .forward::<CAN_CLMUL>(object.to_raw_address());
        let ret = ObjectReference::from_raw_address(addr).unwrap();
        // We can't expect the VO bit to be valid whilst in the compaction loop.
        // If we are fixing a reference to an object which precedes the referent
        // the VO bit will have been cleared already.
        // Thus the assertion really only is any good whilst we are fixing
        // the roots.
        #[cfg(feature = "vo_bit")]
        if _vo_bit_valid {
            debug_assert!(
                crate::util::metadata::vo_bit::is_vo_bit_set(ret),
                "{:x}: VO bit not set",
                object
            );
        }
        ret
    }

    pub fn add_remset_tasks(
        &'static self,
        remset: &crate::util::remset::RemSet<VM>,
        stage: WorkBucketStage,
    ) {
        fn inner<VM: VMBinding, const CAN_CLMUL: bool>(
            this: &'static OldPassSpace<VM>,
            remset: &crate::util::remset::RemSet<VM>,
            stage: WorkBucketStage,
        ) {
            let mut packets = vec![];
            remset.flush_all(&mut |entries| {
                let slots = entries.iter().map(|e| e.decode().0).collect();
                packets
                    .push(Box::new(UpdateSlots::<VM, CAN_CLMUL>::new(this, slots))
                        as Box<dyn GCWork<VM>>);
            });
            this.scheduler.work_buckets[stage].bulk_add(packets);
        }
        if self.forwarding.supports_clmul() {
            inner::<VM, true>(self, remset, stage)
        } else {
            inner::<VM, false>(self, remset, stage)
        }
    }

    pub fn compact_task(&'static self, counters: &'static Counters) -> Box<dyn GCWork<VM>> {
        if self.forwarding.supports_clmul() {
            Box::new(Compact::<VM, true>::new(self, counters))
        } else {
            Box::new(Compact::<VM, false>::new(self, counters))
        }
    }

    pub fn compact<const CAN_CLMUL: bool>(&self, worker: &mut GCWorker<VM>, counters: &Counters) {
        #[cfg(feature = "vo_bit")]
        {
            let start = self.forwarding.first_address;
            let end = self.pr.cursor();
            #[cfg(debug_assertions)]
            self.forwarding
                .scan_marked_objects(start, end, &mut |object: ObjectReference| {
                    debug_assert!(
                        crate::util::metadata::vo_bit::is_vo_bit_set(object),
                        "{:x}: VO bit not set",
                        object
                    );
                });
            for (region_start, size) in self.pr.iterate_allocated_regions() {
                crate::util::metadata::vo_bit::bzero_vo_bit(region_start, size);
            }
        }
        let mut seen: u64 = 0;
        let mut threaded: u64 = 0;
        let thread_references = &mut |object: ObjectReference, old: ObjectReference| {
            if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
                VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                    if let Some(o) = s.load() {
                        if self.in_space(o) {
                            seen += 1;
                            if forwarding::block_number(o) > forwarding::block_number(old) {
                                threaded += 1;
                                trace!("threading {o}");
                                VM::VMObjectModel::push_threading_list(o, s);
                            } else {
                                trace!("forwarding {o} to {}", self.forward::<CAN_CLMUL>(o, true));
                                s.store(self.forward::<CAN_CLMUL>(o, true));
                            }
                        }
                    }
                });
            } else {
                panic!("nah I've really got to look at slots here");
            }
        };
        let mut to = Address::ZERO;
        self.forwarding
            .calculate_offset_vector(&self.pr, &mut |obj: ObjectReference| {
                let new_object = self.forward::<CAN_CLMUL>(obj, false);
                debug_assert!(
                    new_object.to_raw_address() >= to,
                    "{0} < {to}",
                    new_object.to_raw_address()
                );
                // Fix threading list
                VM::VMObjectModel::walk_threading_list(obj, &mut |slot| {
                    slot.store(new_object);
                });
                VM::VMObjectModel::reset_threading_list(obj);
                // We set the end bits based on the sizes of objects when they are
                // marked, and we compute the live data and thus the forwarding
                // addresses based on those sizes. The forwarding addresses would be
                // incorrect if the sizes of objects were to change.
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                debug_assert!(copied_size == VM::VMObjectModel::get_current_size(obj));
                // Copy object
                trace!(" copy from {} to {}", obj, new_object);
                let end_of_new_object = VM::VMObjectModel::copy_to(obj, new_object, Address::ZERO);
                // Thread references
                thread_references(new_object, obj);
                // update VO bit
                #[cfg(feature = "vo_bit")]
                vo_bit::set_vo_bit(new_object);
                to = new_object.to_object_start::<VM>() + copied_size;
                debug_assert_eq!(end_of_new_object, to);
            });
        debug!("Compact end: to = {}", to);
        counters.threaded.clone().lock().unwrap().inc_by(threaded);
        counters.seen.clone().lock().unwrap().inc_by(seen);
        // Reset the bump pointer
        self.pr.reset_cursor(to);
    }

    fn update_slots<const CAN_CLMUL: bool>(&self, slots: &[VM::VMSlot]) {
        for s in slots {
            if let Some(o) = s.load() {
                trace!("forwarding {o} to {}", self.forward::<CAN_CLMUL>(o, true));
                s.store(self.forward::<CAN_CLMUL>(o, true));
            }
        }
    }
}

/// Compact live objects in the heap.
pub struct Compact<VM: VMBinding, const CAN_CLMUL: bool> {
    op_space: &'static OldPassSpace<VM>,
    counters: &'static Counters,
}

impl<VM: VMBinding, const CAN_CLMUL: bool> GCWork<VM> for Compact<VM, CAN_CLMUL> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.op_space.compact::<CAN_CLMUL>(worker, self.counters);
    }
}

impl<VM: VMBinding, const CAN_CLMUL: bool> Compact<VM, CAN_CLMUL> {
    pub fn new(op_space: &'static OldPassSpace<VM>, counters: &'static Counters) -> Self {
        Self { op_space, counters }
    }
}

/// Update references in a vector of remembered slots.
pub struct UpdateSlots<VM: VMBinding, const CAN_CLMUL: bool> {
    op_space: &'static OldPassSpace<VM>,
    slots: Vec<VM::VMSlot>,
}

impl<VM: VMBinding, const CAN_CLMUL: bool> GCWork<VM> for UpdateSlots<VM, CAN_CLMUL> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.op_space.update_slots::<CAN_CLMUL>(&self.slots);
    }
}

impl<VM: VMBinding, const CAN_CLMUL: bool> UpdateSlots<VM, CAN_CLMUL> {
    pub fn new(op_space: &'static OldPassSpace<VM>, slots: Vec<VM::VMSlot>) -> Self {
        Self { op_space, slots }
    }
}
