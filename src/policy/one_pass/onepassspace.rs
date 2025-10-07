use crate::plan::VectorObjectQueue;
use crate::policy::one_pass::forwarding;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::policy::sft::GCWorkerMutRef;
use crate::policy::sft::SFT;
use crate::policy::space::{CommonSpace, Space};
use crate::scheduler::GCWorker;
use crate::util::copy::CopySemantics;
use crate::util::heap::{MonotonePageResource, PageResource};
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::{self, ObjectEnumerator};
use crate::util::{Address, ObjectReference};
use crate::vm::slot::Slot;
use crate::{vm::*, ObjectQueue};
use atomic::Ordering;
use std::collections::HashMap;

pub(crate) const TRACE_KIND_MARK: TraceKind = 0;
pub(crate) const TRACE_KIND_FORWARD_ROOT: TraceKind = 1;

/// OnePassSpace is a stop-the-world and serial implementation of
/// the One-Pass Compactor, as described in Cory and Petrank,
/// [The One Pass (OP) Compactor: An Intellectual Abstract](https://dl.acm.org/doi/pdf/10.1145/3652024.3665513).
pub struct OnePassSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: MonotonePageResource<VM>,
    forwarding: forwarding::ForwardingMetadata<VM>,
}

pub(crate) const GC_MARK_BIT_MASK: u8 = 1;

impl<VM: VMBinding> SFT for OnePassSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        if self.forwarding.has_calculated_forwarding_addresses() {
            Some(self.forward(object, false))
        } else {
            None
        }
    }

    fn is_live(&self, object: ObjectReference) -> bool {
        Self::is_marked(object)
    }

    #[cfg(feature = "object_pinning")]
    fn pin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of OnePassSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of OnePassSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, _object: ObjectReference) -> bool {
        false
    }

    fn is_movable(&self) -> bool {
        true
    }

    fn initialize_object_metadata(&self, _object: ObjectReference, _alloc: bool) {
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
        println!("marked = {}", OnePassSpace::<VM>::is_marked(object));
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for OnePassSpace<VM> {
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

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for OnePassSpace<VM> {
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
        } else if KIND == TRACE_KIND_FORWARD_ROOT {
            self.trace_forward_root(queue, object)
        } else {
            unreachable!()
        }
    }
    fn may_move_objects<const KIND: crate::policy::gc_work::TraceKind>() -> bool {
        if KIND == TRACE_KIND_MARK {
            false
        } else if KIND == TRACE_KIND_FORWARD_ROOT {
            true
        } else {
            unreachable!()
        }
    }
}

impl<VM: VMBinding> OnePassSpace<VM> {
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
        let common = CommonSpace::new(args.into_policy_args(true, false, local_specs));

        OnePassSpace {
            pr: MonotonePageResource::new_contiguous(common.start, common.extent, vm_map),
            forwarding: forwarding::ForwardingMetadata::new(common.start),
            common,
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
        if OnePassSpace::<VM>::test_and_mark(object) {
            queue.enqueue(object);
            self.forwarding.mark_last_word_of_object(object);
        }
        object
    }

    pub fn trace_forward_root<Q: ObjectQueue>(
        &self,
        _queue: &mut Q,
        object: ObjectReference,
    ) -> ObjectReference {
        self.forward(object, true)
    }

    pub fn test_and_mark(object: ObjectReference) -> bool {
        let old = forwarding::MARK_SPEC.fetch_or_atomic(
            object.to_raw_address(),
            GC_MARK_BIT_MASK,
            Ordering::SeqCst,
        );
        (old & GC_MARK_BIT_MASK) == 0
    }

    pub fn is_marked(object: ObjectReference) -> bool {
        let old_value =
            forwarding::MARK_SPEC.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst);
        let mark_bit = old_value & GC_MARK_BIT_MASK;
        mark_bit != 0
    }

    pub fn forward(&self, object: ObjectReference, _vo_bit_valid: bool) -> ObjectReference {
        if !self.in_space(object) {
            return object;
        }
        // We can't expect the VO bit to be valid whilst in the compaction loop.
        // If we are fixing a reference to an object which precedes the referent
        // the VO bit will have been cleared already.
        // Thus the assertion really only is any good whilst we are fixing
        // the roots.
        #[cfg(feature = "vo_bit")]
        if _vo_bit_valid {
            debug_assert!(
                crate::util::metadata::vo_bit::is_vo_bit_set(object),
                "{:x}: VO bit not set",
                object
            );
        }
        ObjectReference::from_raw_address(self.forwarding.forward(object.to_raw_address())).unwrap()
    }

    pub fn compact(&self, worker: &mut GCWorker<VM>, los: &LargeObjectSpace<VM>) {
        #[cfg(feature = "vo_bit")]
        {
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
        type SlotTable<VM> = HashMap<ObjectReference, Vec<<VM as VMBinding>::VMSlot>>;
        let mut to = Address::ZERO;
        let add_forwards_slot = &mut |forwards_slots: &mut SlotTable<VM>, object: ObjectReference, s: VM::VMSlot| {
            if let Option::Some(v) = forwards_slots.get_mut(&object) {
                v.push(s);
            } else {
                forwards_slots.insert(object, vec![s]);
            }
        };
        let update_references = &mut |forwards_slots: &mut SlotTable<VM>, object: ObjectReference| {
            if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
                VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                    if let Some(o) = s.load() {
                        if o <= object {
                            s.store(self.forward(o, false));
                        } else {
                            add_forwards_slot(forwards_slots, o, s);
                        }
                    }
                });
            } else {
                panic!("nah I've really got to look at slots here");
            }
        };
        let mut forwards_slots: SlotTable<VM> = HashMap::new();
        self.forwarding
            .calculate_offset_vector(&self.pr, &mut |obj: ObjectReference| {
                // We set the end bits based on the sizes of objects when they are
                // marked, and we compute the live data and thus the forwarding
                // addresses based on those sizes. The forwarding addresses would be
                // incorrect if the sizes of objects were to change.
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                debug_assert!(copied_size == VM::VMObjectModel::get_current_size(obj));
                let new_object = self.forward(obj, false);
                debug_assert!(
                    new_object.to_raw_address() >= to,
                    "{0} < {to}",
                    new_object.to_raw_address()
                );
                // copy object
                trace!(" copy from {} to {}", obj, new_object);
                let end_of_new_object = VM::VMObjectModel::copy_to(obj, new_object, Address::ZERO);
                // update VO bit
                #[cfg(feature = "vo_bit")]
                vo_bit::set_vo_bit(new_object);
                to = new_object.to_object_start::<VM>() + copied_size;
                debug_assert_eq!(end_of_new_object, to);
                update_references(&mut forwards_slots, new_object);
                if let Option::Some(v) = forwards_slots.get(&obj) {
                    for slot in v {
                        slot.store(new_object)
                    }
                }
            });
        // Update references from the LOS to Compressor too.
        los.enumerate_objects(&mut object_enum::ClosureObjectEnumerator::<_, VM>::new(
            &mut |o| update_references(&mut forwards_slots, o),
        ));
        debug!("Compact end: to = {}", to);
        // reset the bump pointer
        self.pr.reset_cursor(to);
    }
}
