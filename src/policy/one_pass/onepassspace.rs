use crate::plan::VectorObjectQueue;
use crate::policy::compressor::forwarding;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::largeobjectspace::LargeObjectSpace;
use crate::policy::one_pass::locking;
use crate::policy::sft::GCWorkerMutRef;
use crate::policy::sft::SFT;
use crate::policy::space::{CommonSpace, Space};
use crate::scheduler::GCWorkScheduler;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::util::copy::CopySemantics;
use crate::util::heap::regionpageresource::AllocatedRegion;
use crate::util::heap::{PageResource, RegionPageResource};
use crate::util::linear_scan::Region;
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
pub(crate) const TRACE_KIND_FORWARD_ROOT: TraceKind = 1;

/// OnePassSpace is a stop-the-world and serial implementation of
/// the One-Pass Compactor, as described in Cory and Petrank,
/// [The One Pass (OP) Compactor: An Intellectual Abstract](https://dl.acm.org/doi/pdf/10.1145/3652024.3665513).
pub struct OnePassSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: RegionPageResource<VM, forwarding::CompressorRegion>,
    forwarding: forwarding::ForwardingMetadata<VM>,
    scheduler: Arc<GCWorkScheduler<VM>>,
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
        self.pr.enumerate(enumerator);
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

pub struct Counters {
    threaded: Arc<Mutex<EventCounter>>,
    seen: Arc<Mutex<EventCounter>>,
    #[cfg(feature = "distances")]
    distances: Vec<Arc<Mutex<EventCounter>>>,
}
impl Counters {
    pub fn new(stats: &Stats) -> Self {
        Self {
            threaded: stats.new_event_counter("threaded", true, true),
            seen: stats.new_event_counter("seen", true, true),
            #[cfg(feature = "distances")]
            distances: (0..48)
                .map(|v| stats.new_event_counter(&format!("distance.{v}"), true, true))
                .collect(),
        }
    }
}

impl<VM: VMBinding> OnePassSpace<VM> {
    pub fn new(args: crate::policy::space::PlanCreateSpaceArgs<VM>) -> Self {
        let vm_map = args.vm_map;
        assert!(
            VM::VMObjectModel::UNIFIED_OBJECT_REFERENCE_ADDRESS,
            "The One Pass Compactor requires a unified object reference address model"
        );
        let local_specs = extract_side_metadata(&[
            MetadataSpec::OnSide(forwarding::MARK_SPEC),
            MetadataSpec::OnSide(forwarding::OFFSET_VECTOR_SPEC),
            MetadataSpec::OnSide(forwarding::SELECTED_SPEC),
            MetadataSpec::OnSide(locking::STATUS_SPEC),
        ]);
        let is_discontiguous = args.vmrequest.is_discontiguous();
        let scheduler = args.scheduler.clone();
        let common = CommonSpace::new(args.into_policy_args(true, false, local_specs));
        assert!(scheduler.num_workers() <= locking::Status::MAX_WORKERS);
        OnePassSpace {
            pr: if is_discontiguous {
                RegionPageResource::new_discontiguous(vm_map)
            } else {
                RegionPageResource::new_contiguous(common.start, common.extent, vm_map)
            },
            forwarding: forwarding::ForwardingMetadata::new(
                forwarding::CompactLimit::AlwaysCompact,
            ),
            common,
            scheduler,
        }
    }

    pub fn prepare(&self) {
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                forwarding::MARK_SPEC
                    .bzero_metadata(r.region.start(), r.region.end() - r.region.start());
                self.forwarding.select_region(r.region);
                locking::reset_metadata(r.region.start(), r.region.end() - r.region.start());
            });
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
        let ret =
            ObjectReference::from_raw_address(self.forwarding.forward(object.to_raw_address()))
                .unwrap();
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

    pub fn add_compact_tasks(&'static self, counters: &'static Counters) {
        let packets: Vec<Box<dyn GCWork<VM>>> = self.pr.with_regions(&mut |r| {
            (0..r.len())
                .map(|i| Box::new(Compact::<VM>::new(self, i, counters)) as Box<dyn GCWork<VM>>)
                .collect()
        });
        self.scheduler.work_buckets[WorkBucketStage::CalculateForwarding].bulk_add(packets);
    }

    pub fn compact_region(&self, worker: &mut GCWorker<VM>, index: usize, counters: &Counters) {
        let mut seen: u64 = 0;
        let mut threaded: u64 = 0;
        #[cfg(feature = "distances")]
        let mut local_counters: Vec<u64> = counters.distances.iter().map(|_| 0).collect();
        let thread_references = &mut |object: ObjectReference| {
            if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
                VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                    if let Some(target) = s.load() {
                        if self.in_space(target) {
                            seen += 1;
                            locking::thread_or_forward(target, &mut |action| match action {
                                locking::ThreadOrForward::Thread => {
                                    threaded += 1;
                                    trace!("threading {target}");
                                    #[cfg(feature = "distances")]
                                    {
                                        let bits = (target.to_raw_address().as_usize()
                                            ^ s.as_address().as_usize())
                                        .ilog2();
                                        local_counters[bits as usize] += 1;
                                    }
                                    VM::VMObjectModel::push_threading_list(target, s);
                                }
                                locking::ThreadOrForward::Forward => {
                                    trace!("forwarding {target} to {}", self.forward(target, true));
                                    s.store(self.forward(target, true));
                                }
                            });
                        }
                    }
                })
            } else {
                panic!("nah I've really got to look at slots here");
            }
        };

        self.pr.with_regions(&mut |regions| {
            let r = &regions[index];
            #[cfg(feature = "vo_bit")]
            {
                #[cfg(debug_assertions)]
                self.forwarding.scan_marked_objects(
                    r.region.start(),
                    r.cursor(),
                    &mut |object: ObjectReference| {
                        debug_assert!(
                            crate::util::metadata::vo_bit::is_vo_bit_set(object),
                            "{:x}: VO bit not set",
                            object
                        );
                    },
                );
            }

            let mut to = r.region.start();
            trace!("forwarding region {:?}", r.region.start());
            self.forwarding.calculate_and_walk_offset_vector(
                r.region,
                r.cursor(),
                &mut |b, f| locking::lock_for_forwarding(b, f),
                &mut |obj: ObjectReference| {
                    let new_object = self.forward(obj, false);
                    while let Some(slot) = VM::VMObjectModel::pop_threading_list(obj) {
                        slot.store(new_object);
                    }
                    // We set the end bits based on the sizes of objects when they are
                    // marked, and we compute the live data and thus the forwarding
                    // addresses based on those sizes. The forwarding addresses would be
                    // incorrect if the sizes of objects were to change.
                    let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                    debug_assert!(copied_size == VM::VMObjectModel::get_current_size(obj));
                    debug_assert!(
                        new_object.to_raw_address() >= to,
                        "{0} < {to}",
                        new_object.to_raw_address()
                    );
                    // copy object
                    trace!(" copy from {} to {}", obj, new_object);
                    let end_of_new_object =
                        VM::VMObjectModel::copy_to(obj, new_object, Address::ZERO);
                    // update VO bit
                    #[cfg(feature = "vo_bit")]
                    vo_bit::set_vo_bit(new_object);
                    to = new_object.to_object_start::<VM>() + copied_size;
                    debug_assert_eq!(end_of_new_object, to);
                    thread_references(new_object);
                },
            );
            debug!("Compact end: to = {}", to);
            self.pr.reset_cursor(r, to);
        });

        counters.threaded.clone().lock().unwrap().inc_by(threaded);
        counters.seen.clone().lock().unwrap().inc_by(seen);
        #[cfg(feature = "distances")]
        for (local, global) in std::iter::zip(local_counters.iter(), counters.distances.iter()) {
            global.clone().lock().unwrap().inc_by(*local);
        }
    }

    pub fn after_compact(&self, worker: &mut GCWorker<VM>, los: &LargeObjectSpace<VM>) {
        self.pr.reset_allocator();
        // Update references from the LOS to OnePassSpace too.
        los.enumerate_objects(&mut object_enum::ClosureObjectEnumerator::<_, VM>::new(
            &mut |object: ObjectReference| {
                if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
                    VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                        if let Some(o) = s.load() {
                            trace!("forwarding {o} to {}", self.forward(o, true));
                            s.store(self.forward(o, true));
                        }
                    });
                } else {
                    panic!("nah I've really got to look at slots here");
                }
            },
        ));
    }
}

/// Compact live objects in a region.
pub struct Compact<VM: VMBinding> {
    one_pass_space: &'static OnePassSpace<VM>,
    index: usize,
    counters: &'static Counters,
}

impl<VM: VMBinding> GCWork<VM> for Compact<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.one_pass_space
            .compact_region(worker, self.index, self.counters);
    }
}

impl<VM: VMBinding> Compact<VM> {
    pub fn new(
        one_pass_space: &'static OnePassSpace<VM>,
        index: usize,
        counters: &'static Counters,
    ) -> Self {
        Self {
            one_pass_space,
            index,
            counters,
        }
    }
}
