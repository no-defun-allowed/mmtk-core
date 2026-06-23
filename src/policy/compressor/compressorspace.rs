use crate::plan::VectorObjectQueue;
#[cfg(feature = "object_pinning")]
use crate::policy::compressor::forwarding;
#[cfg(feature = "object_pinning")]
use crate::policy::compressor::forwarding::does_new_address_intersect_pinned_pages;
#[cfg(all(feature = "object_pinning", debug_assertions))]
use crate::policy::compressor::forwarding::is_page_pinned;
#[cfg(all(feature = "object_pinning", debug_assertions))]
use crate::policy::compressor::forwarding::{
    does_new_address_intersect_pinned_objects, COMPUTING_FORWARDING_INFO, FORWARDING_MAP,
};
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::sft::{GCWorkerMutRef, SFT};
use crate::policy::space::{CommonSpace, Space};
#[cfg(feature = "object_pinning")]
use crate::scheduler::gc_work::ScanObjects;
use crate::scheduler::GCWorkContext;
use crate::scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage};
#[cfg(feature = "object_pinning")]
use crate::util::constants::BYTES_IN_PAGE;
#[cfg(debug_assertions)]
use crate::util::constants::BYTES_IN_WORD;
use crate::util::conversions::raw_is_aligned;
use crate::util::copy::CopySemantics;
use crate::util::heap::regionpageresource::AllocatedRegion;
use crate::util::heap::{PageResource, RegionPageResource};
use crate::util::linear_scan::Region;
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::ObjectEnumerator;
use crate::util::options::{PagePinningMode, PinningMode};
#[cfg(all(feature = "object_pinning", debug_assertions))]
use crate::util::os::OSMemory;
use crate::util::{Address, ObjectReference, VMThread, VMWorkerThread};
use crate::vm::slot::Slot;
use crate::MMTK;
use crate::{vm::*, ObjectQueue};
use atomic::Ordering;
#[cfg(feature = "object_pinning")]
use std::collections::HashSet;
#[cfg(feature = "object_pinning")]
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::Arc;
#[cfg(feature = "object_pinning")]
use std::sync::RwLock;

pub(crate) const TRACE_KIND_MARK: TraceKind = 0;
pub(crate) const TRACE_KIND_FORWARD: TraceKind = 1;

#[cfg(feature = "object_pinning")]
static CACHED_PINNED_PAGES: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "object_pinning")]
static NUM_GCS: AtomicU32 = AtomicU32::new(0);

/// [`CompressorSpace`] is a stop-the-world implementation of
/// the Compressor, as described in Kermany and Petrank,
/// [The Compressor: concurrent, incremental, and parallel compaction](https://dl.acm.org/doi/10.1145/1133255.1134023).
///
/// [`CompressorSpace`] makes two main diversions from the paper
/// (aside from [`CompressorSpace`] being stop-the-world):
/// - The heap is structured into regions ([`forwarding::CompressorRegion`])
///   which the collector compacts separately.
/// - The collector compacts each region in-place, instead of using two virtual
///   spaces as in Kermany and Petrank. The virtual spaces side-step a race which
///   would occur if multiple threads attempted to compact one heap in place: one thread
///   could move an object to the location of another object which has yet to be moved by
///   another thread. Kermany and Petrank move objects between from- and to- virtual spaces,
///   preventing the old objects from being overwritten. (They reclaim memory by unmapping
///   pages of the from-virtual space after moving all objects out of said pages.)
///   We instead side-step this race by assigning only a single thread to each region, and
///   running multiple single-threaded Compressors at once.
pub struct CompressorSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: RegionPageResource<VM, forwarding::CompressorRegion>,
    forwarding: forwarding::ForwardingMetadata<VM>,
    scheduler: Arc<GCWorkScheduler<VM>>,
    #[cfg(feature = "object_pinning")]
    cached_pinned_pages: RwLock<HashSet<Address>>,
}

/// The number of bytes of the heap that each CalculateOffsetVector
/// work packet should process. Calculating the offset vector is very fast,
/// and we are often swamped by scheduling overhead when we
/// only process one region per work packet.
const OFFSET_VECTOR_PACKET_BYTES: usize = 1 << 21;

impl<VM: VMBinding> SFT for CompressorSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        // Check if forwarding addresses have been calculated before attempting
        // to forward objects
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
    fn pin_object(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.pin_object::<VM>(object)
    }

    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.unpin_object::<VM>(object)
    }

    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.is_object_pinned::<VM>(object)
    }

    fn is_movable(&self) -> bool {
        true
    }

    fn initialize_object_metadata(&self, _object: ObjectReference, _bytes: usize) {
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(_object);
    }

    #[cfg(feature = "sanity")]
    fn is_sane(&self) -> bool {
        true
    }

    #[cfg(feature = "vo_bit")]
    fn is_mmtk_object(&self, addr: Address) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::is_vo_bit_set_for_addr(addr)
    }

    #[cfg(feature = "vo_bit")]
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
        // We should not use trace_object for compressor space.
        // Depending on which trace it is, we should manually call either trace_mark or trace_forward.
        panic!("sft_trace_object() cannot be used with CompressorSpace")
    }

    fn debug_print_object_info(&self, object: ObjectReference) {
        println!("marked = {}", CompressorSpace::<VM>::is_marked(object));
        println!("forwarding = {:?}", self.get_forwarded_object(object));
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for CompressorSpace<VM> {
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
        panic!("compressorspace only releases pages enmasse")
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

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for CompressorSpace<VM> {
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

impl<VM: VMBinding> CompressorSpace<VM> {
    pub fn new(args: crate::policy::space::PlanCreateSpaceArgs<VM>) -> Self {
        let vm_map = args.vm_map;
        assert!(
            VM::VMObjectModel::UNIFIED_OBJECT_REFERENCE_ADDRESS,
            "The Compressor requires a unified object reference address model"
        );
        let local_specs = extract_side_metadata(&[
            MetadataSpec::OnSide(forwarding::MARK_SPEC),
            MetadataSpec::OnSide(forwarding::OFFSET_VECTOR_SPEC),
            MetadataSpec::OnSide(forwarding::SELECTED_SPEC),
            #[cfg(feature = "object_pinning")]
            *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC,
            #[cfg(feature = "object_pinning")]
            MetadataSpec::OnSide(forwarding::PINNED_PAGE_SPEC),
        ]);
        let is_discontiguous = args.vmrequest.is_discontiguous();
        let scheduler = args.scheduler.clone();
        let common = CommonSpace::new(args.into_policy_args(true, false, local_specs));
        let percent = *common.options.compressor_compact_max_percent;
        let use_clmul = *common.options.compressor_use_clmul;

        let pinning_mode = *common.options.compressor_pinning_mode;

        CompressorSpace {
            pr: if is_discontiguous {
                RegionPageResource::new_discontiguous(vm_map)
            } else {
                RegionPageResource::new_contiguous(common.start, common.extent, vm_map)
            },
            forwarding: forwarding::ForwardingMetadata::new(
                forwarding::CompactLimit::Percentage(percent),
                use_clmul,
                pinning_mode,
            ),
            common,
            scheduler,
            #[cfg(feature = "object_pinning")]
            cached_pinned_pages: RwLock::new(HashSet::new()),
        }
    }

    pub fn prepare<Context: GCWorkContext<VM = VM>>(&self) {
        #[cfg(feature = "object_pinning")]
        NUM_GCS.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "object_pinning")]
        let is_pinning = match self.forwarding.pinning_mode {
            PinningMode::NoPinning => false,
            _ => true,
        };
        #[cfg(feature = "object_pinning")]
        let needs_page_pinning = matches!(
            self.forwarding.pinning_mode,
            PinningMode::RandomPagePinning(PagePinningMode::EveryGC, _)
        ) || matches!(
            self.forwarding.pinning_mode,
            PinningMode::RandomPagePinning(PagePinningMode::CachedEveryGC, _)
        ) || (self
            .common()
            .global_state
            .is_harness_begin_gc
            .load(Ordering::Relaxed)
            && matches!(
                self.forwarding.pinning_mode,
                PinningMode::RandomPagePinning(PagePinningMode::FirstGC, _)
            ));
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                forwarding::MARK_SPEC
                    .bzero_metadata(r.region.start(), forwarding::CompressorRegion::BYTES);
                #[cfg(feature = "object_pinning")]
                if is_pinning {
                    match VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.as_spec() {
                        MetadataSpec::OnSide(spec) => spec
                            .bzero_metadata(r.region.start(), forwarding::CompressorRegion::BYTES),
                        MetadataSpec::InHeader(_) => {
                            panic!("Local pinning bit needs to be in side metadata")
                        }
                    };
                    // Reset the pinned page metadata if we have to pin pages this GC.
                    if needs_page_pinning {
                        forwarding::PINNED_PAGE_SPEC
                            .bzero_metadata(r.region.start(), forwarding::CompressorRegion::BYTES);
                    }
                }
            });

        #[cfg(feature = "object_pinning")]
        {
            // Pin a fraction of allocated pages at the start of GC. We will
            // individually pin live objects in these pages later.
            if needs_page_pinning {
                self.pin_pages();
                self.add_scan_pinned_pages_tasks::<Context>();
            }

            #[cfg(debug_assertions)]
            {
                let pinning_pages = matches!(
                    self.forwarding.pinning_mode,
                    PinningMode::RandomPagePinning(PagePinningMode::CachedEveryGC, _)
                ) || matches!(
                    self.forwarding.pinning_mode,
                    PinningMode::RandomPagePinning(PagePinningMode::FirstGC, _)
                );
                if pinning_pages {
                    let mut total_pages = 0;
                    let mut pages_pinned = 0;
                    self.pr.enumerate_regions(&mut |r: &AllocatedRegion<
                        forwarding::CompressorRegion,
                    >| {
                        let mut page = r.region.start();
                        let end = r.cursor();
                        while page < end {
                            use crate::policy::compressor::forwarding::is_page_pinned;
                            if is_page_pinned(page) {
                                pages_pinned += 1;
                            }
                            page += BYTES_IN_PAGE;
                            total_pages += 1;
                        }
                    });
                    if pages_pinned > 0 {
                        println!(
                            "Actually have pinned {}/{} pages ({:.2}%) = {} KB",
                            pages_pinned,
                            total_pages,
                            (pages_pinned as f64 / total_pages as f64) * 100.0,
                            pages_pinned * BYTES_IN_PAGE / 1024
                        );
                    }
                }
            }
        }
    }

    #[cfg(all(feature = "object_pinning", debug_assertions))]
    fn protect_pinned_pages(&self, read_write: bool) {
        let access = if read_write {
            crate::util::os::MmapProtection::ReadWrite
        } else {
            crate::util::os::MmapProtection::NoAccess
        };

        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                let mut page = r.region.start();
                let end = r.region.end();
                while page < end {
                    if is_page_pinned(page) {
                        crate::util::os::OS::set_memory_access(page, BYTES_IN_PAGE, access)
                            .unwrap();
                    }
                    page += BYTES_IN_PAGE;
                }
            });
    }

    fn add_scan_pinned_pages_tasks<Context: GCWorkContext<VM = VM>>(&self) {
        // SAFETY: CompressorSpace reference is always valid within this collection cycle.
        let space = unsafe { &*(self as *const Self) };
        let mut packets = vec![];
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                packets.push(Box::new(ScanPinnedPages::<VM, Context>::new(
                    space,
                    r.region,
                    r.cursor(),
                )) as Box<dyn GCWork<VM>>);
            });
        self.scheduler.work_buckets[WorkBucketStage::PinningRootsTrace].bulk_add(packets);
        #[cfg(debug_assertions)]
        self.scheduler.work_buckets[WorkBucketStage::PinningRootsTrace]
            .set_sentinel(Box::new(ProtectPinnedPages::new(space)));
    }

    fn scan_pinned_pages(
        &self,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) -> Vec<ObjectReference> {
        let start = region.start();
        let mut curr = start;
        let end = cursor;
        let mut pinned_objects = vec![];
        while curr < end {
            // SAFETY: No one will modify the VO-bits when we are scanning pinned pages
            if unsafe { vo_bit::is_vo_addr(curr) } {
                // SAFETY: This address is a valid object
                let obj = unsafe { ObjectReference::from_raw_address_unchecked(curr) };
                let obj_size = VM::VMObjectModel::get_current_size(obj);
                let (intersects_pinned_page, _) =
                    does_new_address_intersect_pinned_pages(curr, obj_size);

                if intersects_pinned_page {
                    // This object is on a pinned page, so we pin it.
                    while !forwarding::is_object_pinned::<VM>(obj) {
                        forwarding::pin_object::<VM>(obj);
                    }
                    debug_assert!(forwarding::is_object_pinned::<VM>(obj));
                    // Mark the pinned object as live.
                    if CompressorSpace::<VM>::test_and_mark(obj) {
                        // Mark the end of the object
                        self.forwarding.mark_rest_of_object(obj);
                    }

                    pinned_objects.push(obj);
                }

                // Skip to end of object
                curr += obj_size;
                debug_assert!(raw_is_aligned(curr.as_usize(), VM::MIN_ALIGNMENT));
            } else {
                curr += VM::MIN_ALIGNMENT;
            }
        }
        pinned_objects
    }

    pub fn release(&self) {
        self.forwarding.release();
        // Unprotect pinned pages
        #[cfg(all(feature = "object_pinning", debug_assertions))]
        self.protect_pinned_pages(true);
    }

    #[cfg(feature = "object_pinning")]
    fn pin_pages(&self) {
        match self.forwarding.pinning_mode {
            PinningMode::RandomPagePinning(PagePinningMode::CachedEveryGC, fraction) => {
                if fraction > 0.0 {
                    // Cache pinned pages in the harness begin GC. We pin pages till the
                    // end of the region
                    if NUM_GCS.load(Ordering::Relaxed) == 3 {
                        self.pin_random_pages(fraction, false, true);
                        CACHED_PINNED_PAGES.store(true, Ordering::Relaxed);
                    }

                    if CACHED_PINNED_PAGES.load(Ordering::Relaxed) {
                        self.pin_cached_pages();
                    }
                }
            }
            PinningMode::RandomPagePinning(page_pinning_mode, fraction) => {
                assert_ne!(page_pinning_mode, PagePinningMode::CachedEveryGC);
                if fraction > 0.0 {
                    let pin_till_end = page_pinning_mode == PagePinningMode::FirstGC;
                    self.pin_random_pages(fraction, pin_till_end, false);
                }
            }
            _ => {
                unreachable!("We should never get here");
            }
        }
    }

    /// Randomly select `fraction` of currently-allocated OS pages to pin.
    #[cfg(feature = "object_pinning")]
    fn pin_random_pages(&self, fraction: f64, pin_till_end: bool, need_to_cache: bool) {
        use rand::Rng;
        use rand::SeedableRng;

        let mut total_pages = 0;
        let mut pages_pinned = 0;
        let fraction = fraction.clamp(0.0, 1.0);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42_u64);
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                let mut page = r.region.start();
                let end = if pin_till_end {
                    r.region.end()
                } else {
                    r.cursor()
                };
                while page < end {
                    if rng.random_bool(fraction) {
                        pages_pinned += 1;
                        if !need_to_cache {
                            forwarding::PINNED_PAGE_SPEC.store_atomic(page, 1_u8, Ordering::SeqCst);
                        } else {
                            self.cached_pinned_pages.write().unwrap().insert(page);
                        }
                        info!("Pinning page {}", page);
                    }
                    page += BYTES_IN_PAGE;
                    total_pages += 1;
                }
            });
        info!(
            "Pinned {}/{} pages ({:.2}%) = {} KB",
            pages_pinned,
            total_pages,
            (pages_pinned as f64 / total_pages as f64) * 100.0,
            pages_pinned * (BYTES_IN_PAGE / 1024),
        );
    }

    /// Pin the pages in `cached_pinned_pages`. Only called when using [`PagePinningMode::CachedEveryGC`].
    #[cfg(feature = "object_pinning")]
    fn pin_cached_pages(&self) {
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                let mut page = r.region.start();
                let end = r.cursor();
                let cached_pinned_pages = self.cached_pinned_pages.read().unwrap();
                while page < end {
                    if cached_pinned_pages.contains(&page) {
                        forwarding::PINNED_PAGE_SPEC.store_atomic(page, 1_u8, Ordering::SeqCst);
                        info!("Pinning cached page {}", page);
                    }
                    page += BYTES_IN_PAGE;
                }
            });
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
        if CompressorSpace::<VM>::test_and_mark(object) {
            queue.enqueue(object);
            self.forwarding.mark_rest_of_object(object);
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

    fn generate_tasks(
        &self,
        f: &mut impl FnMut(&AllocatedRegion<forwarding::CompressorRegion>, usize) -> Box<dyn GCWork<VM>>,
    ) -> Vec<Box<dyn GCWork<VM>>> {
        let mut packets = vec![];
        let mut index = 0;
        self.pr.enumerate_regions(&mut |r| {
            packets.push(f(r, index));
            index += 1;
        });
        packets
    }

    /// Check if an object is pinned.
    #[allow(unused)]
    fn is_pinned(&self, _object: ObjectReference) -> bool {
        #[cfg(feature = "object_pinning")]
        return self.is_object_pinned(_object);

        #[cfg(not(feature = "object_pinning"))]
        false
    }

    pub fn add_offset_vector_tasks(&'static self) {
        let mut regions = vec![];
        self.pr.enumerate_regions(&mut |r| {
            regions.push((r.region, r.cursor()));
        });
        let offset_vector_packets: Vec<Box<dyn GCWork<VM>>> = regions
            .chunks(OFFSET_VECTOR_PACKET_BYTES / forwarding::CompressorRegion::BYTES)
            .map(|c| {
                Box::new(CalculateOffsetVector::<VM>::new(self, c.to_vec())) as Box<dyn GCWork<VM>>
            })
            .collect();
        self.scheduler.work_buckets[WorkBucketStage::CalculateForwarding]
            .bulk_add(offset_vector_packets);
        #[cfg(debug_assertions)]
        self.scheduler.work_buckets[WorkBucketStage::CalculateForwarding]
            .set_sentinel(Box::new(AfterCalculateOffsetVector::new(self)));
    }

    pub fn calculate_offset_vector_for_region(
        &self,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) {
        self.forwarding.calculate_offset_vector(region, cursor);
    }

    pub fn forward<const CAN_CLMUL: bool>(
        &self,
        object: ObjectReference,
        _vo_bit_valid: bool,
    ) -> ObjectReference {
        if !self.in_space(object) {
            return object;
        }

        // We can't expect the VO bit to be valid whilst compacting the heap.
        // If we are fixing a reference to an object which was moved before the referent,
        // the relevant VO bit will have been cleared, and this assertion would fail.
        // Thus we can only ever expect the VO bit to be valid whilst fixing the roots.
        #[cfg(feature = "vo_bit")]
        if _vo_bit_valid {
            debug_assert!(
                crate::util::metadata::vo_bit::is_vo_bit_set(object),
                "{:x}: VO bit not set",
                object
            );
        }
        let to = self
            .forwarding
            .forward::<CAN_CLMUL>(object.to_raw_address());
        ObjectReference::from_raw_address(to).unwrap()
    }

    pub fn update_references<const CAN_CLMUL: bool>(
        &self,
        worker: &mut GCWorker<VM>,
        object: ObjectReference,
    ) {
        #[cfg(feature = "vo_bit")]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "{:?}: VO bit not set",
            object
        );
        #[cfg(feature = "object_pinning")]
        if self.is_pinned(object) {
            let size = forwarding::get_object_size_from_mark_bits(object.to_object_start::<VM>());
            let (intersect_pinned_page, _) =
                does_new_address_intersect_pinned_pages(object.to_raw_address(), size);
            if intersect_pinned_page {
                // We don't need to update references in an object that
                // intersects a pinned page because we have already pinned the
                // directly reachable objects and hence we have no reason to
                // update references.
                return;
            }
        }

        if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
            VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                if let Some(o) = s.load() {
                    s.store(self.forward::<CAN_CLMUL>(o, false));
                }
            });
        } else {
            VM::VMScanning::scan_object_and_trace_edges(worker.tls, object, &mut |o| {
                self.forward::<CAN_CLMUL>(o, false)
            });
        }
    }

    pub fn add_remset_tasks(
        &'static self,
        remset: &crate::util::remset::RemSet<VM>,
        stage: WorkBucketStage,
    ) {
        fn inner<VM: VMBinding, const CAN_CLMUL: bool>(
            this: &'static CompressorSpace<VM>,
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

    pub fn add_compact_tasks(&'static self) {
        fn inner<VM: VMBinding, const CAN_CLMUL: bool>(this: &'static CompressorSpace<VM>) {
            let compact_packets: Vec<Box<dyn GCWork<VM>>> =
                this.generate_tasks(&mut |_, i| Box::new(Compact::<VM, CAN_CLMUL>::new(this, i)));
            this.scheduler.work_buckets[WorkBucketStage::Compact].bulk_add(compact_packets);
        }
        if self.forwarding.supports_clmul() {
            inner::<VM, true>(self)
        } else {
            inner::<VM, false>(self)
        }
    }

    pub fn compact_region<const CAN_CLMUL: bool>(&self, worker: &mut GCWorker<VM>, index: usize) {
        self.pr.with_regions(&mut |regions| {
            let r = &regions[index];
            let start = r.region.start();
            debug!("\nCompacting region {}", start);
            let end = r.cursor();
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
            }
            let mut to = r.region.start();
            let mut objects = 0;
            let mut total_live_bytes = 0;
            let mut total_copied_bytes = 0;
            if self.forwarding.is_forwarding_region(r.region) {
                #[cfg(feature = "vo_bit")]
                crate::util::metadata::vo_bit::bzero_vo_bit(start, end - start);
                self.forwarding
                    .scan_marked_objects(start, end, &mut |obj: ObjectReference| {
                        objects += 1;
                        // We set the end bits based on the sizes of objects when they are
                        // marked, and we compute the live data and thus the forwarding
                        // addresses based on those sizes. The forwarding addresses would be
                        // incorrect if the sizes of objects were to change.
                        let copied_size = if self.is_pinned(obj) {
                            forwarding::get_object_size_from_mark_bits(obj.to_object_start::<VM>())
                        } else {
                            VM::VMObjectModel::get_size_when_copied(obj)
                        };
                        // debug_assert!(copied_size == VM::VMObjectModel::get_current_size(obj));
                        let new_object = self.forward::<CAN_CLMUL>(obj, false);
                        assert!(
                            obj.to_object_start::<VM>() >= new_object.to_object_start::<VM>(),
                            "Object {obj} was forwarded to {new_object} which is after it, potentially overwriting data!",
                        );
                        #[cfg(debug_assertions)]
                        {
                            if !self.is_pinned(obj) {
                                let (intersects_pinned, pinned_object) = does_new_address_intersect_pinned_objects::<VM>(new_object.to_raw_address(), copied_size);
                                debug_assert!(
                                    !intersects_pinned,
                                    "Moving object {obj} -> {} (size {copied_size}) intersects with pinned object {:?}",
                                    new_object.to_raw_address(),
                                    pinned_object,
                                );
                                let (intersects_pinned_page, pinned_page) = does_new_address_intersect_pinned_pages(new_object.to_raw_address(), copied_size);
                                debug_assert!(
                                    !intersects_pinned_page,
                                    "Moving object {obj} -> {} (size {copied_size}) intersects with pinned page {:?}",
                                    new_object.to_raw_address(),
                                    pinned_page,
                                );
                            } else {
                                debug_assert_eq!(obj, new_object, "Pinned object {obj} was forwarded to {new_object}!");
                            }
                        }
                        // copy object
                        trace!("copy from {} to {}", obj, new_object);
                        if obj != new_object {
                            VM::VMObjectModel::copy_to(obj, new_object, Address::ZERO);
                            total_copied_bytes += copied_size;
                        }
                        total_live_bytes += copied_size;
                        // update VO bit
                        #[cfg(feature = "vo_bit")]
                        vo_bit::set_vo_bit(new_object);
                        to = to.max(new_object.to_object_start::<VM>() + copied_size);
                        self.update_references::<CAN_CLMUL>(worker, new_object);
                    });
                debug_assert!(to <= r.cursor());
                info!(
                    "Compacted region [{}, {}) -> {to} with {objects} objects; saved {} bytes (copied {} bytes; live {} bytes)",
                    r.region.start(), r.cursor(), r.cursor() - to, total_copied_bytes, total_live_bytes,
                );
                self.pr.reset_cursor(r, to);
            } else {
                self.forwarding.scan_marked_objects(start, end, &mut |obj: ObjectReference| {
                    self.update_references::<CAN_CLMUL>(worker, obj);
                });
            }
        });
    }

    pub fn update_slots<const CAN_CLMUL: bool>(&self, slots: &[VM::VMSlot]) {
        debug!("\nUpdating {} slots in remset", slots.len());
        for s in slots {
            if let Some(o) = s.load() {
                trace!("Forwarding {o} -> {}", self.forward::<false>(o, false));
                s.store(self.forward::<CAN_CLMUL>(o, false));
            }
        }
    }

    pub fn after_compact(&self) {
        self.pr.reset_allocator();
        self.pr.with_regions(&mut |r| draw_region_usage(r));

        #[cfg(debug_assertions)]
        {
            let map = FORWARDING_MAP.lock().unwrap();
            map.iter().for_each(|(from_obj, to_obj)| {
                let from_obj = ObjectReference::from_raw_address(*from_obj).unwrap();
                let to_obj = ObjectReference::from_raw_address(*to_obj).unwrap();
                #[cfg(feature = "vo_bit")]
                debug_assert!(
                    vo_bit::is_vo_bit_set(to_obj),
                    "Forwarded object {:?} -> {:?} does not have VO bit set after compaction!",
                    from_obj,
                    to_obj,
                );
                if self.is_object_pinned(from_obj) {
                    debug_assert_eq!(
                        from_obj, to_obj,
                        "Pinned object {:?} was forwarded to {:?}!",
                        from_obj, to_obj
                    );
                }
            });
        }
    }
}

#[cfg(debug_assertions)]
pub struct AfterCalculateOffsetVector<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(debug_assertions)]
impl<VM: VMBinding> AfterCalculateOffsetVector<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

#[cfg(debug_assertions)]
impl<VM: VMBinding> GCWork<VM> for AfterCalculateOffsetVector<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        {
            let map = FORWARDING_MAP.lock().unwrap();
            map.iter().for_each(|(from_obj, to_obj)| {
                let from_obj = ObjectReference::from_raw_address(*from_obj).unwrap();
                let to_obj = ObjectReference::from_raw_address(*to_obj).unwrap();
                #[cfg(feature = "vo_bit")]
                debug_assert!(
                    crate::util::metadata::vo_bit::is_vo_bit_set(from_obj),
                    "{:?}: VO bit not set",
                    from_obj,
                );
                if self.compressor_space.is_object_pinned(from_obj) {
                    debug_assert_eq!(
                        from_obj, to_obj,
                        "Pinned object {:?} was forwarded to {:?}!",
                        from_obj, to_obj
                    );
                } else {
                    let size = VM::VMObjectModel::get_size_when_copied(from_obj);
                    let (intersects_pinned, pinned_object) =
                        does_new_address_intersect_pinned_objects::<VM>(
                            to_obj.to_raw_address(),
                            size,
                        );
                    debug_assert!(
                        !intersects_pinned,
                        "Moving object {:?} -> {:?} intersects with pinned object {:?}!",
                        from_obj, to_obj, pinned_object,
                    );
                    if let PinningMode::RandomPagePinning(..) = self.compressor_space.forwarding.pinning_mode {
                        let to_obj_start = to_obj.to_object_start::<VM>();
                        let to_obj_end = to_obj_start
                            + size
                            - BYTES_IN_WORD;
                        let (intersects_pinned_page, pinned_page) =
                             does_new_address_intersect_pinned_pages(from_obj.to_object_start::<VM>(), size);
                        debug_assert!(
                            !intersects_pinned_page,
                            "Object {:?} (size {}) intersects pinned page {} but was not pinned!",
                            from_obj, size, pinned_page.unwrap(),
                        );
                        let (intersects_pinned_page, pinned_page) =
                             does_new_address_intersect_pinned_pages(to_obj_start, size);
                        debug_assert!(
                            !intersects_pinned_page,
                            "Object {:?} is forwarded to [{}, {}), which intersects with pinned page {}!",
                            from_obj, to_obj_start, to_obj_end, pinned_page.unwrap(),
                        );
                    }
                }
            });
        }

        COMPUTING_FORWARDING_INFO.store(false, Ordering::SeqCst);
    }
}

#[cfg(feature = "object_pinning")]
pub struct ScanPinnedPages<VM: VMBinding, Context: GCWorkContext<VM = VM>> {
    compressor_space: &'static CompressorSpace<VM>,
    region: forwarding::CompressorRegion,
    cursor: Address,
    phantom: std::marker::PhantomData<Context>,
}

#[cfg(feature = "object_pinning")]
impl<VM: VMBinding, Context: GCWorkContext<VM = VM>> ScanPinnedPages<VM, Context> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        region: forwarding::CompressorRegion,
        cursor: Address,
    ) -> Self {
        Self {
            compressor_space,
            region,
            cursor,
            phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "object_pinning")]
impl<VM: VMBinding, Context: GCWorkContext<VM = VM>> GCWork<VM> for ScanPinnedPages<VM, Context> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let pinned_objects = self
            .compressor_space
            .scan_pinned_pages(self.region, self.cursor);

        let mut closure = |slot: VM::VMSlot| {
            let Some(object) = slot.load() else { return };
            while !forwarding::is_object_pinned::<VM>(object) {
                forwarding::pin_object::<VM>(object);
                debug!(
                    "Pinning reachable object at {:?} of size {} bytes",
                    object.to_raw_address(),
                    VM::VMObjectModel::get_current_size(object),
                );
            }
            debug_assert!(
                forwarding::is_object_pinned::<VM>(object),
                "Directly reachable object {:?} should be pinned",
                object
            );
        };

        pinned_objects.iter().for_each(|obj| {
            debug_assert!(
                forwarding::is_object_pinned::<VM>(*obj),
                "Object {:?} should be pinned",
                obj
            );
            VM::VMScanning::scan_object(
                VMWorkerThread(VMThread::UNINITIALIZED),
                *obj,
                &mut closure,
            );
        });

        self.compressor_space.scheduler.work_buckets[WorkBucketStage::PinningRootsTrace].add_boxed(
            Box::new(ScanObjects::<Context::DefaultProcessEdges>::new(
                pinned_objects,
                false,
                WorkBucketStage::PinningRootsTrace,
            )),
        );
    }
}

#[cfg(all(feature = "object_pinning", debug_assertions))]
struct ProtectPinnedPages<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(all(feature = "object_pinning", debug_assertions))]
impl<VM: VMBinding> ProtectPinnedPages<VM> {
    fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

#[cfg(all(feature = "object_pinning", debug_assertions))]
impl<VM: VMBinding> GCWork<VM> for ProtectPinnedPages<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        // Protect pinned pages for the duration of the GC so that we don't touch them accidentally
        // TODO(kunals): Reference processing may need to read from pinned pages.
        // We disable reference processing for now
        self.compressor_space.protect_pinned_pages(false);
    }
}

/// Calculate the offset vector for a region.
pub struct CalculateOffsetVector<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
    regions: Vec<(forwarding::CompressorRegion, Address)>,
}

impl<VM: VMBinding> GCWork<VM> for CalculateOffsetVector<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        for (region, cursor) in self.regions.iter() {
            self.compressor_space
                .calculate_offset_vector_for_region(*region, *cursor);
        }
    }
}

pub(crate) fn draw_region_usage(regions: &[AllocatedRegion<forwarding::CompressorRegion>]) {
    if log::log_enabled!(log::Level::Info) {
        regions
            .chunks(64)
            .map(|c| {
                c.iter().map(|r| {
                    let scale = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
                    let used = r.cursor() - r.region.start();
                    let index = (used * (scale.len() - 1)) / forwarding::CompressorRegion::BYTES;
                    scale[index]
                })
            })
            .for_each(|c| info!("Region usage: {}", c.collect::<String>()));
    }
}

impl<VM: VMBinding> CalculateOffsetVector<VM> {
    pub fn new(
        compressor_space: &'static CompressorSpace<VM>,
        regions: Vec<(forwarding::CompressorRegion, Address)>,
    ) -> Self {
        Self {
            compressor_space,
            regions,
        }
    }
}

/// Compact live objects in a region.
pub struct Compact<VM: VMBinding, const CAN_CLMUL: bool> {
    compressor_space: &'static CompressorSpace<VM>,
    index: usize,
}

impl<VM: VMBinding, const CAN_CLMUL: bool> GCWork<VM> for Compact<VM, CAN_CLMUL> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space
            .compact_region::<CAN_CLMUL>(worker, self.index);
    }
}

impl<VM: VMBinding, const CAN_CLMUL: bool> Compact<VM, CAN_CLMUL> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>, index: usize) -> Self {
        Self {
            compressor_space,
            index,
        }
    }
}

/// Update references in a vector of remembered slots.
pub struct UpdateSlots<VM: VMBinding, const CAN_CLMUL: bool> {
    compressor_space: &'static CompressorSpace<VM>,
    slots: Vec<VM::VMSlot>,
}

impl<VM: VMBinding, const CAN_CLMUL: bool> GCWork<VM> for UpdateSlots<VM, CAN_CLMUL> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.compressor_space.update_slots::<CAN_CLMUL>(&self.slots);
    }
}

impl<VM: VMBinding, const CAN_CLMUL: bool> UpdateSlots<VM, CAN_CLMUL> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>, slots: Vec<VM::VMSlot>) -> Self {
        Self {
            compressor_space,
            slots,
        }
    }
}
