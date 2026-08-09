use crate::plan::VectorObjectQueue;
#[cfg(feature = "object_pinning")]
use crate::policy::compressor::forwarding;
#[cfg(feature = "object_pinning")]
use crate::policy::compressor::forwarding::does_new_address_intersect_pinned_pages;
#[cfg(all(feature = "object_pinning", debug_assertions))]
use crate::policy::compressor::forwarding::is_page_pinned;
#[cfg(feature = "object_pinning")]
use crate::policy::compressor::forwarding::Block;
#[cfg(all(feature = "object_pinning", debug_assertions))]
use crate::policy::compressor::forwarding::{
    does_new_address_intersect_pinned_objects, COMPUTING_FORWARDING_INFO, FORWARDING_MAP,
};
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::sft::{GCWorkerMutRef, SFT};
use crate::policy::space::{CommonSpace, Space};
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
#[cfg(feature = "object_pinning")]
use crate::util::linear_scan::RegionIterator;
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::ObjectEnumerator;
use crate::util::options::{PagePinningMode, PinningMode};
#[cfg(all(feature = "object_pinning", debug_assertions))]
use crate::util::os::OSMemory;
use crate::util::{Address, ObjectReference};
use crate::vm::slot::Slot;
use crate::{vm::*, ObjectQueue};
use crate::{AllocationSemantics, MMTK};
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
pub(super) static NUM_GCS: AtomicU32 = AtomicU32::new(0);

/// Pins a fixed fraction of all pages (pin_rate), but spends that budget
/// disproportionately on mature pages.
///
/// Parameters:
///   pin_rate:    desired fraction of ALL pages to pin  (e.g. 0.25)
///   bias:        P(pin|mature) / P(pin|nursery)        (e.g. 3.0 → mature 3× more likely)
///   mature_frac: fraction of the page set that is mature (needed to normalize)
///
/// Derivation:
///   p_n × (mature_frac × bias + nursery_frac) = pin_rate
///   p_m = bias × p_n
struct PinRng {
    state: u64,
    mature_threshold: u32,
    nursery_threshold: u32,
}

impl PinRng {
    fn new(seed: u64, pin_rate: f64, bias: f64, mature_frac: f64) -> Self {
        assert!(bias >= 1.0, "bias must be >= 1.0");
        assert!((0.0..=1.0).contains(&pin_rate));
        assert!((0.0..=1.0).contains(&mature_frac));
        let nursery_frac = 1.0 - mature_frac;
        let p_nursery = pin_rate / (mature_frac * bias + nursery_frac);
        let mut p_mature = bias * p_nursery;
        if p_mature > 1.0 {
            warn!("bias {bias} too high for mature_frac {mature_frac}: p_mature={p_mature:.3} > 1");
            p_mature = 1.0;
        }
        Self {
            state: seed.wrapping_add(0x9e3779b97f4a7c15),
            mature_threshold: (p_mature * u32::MAX as f64) as u32,
            nursery_threshold: (p_nursery * u32::MAX as f64) as u32,
        }
    }

    fn next_u32(&mut self) -> u32 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (z ^ (z >> 31)) as u32
    }

    pub fn should_pin(&mut self, is_mature: bool) -> bool {
        let t = if is_mature {
            self.mature_threshold
        } else {
            self.nursery_threshold
        };
        self.next_u32() < t
    }
}

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
    num_pinned_pages: AtomicU32,
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
        forwarding::pin_object::<VM>(object)
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
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        debug_assert!(
            KIND != TRACE_KIND_TRANSITIVE_PIN,
            "Compressor does not support transitive pin trace."
        );
        if KIND == TRACE_KIND_MARK {
            self.trace_mark_object(queue, object, worker)
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
            num_pinned_pages: AtomicU32::new(0),
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

        #[cfg(feature = "object_pinning")]
        if is_pinning {
            let mut stub_table = self.forwarding.stub_table.write().unwrap();
            stub_table.clear();
        }

        #[cfg(feature = "object_pinning")]
        let mut total_nursery = 0_usize;
        #[cfg(feature = "object_pinning")]
        let mut total_allocated = 0_usize;

        let mut default_bytes = 0_usize;
        let mut ref_bytes = 0_usize;
        let mut non_ref_bytes = 0_usize;
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                forwarding::MARK_SPEC
                    .bzero_metadata(r.region.start(), forwarding::CompressorRegion::BYTES);
                match r.semantics {
                    AllocationSemantics::Default => default_bytes += r.cursor() - r.region.start(),
                    AllocationSemantics::ReferenceArray => {
                        ref_bytes += r.cursor() - r.region.start()
                    }
                    AllocationSemantics::PrimitiveArray => {
                        non_ref_bytes += r.cursor() - r.region.start()
                    }
                    _ => unreachable!("Unsupported allocation semantics: {:?}", r.semantics),
                }
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
                    if *self
                        .common()
                        .options
                        .compressor_check_candidate_before_pinning
                    {
                        let mut page = r.region.start();
                        let end = r.cursor();
                        while page < end {
                            use crate::util::metadata::vo_bit::find_object_from_internal_pointer;

                            // Does an object live on this page?
                            let mut pinning_candidate = find_object_from_internal_pointer::<VM>(
                                page + BYTES_IN_PAGE - 1,
                                BYTES_IN_PAGE,
                            )
                            .is_some();

                            if !pinning_candidate {
                                use crate::plan::MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN;

                                // Does an object span into this page?
                                let potential_spanning_object =
                                    find_object_from_internal_pointer::<VM>(
                                        page,
                                        MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
                                    );

                                if let Some(potential_spanning_object) = potential_spanning_object {
                                    let size = VM::VMObjectModel::get_current_size(
                                        potential_spanning_object,
                                    );
                                    pinning_candidate =
                                        potential_spanning_object.to_object_start::<VM>() + size
                                            > page;
                                }
                            }

                            if pinning_candidate {
                                let is_mature = page <= r.prev_cursor();
                                trace!(
                                    "Page {} is a {} pinning candidate",
                                    page,
                                    if is_mature { "mature" } else { "nursery" }
                                );
                                if !is_mature {
                                    total_nursery += BYTES_IN_PAGE;
                                }
                                total_allocated += BYTES_IN_PAGE;
                                // Set the pinning bit for this page. We use this bit to figure out pinning
                                // candidates in `Self::pin_random_pages`. We don't want to pin pages that
                                // are not pinning candidates
                                forwarding::PINNED_PAGE_SPEC.store_atomic(
                                    page,
                                    1_u8,
                                    Ordering::SeqCst,
                                );
                            }

                            page += BYTES_IN_PAGE;
                        }
                    } else {
                        total_nursery += r.cursor() - r.prev_cursor();
                        total_allocated += r.cursor() - r.region.start();
                    }
                }
            });

        if *self
            .common()
            .options
            .compressor_print_region_semantics_stats
        {
            use std::io::Write;

            let filename: &str = &self.common().options.compressor_region_semantics_stats_file;
            let total_bytes = default_bytes + ref_bytes + non_ref_bytes;
            let mut metadata_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(filename)
                .unwrap();

            let epoch = super::NUM_GCS.load(Ordering::SeqCst);
            writeln!(metadata_file, "GC epoch {}:", epoch).unwrap();
            writeln!(
                metadata_file,
                "  Total pages: {} KB;\n  Default pages: {:.2}% ({} KB)\n  Reference pages: {:.2}% ({} KB)\n  Non-reference pages: {:.2}% ({} KB)",
                total_bytes / 1024,
                (default_bytes as f64 / total_bytes as f64) * 100.0,
                default_bytes / 1024,
                (ref_bytes as f64 / total_bytes as f64) * 100.0,
                ref_bytes / 1024,
                (non_ref_bytes as f64 / total_bytes as f64) * 100.0,
                non_ref_bytes / 1024
            ).unwrap();
        }

        #[cfg(feature = "object_pinning")]
        {
            // Pin a fraction of allocated pages at the start of GC. We will
            // individually pin live objects in these pages later.
            if needs_page_pinning {
                let mature_fraction =
                    (total_allocated - total_nursery) as f64 / total_allocated as f64;
                self.pin_pages(mature_fraction);
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
        self.scheduler.work_buckets[WorkBucketStage::PinningRootsTrace]
            .set_sentinel(Box::new(AfterScanPinnedPages::new(space)));
    }

    fn scan_pinned_pages(&self, region: forwarding::CompressorRegion, cursor: Address) {
        let start = region.start();
        let mut curr = start;
        let end = cursor;
        while curr < end {
            // SAFETY: No one will modify the VO-bits when we are scanning pinned pages
            if unsafe { vo_bit::is_vo_addr(curr) } {
                // SAFETY: This address is a valid object
                let obj = unsafe { ObjectReference::from_raw_address_unchecked(curr) };
                let obj_size = VM::VMObjectModel::get_current_size(obj);
                let (intersects_pinned_page, _) =
                    does_new_address_intersect_pinned_pages(curr, obj_size);

                let block = forwarding::Block::from_unaligned_address(curr);
                if intersects_pinned_page {
                    self.forwarding.stub_table.write().unwrap().add_stub(obj);
                    if forwarding::pin_block(block) {
                        info!(
                            "Pinning new block {:?} because of pinned object {:?}",
                            block, obj
                        );
                    }
                }

                // Skip to end of object
                curr += obj_size;
                debug_assert!(raw_is_aligned(curr.as_usize(), VM::MIN_ALIGNMENT));
            } else {
                curr += VM::MIN_ALIGNMENT;
            }
        }
    }

    pub fn release(&self) {
        self.forwarding.release();
        // Unprotect pinned pages
        #[cfg(all(feature = "object_pinning", debug_assertions))]
        self.protect_pinned_pages(true);
        #[cfg(feature = "object_pinning")]
        self.forwarding
            .stub_table
            .read()
            .unwrap()
            .regenerate_objects();
    }

    #[cfg(feature = "object_pinning")]
    fn pin_pages(&self, mature_fraction: f64) {
        match self.forwarding.pinning_mode {
            PinningMode::RandomPagePinning(PagePinningMode::CachedEveryGC, fraction) => {
                if fraction > 0.0 {
                    // Cache pinned pages in the harness begin GC. We pin pages till the
                    // end of the region
                    if NUM_GCS.load(Ordering::Relaxed) == 3 {
                        self.pin_random_pages(fraction, mature_fraction, false, true);
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
                    self.pin_random_pages(fraction, mature_fraction, pin_till_end, false);
                }
            }
            _ => {
                unreachable!("We should never get here");
            }
        }
    }

    /// Randomly select `fraction` of currently-allocated OS pages to pin.
    #[cfg(feature = "object_pinning")]
    fn pin_random_pages(
        &self,
        fraction: f64,
        mature_fraction: f64,
        pin_till_end: bool,
        need_to_cache: bool,
    ) {
        let mut total_pages = 0;
        let mut pages_pinned = 0;
        let fraction = fraction.clamp(0.0, 1.0);
        let mut rng = PinRng::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            fraction,
            *self.common().options.compressor_mature_pinning_bias,
            mature_fraction,
        );
        let check_pinning_candidates = *self
            .common()
            .options
            .compressor_check_candidate_before_pinning;
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::CompressorRegion>| {
                let mut page = r.region.start();
                let end = if pin_till_end {
                    r.region.end()
                } else {
                    r.cursor()
                };
                while page < end {
                    let is_pinning_candidate = if check_pinning_candidates {
                        forwarding::PINNED_PAGE_SPEC.load_atomic::<u8>(page, Ordering::SeqCst)
                            == 1_u8
                    } else {
                        true
                    };
                    let mature = page <= r.prev_cursor();
                    if is_pinning_candidate {
                        if rng.should_pin(mature) {
                            pages_pinned += 1;
                            if !need_to_cache {
                                forwarding::PINNED_PAGE_SPEC.store_atomic(
                                    page,
                                    1_u8,
                                    Ordering::SeqCst,
                                );
                                let block_start = forwarding::Block::from_aligned_address(page);
                                let block_end =
                                    forwarding::Block::from_aligned_address(page + BYTES_IN_PAGE);
                                for block in RegionIterator::<Block>::new(block_start, block_end) {
                                    forwarding::pin_block(block);
                                }
                            } else {
                                self.cached_pinned_pages.write().unwrap().insert(page);
                                if check_pinning_candidates {
                                    forwarding::PINNED_PAGE_SPEC.store_atomic(
                                        page,
                                        0_u8,
                                        Ordering::SeqCst,
                                    );
                                }
                            }
                            info!(
                                "Pinning {:?} page {}: {}",
                                r.semantics,
                                page,
                                if mature { "mature" } else { "nursery" }
                            );
                        } else {
                            if check_pinning_candidates {
                                forwarding::PINNED_PAGE_SPEC.store_atomic(
                                    page,
                                    0_u8,
                                    Ordering::SeqCst,
                                );
                            }
                        }
                    }
                    page += BYTES_IN_PAGE;
                    total_pages += 1;
                }
            });
        self.num_pinned_pages
            .store(pages_pinned as u32, Ordering::Relaxed);
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
                        let block_start = forwarding::Block::from_aligned_address(page);
                        let block_end =
                            forwarding::Block::from_aligned_address(page + BYTES_IN_PAGE);
                        for block in RegionIterator::<Block>::new(block_start, block_end) {
                            forwarding::pin_block(block);
                        }
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
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        #[cfg(feature = "vo_bit")]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "{:x}: VO bit not set",
            object
        );
        let stub_table = self.forwarding.stub_table.read().unwrap();
        if stub_table.has_stub(object) {
            stub_table.mark_object_stub(queue, object, &self.forwarding, worker);
        } else if CompressorSpace::<VM>::test_and_mark(object) {
            drop(stub_table);
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

        // There is an edge case here wherein an object starts on an unpinned
        // page and spans into a pinned page, and hence itself is pinned and in
        // the stub table. However, if such an object dies (since we have
        // exact GCs), another object may get moved to its location.
        //
        // In this case, the new object's address will be in the stub table,
        // however it is not the same object that intersected a pinned page.
        // Hence, we need to prune the stub table to remove dead entries. This
        // saves on space as well so in general, it seems like a good idea. We
        // prune the stub table at the end of the `CalculateForwarding` phase of
        // the GC in [`AfterCalculateOffsetVector`].
        #[cfg(feature = "object_pinning")]
        {
            let mut stub_table = self.forwarding.stub_table.write().unwrap();
            if stub_table.has_stub(object) {
                let mut update_closure = |o: ObjectReference| self.forward::<CAN_CLMUL>(o, false);
                stub_table.update_object_stub(object, &mut update_closure);
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
                        let copied_size = if self.is_pinned(obj) || forwarding::is_object_in_pinned_block::<VM>(obj) {
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
                            if !(self.is_pinned(obj) || forwarding::is_object_in_pinned_block::<VM>(obj)) {
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
                let perfect_compaction_end = r.region.start() + total_live_bytes;
                info!(
                    "Compacted {:?} region [{}, {}) -> {to} with {objects} objects; saved {} bytes ({:.2}% savings) (copied {} bytes; live {} bytes)",
                    r.semantics,
                    r.region.start(), r.cursor(), r.cursor() - to,
                    (r.cursor() - to) as f64 / (r.cursor() - perfect_compaction_end) as f64 * 100.0, total_copied_bytes, total_live_bytes,
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

pub struct AfterCalculateOffsetVector<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

impl<VM: VMBinding> AfterCalculateOffsetVector<VM> {
    pub fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

impl<VM: VMBinding> GCWork<VM> for AfterCalculateOffsetVector<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        info!("Finished calculating offset vector for all regions");
        {
            let mut stub_table = self.compressor_space.forwarding.stub_table.write().unwrap();
            stub_table.prune_stubs();
        }
        info!("Finished pruning stub table");
        #[cfg(debug_assertions)]
        {
            let map = FORWARDING_MAP.lock().unwrap();
            info!("Forwarding map has {} live objects", map.len());
            map.iter().for_each(|(from_obj, to_obj)| {
                use crate::policy::compressor::forwarding::is_object_in_pinned_block;

                let from_obj = ObjectReference::from_raw_address(*from_obj).unwrap();
                let to_obj = ObjectReference::from_raw_address(*to_obj).unwrap();
                #[cfg(feature = "vo_bit")]
                debug_assert!(
                    crate::util::metadata::vo_bit::is_vo_bit_set(from_obj),
                    "{:?}: VO bit not set",
                    from_obj,
                );
                if self.compressor_space.is_object_pinned(from_obj) || is_object_in_pinned_block::<VM>(from_obj) {
                    debug_assert_eq!(
                        from_obj, to_obj,
                        "Pinned object {:?} was forwarded to {:?}!",
                        from_obj, to_obj
                    );
                } else {
                    let size = VM::VMObjectModel::get_size_when_copied(from_obj);
                    debug_assert!(
                        from_obj.to_object_start::<VM>() >= to_obj.to_object_start::<VM>(),
                        "Object {:?} was forwarded to {:?} which is after it, potentially overwriting data!",
                        from_obj, to_obj
                    );
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

            COMPUTING_FORWARDING_INFO.store(false, Ordering::SeqCst);
            info!("Finished checking forwarding map for correctness");
        }
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
        self.compressor_space
            .scan_pinned_pages(self.region, self.cursor);
    }
}

#[cfg(feature = "object_pinning")]
struct AfterScanPinnedPages<VM: VMBinding> {
    compressor_space: &'static CompressorSpace<VM>,
}

#[cfg(feature = "object_pinning")]
impl<VM: VMBinding> AfterScanPinnedPages<VM> {
    fn new(compressor_space: &'static CompressorSpace<VM>) -> Self {
        Self { compressor_space }
    }
}

#[cfg(feature = "object_pinning")]
impl<VM: VMBinding> GCWork<VM> for AfterScanPinnedPages<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        if *self
            .compressor_space
            .common()
            .options
            .compressor_print_stub_table_stats
        {
            let num_pinned_pages = self
                .compressor_space
                .num_pinned_pages
                .load(Ordering::Relaxed);
            let filename: &str = &self
                .compressor_space
                .common()
                .options
                .compressor_stub_table_metadata_file;
            self.compressor_space
                .forwarding
                .stub_table
                .read()
                .unwrap()
                .print_table_metrics(filename, num_pinned_pages);
        }
        // Protect pinned pages for the duration of the GC so that we don't touch them accidentally
        // TODO(kunals): Reference processing may need to read from pinned pages.
        // We disable reference processing for now
        #[cfg(debug_assertions)]
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
