use crate::util::constants::BYTES_IN_PAGE;
use crate::util::heap::layout::vm_layout::BYTES_IN_CHUNK;
use crate::util::heap::layout::VMMap;
use crate::util::heap::pageresource::{CommonPageResource, PRAllocFail, PRAllocResult};
use crate::util::heap::space_descriptor::SpaceDescriptor;
use crate::util::heap::{FreeListPageResource, PageResource};
use crate::util::linear_scan::Region;
use crate::util::object_enum::ObjectEnumerator;
use crate::util::Address;
use crate::util::VMThread;
use crate::vm::VMBinding;
use enum_map::EnumMap;
use std::ops::Range;
use std::sync::{Mutex, RwLock};
use crate::AllocationSemantics;

/// A region in a [`RegionPageResource`] and its free list.
pub struct AllocatedRegion<R: Region> {
    pub region: R,
    free_list: Mutex<Vec<Range<Address>>>,
    // Regions we just allocated from a chunk
    // aren't assigned any semantics.
    pub semantics: Option<AllocationSemantics>,
}

impl<R: Region> AllocatedRegion<R> {
    pub fn used_bytes(&self) -> usize {
        let free_list = self.free_list.lock().unwrap();
        R::BYTES - free_list.iter().map(|r| r.end - r.start).sum::<usize>()
    }
}

struct Sync<R: Region> {
    all_regions: Vec<AllocatedRegion<R>>,
    next_regions: EnumMap<AllocationSemantics, usize>,
}

/// A [`PageResource`] which allocates pages from a region-structured heap.
/// We assume that allocations are much smaller than regions, as we
/// scan linearly over all regions to allocate, and do not revisit regions
/// before a garbage collection cycle.
pub struct RegionPageResource<VM: VMBinding, R: Region> {
    flpr: FreeListPageResource<VM>,
    sync: RwLock<Sync<R>>,
}

impl<VM: VMBinding, R: Region + 'static> PageResource<VM> for RegionPageResource<VM, R> {
    fn common(&self) -> &CommonPageResource {
        self.flpr.common()
    }

    fn common_mut(&mut self) -> &mut CommonPageResource {
        self.flpr.common_mut()
    }

    fn update_discontiguous_start(&mut self, start: Address) {
        self.flpr.update_discontiguous_start(start)
    }

    fn alloc_pages(
        &self,
        space_descriptor: SpaceDescriptor,
        reserved_pages: usize,
        required_pages: usize,
        semantics: AllocationSemantics,
        tls: VMThread,
    ) -> Result<PRAllocResult, PRAllocFail> {
        assert!(reserved_pages <= Self::REGION_PAGES);
        assert!(required_pages <= reserved_pages);
        self.alloc(
            space_descriptor,
            reserved_pages,
            required_pages,
            semantics,
            tls,
        )
    }

    fn get_available_physical_pages(&self) -> usize {
        self.flpr.get_available_physical_pages()
    }
}

impl<VM: VMBinding, R: Region + 'static> RegionPageResource<VM, R> {
    const REGION_PAGES: usize = R::BYTES / BYTES_IN_PAGE;

    pub fn new_contiguous(start: Address, bytes: usize, vm_map: &'static dyn VMMap) -> Self {
        Self::new(FreeListPageResource::new_contiguous(start, bytes, vm_map))
    }

    pub fn new_discontiguous(vm_map: &'static dyn VMMap) -> Self {
        Self::new(FreeListPageResource::new_discontiguous(vm_map))
    }

    fn new(flpr: FreeListPageResource<VM>) -> Self {
        Self {
            flpr,
            sync: RwLock::new(Sync {
                all_regions: vec![],
                next_regions: EnumMap::from_fn(|_| 0),
            }),
        }
    }

    fn alloc(
        &self,
        space_descriptor: SpaceDescriptor,
        reserved_pages: usize,
        required_pages: usize,
        semantics: AllocationSemantics,
        tls: VMThread,
    ) -> Result<PRAllocResult, PRAllocFail> {
        let mut b = self.sync.write().unwrap();
        let succeed = |start: Address, new_chunk: bool| {
            Result::Ok(PRAllocResult {
                start,
                pages: required_pages,
                new_chunk,
            })
        };
        let bytes = reserved_pages * BYTES_IN_PAGE;
        // First try to reuse a region.
        let mut idx = b.next_regions[semantics];
        while idx < b.all_regions.len() {
            let (addr, pages_wasted) = self.allocate_from_region(&mut b.all_regions[idx], bytes, semantics);
            self.common().accounting.reserve_and_commit(pages_wasted);
            if let Some(address) = addr {
                self.commit_pages(reserved_pages, required_pages, tls);
                b.next_regions[semantics] = idx;
                return succeed(address, false);
            }
            idx += 1;
        }
        b.next_regions[semantics] = idx;
        // Else allocate a new chunk to carve regions from.
        let chunk_start = self.flpr.allocate_one_chunk_no_commit(space_descriptor)?.start;
        assert!(chunk_start.is_aligned_to(BYTES_IN_CHUNK));
        assert!(R::BYTES < BYTES_IN_CHUNK); // XXX: where to do this properly?
        for i in 0..(BYTES_IN_CHUNK / R::BYTES) {
            let region_start = chunk_start + R::BYTES * i;
            b.all_regions.push(AllocatedRegion {
                region: R::from_aligned_address(region_start),
                free_list: Mutex::new(vec![region_start..(region_start + R::BYTES)]),
                semantics: None,
            });
        }
        // This allocation from the first new region has to succeed, and can't waste space
        // as it's at the very start of the region.
        self.commit_pages(reserved_pages, required_pages, tls);
        succeed(
            self.allocate_from_region(&mut b.all_regions[idx], bytes, semantics)
                .0.expect("allocation should fit in new region"),
            true,
        )
    }

    fn allocate_from_region(
        &self,
        alloc: &mut AllocatedRegion<R>,
        bytes: usize,
        semantics: AllocationSemantics,
    ) -> (Option<Address>, usize) {
        // Viable regions either have the right semantics,
        // or haven't assigned semantics yet.
        if let Some(s) = alloc.semantics {
            if semantics != s {
                return (None, 0);
            }
        }
        // If the region hasn't been assigned any semantics yet,
        // due to being newly allocated, assign it now.
        alloc.semantics = Some(semantics);
        let mut bytes_wasted = 0;
        let mut free_list = alloc.free_list.lock().unwrap();
        loop {
            match free_list.pop() {
                None => return (None, bytes_wasted / BYTES_IN_PAGE),
                Some(range) => {
                    if range.end - range.start >= bytes {
                        free_list.push((range.start + bytes)..(range.end));
                        return (Some(range.start), bytes_wasted / BYTES_IN_PAGE);
                    } else {
                        // TODO(kunals): Maybe we should keep the hole in the
                        // free list in case we can allocate from it later.
                        bytes_wasted += range.end - range.start;
                    }
                }
            }
        }
    }

    pub fn reset_free_list(&self, region: &AllocatedRegion<R>, new_free_list: &[(Address, Address)]) {
        let mut free_list = region.free_list.lock().unwrap();
        let old_free_bytes = free_list.iter().map(|r| r.end - r.start).sum::<usize>();
        // Get whole pages out of the free list. We reverse so that popping
        // the vector later will give us the first range on the free list first.
        let new_free_list = new_free_list.iter()
            .map(|(s, e)| *s..*e)
            .filter(|r| !r.is_empty())
            .rev()
            .collect::<Vec<_>>();
        info!("free list: {new_free_list:?}");
        // TODO(kunals): If a region is completely free, we should reset its semantics to None
        let new_free_bytes = new_free_list.iter().map(|r| r.end - r.start).sum::<usize>();
        if new_free_bytes > old_free_bytes {
            let freed_pages = (new_free_bytes - old_free_bytes) / BYTES_IN_PAGE;
            self.common().accounting.release(freed_pages);
        }
        *free_list = new_free_list;
    }

    /// Reset the allocator state after a collection, so that the allocator will
    /// revisit regions which the garbage collector has compacted.
    pub fn reset_allocator(&self) {
        self.sync.write().unwrap().next_regions = EnumMap::from_fn(|_| 0);
    }

    pub fn enumerate(&self, enumerator: &mut dyn ObjectEnumerator) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.all_regions.iter() {
            enumerator.visit_address_range(alloc.region.start(), alloc.region.end());
        }
    }

    pub fn with_regions<T>(&self, f: &mut impl FnMut(&Vec<AllocatedRegion<R>>) -> T) -> T {
        let sync = self.sync.read().unwrap();
        f(&sync.all_regions)
    }

    pub fn enumerate_regions(&self, enumerator: &mut impl FnMut(&AllocatedRegion<R>)) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.all_regions.iter() {
            enumerator(alloc);
        }
    }
}
