use crate::util::constants::BYTES_IN_PAGE;
use crate::util::heap::layout::VMMap;
use crate::util::heap::pageresource::{CommonPageResource, PRAllocFail, PRAllocResult};
use crate::util::heap::space_descriptor::SpaceDescriptor;
use crate::util::heap::{MonotonePageResource, PageResource};
use crate::util::linear_scan::Region;
use crate::util::object_enum::ObjectEnumerator;
use crate::util::Address;
use crate::util::VMThread;
use crate::vm::VMBinding;
use crate::AllocationSemantics;
use atomic::Atomic;
use std::sync::atomic::Ordering;
use std::sync::RwLock;

/// A region in a [`RegionPageResource`] and its allocation cursor.
pub struct AllocatedRegion<R: Region> {
    pub region: R,
    cursor: Atomic<Address>,
    prev_cursor: Atomic<Address>,
    pub semantics: AllocationSemantics,
}

impl<R: Region> AllocatedRegion<R> {
    pub fn cursor(&self) -> Address {
        self.cursor.load(Ordering::Relaxed)
    }

    fn set_cursor(&self, a: Address) {
        self.cursor.store(a, Ordering::Relaxed);
    }

    pub fn prev_cursor(&self) -> Address {
        self.prev_cursor.load(Ordering::Relaxed)
    }

    fn set_prev_cursor(&self, a: Address) {
        self.prev_cursor.store(a, Ordering::Relaxed);
    }
}

struct Sync<R: Region> {
    all_regions: Vec<AllocatedRegion<R>>,
    next_region: usize,
}

/// A [`PageResource`] which allocates pages from a region-structured heap.
/// We assume that allocations are much smaller than regions, as we
/// scan linearly over all regions to allocate, and do not revisit regions
/// before a garbage collection cycle.
pub struct RegionPageResource<VM: VMBinding, R: Region> {
    mpr: MonotonePageResource<VM>,
    sync: RwLock<Sync<R>>,
}

impl<VM: VMBinding, R: Region + 'static> PageResource<VM> for RegionPageResource<VM, R> {
    fn common(&self) -> &CommonPageResource {
        self.mpr.common()
    }

    fn common_mut(&mut self) -> &mut CommonPageResource {
        self.mpr.common_mut()
    }

    fn update_discontiguous_start(&mut self, start: Address) {
        self.mpr.update_discontiguous_start(start)
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
        self.mpr.get_available_physical_pages()
    }
}

impl<VM: VMBinding, R: Region + 'static> RegionPageResource<VM, R> {
    const REGION_PAGES: usize = R::BYTES / BYTES_IN_PAGE;

    pub fn new_contiguous(start: Address, bytes: usize, vm_map: &'static dyn VMMap) -> Self {
        Self::new(MonotonePageResource::new_contiguous(start, bytes, vm_map))
    }

    pub fn new_discontiguous(vm_map: &'static dyn VMMap) -> Self {
        Self::new(MonotonePageResource::new_discontiguous(vm_map))
    }

    fn new(mpr: MonotonePageResource<VM>) -> Self {
        Self {
            mpr,
            sync: RwLock::new(Sync {
                all_regions: vec![],
                next_region: 0,
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
        // XXX(kunals): We always scan from the first region. Since the list of
        // regions contains all the flavors of allocation semantics, we need to
        // check if there's a previous region that can help satisfy this
        // allocation request.
        let mut idx = 0;
        while idx < b.all_regions.len() {
            let cursor = idx;
            if let Option::Some(address) =
                self.allocate_from_region(&mut b.all_regions[cursor], bytes, semantics)
            {
                self.commit_pages(reserved_pages, required_pages, tls);
                return succeed(address, false);
            }
            idx += 1;
        }
        // Else allocate a new region.
        let PRAllocResult {
            start, new_chunk, ..
        } = self.mpr.alloc_pages(
            space_descriptor,
            Self::REGION_PAGES,
            Self::REGION_PAGES,
            semantics,
            tls,
        )?;
        b.all_regions.push(AllocatedRegion {
            region: R::from_aligned_address(start),
            cursor: Atomic::<Address>::new(start),
            prev_cursor: Atomic::<Address>::new(start),
            semantics,
        });
        let cursor = b.all_regions.len() - 1;
        succeed(
            self.allocate_from_region(&mut b.all_regions[cursor], bytes, semantics)
                .unwrap(),
            new_chunk,
        )
    }

    fn allocate_from_region(
        &self,
        alloc: &mut AllocatedRegion<R>,
        bytes: usize,
        semantics: AllocationSemantics,
    ) -> Option<Address> {
        if semantics != alloc.semantics {
            return Option::None;
        }
        let free = alloc.cursor();
        if free + bytes > alloc.region.end() {
            Option::None
        } else {
            alloc.set_cursor(free + bytes);
            Option::Some(free)
        }
    }

    /// Reset the allocation cursor for one region.
    pub fn reset_cursor(&self, alloc: &AllocatedRegion<R>, address: Address) {
        let old = alloc.cursor();
        let new = address.align_up(BYTES_IN_PAGE);
        let pages = (old - new) / BYTES_IN_PAGE;
        self.common().accounting.release(pages);
        alloc.set_cursor(new);
        // After compaction, the previous cursor should be set to the new cursor,
        // so that we can distinguish between mature and nursery objects.
        alloc.set_prev_cursor(new);
    }

    /// Reset the allocator state after a collection, so that the allocator will
    /// revisit regions which the garbage collector has compacted.
    pub fn reset_allocator(&self) {
        self.sync.write().unwrap().next_region = 0;
    }

    pub fn enumerate(&self, enumerator: &mut dyn ObjectEnumerator) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.all_regions.iter() {
            enumerator.visit_address_range(alloc.region.start(), alloc.cursor());
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
