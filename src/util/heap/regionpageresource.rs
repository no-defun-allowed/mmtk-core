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
use std::sync::RwLock;
use crate::AllocationSemantics;

pub struct AllocatedRegion<R: Region> {
    pub region: R,
    pub semantics: AllocationSemantics,
}

struct Sync<R: Region> {
    used_regions: Vec<AllocatedRegion<R>>,
    free_regions: Vec<R>,
}

/// A [`PageResource`] which allocates pages from a region-structured heap.
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
        assert_eq!(reserved_pages, Self::REGION_PAGES);
        assert_eq!(required_pages, reserved_pages);
        self.alloc(
            space_descriptor,
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
                used_regions: vec![],
                free_regions: vec![],
            }),
        }
    }

    fn alloc(
        &self,
        space_descriptor: SpaceDescriptor,
        semantics: AllocationSemantics,
        tls: VMThread,
    ) -> Result<PRAllocResult, PRAllocFail> {
        let mut b = self.sync.write().unwrap();
        let succeed = |start: Address, new_chunk: bool| {
            Result::Ok(PRAllocResult {
                start,
                pages: Self::REGION_PAGES,
                new_chunk,
            })
        };
        // First try to take a free region.
        match b.free_regions.pop() {
            Some(r) => {
                b.used_regions.push(AllocatedRegion {
                    region: r,
                    semantics
                });
                self.commit_pages(Self::REGION_PAGES, Self::REGION_PAGES, tls);
                succeed(r.start(), false)
            }
            None => {
                // Else allocate a new chunk to carve regions from.
                let chunk_start = self.flpr.allocate_one_chunk_no_commit(space_descriptor)?.start;
                assert!(chunk_start.is_aligned_to(BYTES_IN_CHUNK));
                assert!(R::BYTES < BYTES_IN_CHUNK); // XXX: where to do this properly?
                // Get the first region to service this allocation.
                let region = R::from_aligned_address(chunk_start);
                self.used_regions.push(AllocatedRegion { region, semantics });
                // Push the remaining regions to the free regions.
                for i in 1..(BYTES_IN_CHUNK / R::BYTES) {
                    let region_start = chunk_start + R::BYTES * i;
                    b.free_regions.push(R::from_aligned_address(region_start))
                };
                succeed(chunk_start, true)
            }
        }
    }

    pub fn reset_allocator(&self) { }

    pub fn enumerate(&self, enumerator: &mut dyn ObjectEnumerator) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.used_regions.iter() {
            enumerator.visit_address_range(alloc.region.start(), alloc.region.end());
        }
    }

    pub fn with_regions<T>(&self, f: &mut impl FnMut(&Vec<AllocatedRegion<R>>) -> T) -> T {
        let sync = self.sync.read().unwrap();
        f(&sync.used_regions)
    }

    pub fn enumerate_regions(&self, enumerator: &mut impl FnMut(&AllocatedRegion<R>)) {
        let sync = self.sync.read().unwrap();
        for alloc in sync.used_regions.iter() {
            enumerator(alloc);
        }
    }
}
