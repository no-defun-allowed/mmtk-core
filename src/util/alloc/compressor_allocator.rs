use std::sync::Arc;

use crate::policy::compressor::CompressorSpace;
use crate::policy::compressor::forwarding::CompressorRegion;
use crate::util::constants::LOG_BYTES_IN_PAGE;
use crate::util::linear_scan::Region;
use crate::util::Address;
use crate::AllocationSemantics;

use crate::util::alloc::{Allocator, BumpPointer};

use crate::policy::space::Space;
use crate::util::conversions::bytes_to_pages_up;
use crate::util::opaque_pointer::*;
use crate::vm::VMBinding;

/// Size of a thread-local allocation buffer. Currently it is set to 4 KB.
const BLOCK_SIZE: usize = 1 << LOG_BYTES_IN_PAGE;
const BLOCK_MASK: usize = BLOCK_SIZE - 1;

#[repr(C)]
pub struct CompressorAllocator<VM: VMBinding> {
    /// [`VMThread`] associated with this allocator instance
    pub tls: VMThread,
    /// Default bump-pointer for normal (mixed) allocations. Here, mixed means
    /// that the object may have both reference and pointer fields.
    pub bump_pointer: BumpPointer,
    /// Bump-pointer for reference-only allocations.
    pub ref_bump_pointer: BumpPointer,
    /// Bump-pointer for primitive allocations.
    pub non_ref_bump_pointer: BumpPointer,
    /// [`CompressorSpace`](src/policy/compressor/CompressorSpace) associated with this allocator instance.
    space: &'static CompressorSpace<VM>,
    pub(in crate::util::alloc) context: Arc<AllocatorContext<VM>>,
}

impl<VM: VMBinding> CompressorAllocator<VM> {
    pub(crate) fn get_bump_pointer(&self, semantics: AllocationSemantics) -> &BumpPointer {
        match semantics {
            AllocationSemantics::Default => &self.bump_pointer,
            AllocationSemantics::ReferenceArray => &self.ref_bump_pointer,
            AllocationSemantics::PrimitiveArray => &self.non_ref_bump_pointer,
            _ => panic!("Unsupported allocation semantics: {:?}", semantics),
        }
    }

    pub(crate) fn get_bump_pointer_mut(
        &mut self,
        semantics: AllocationSemantics,
    ) -> &mut BumpPointer {
        match semantics {
            AllocationSemantics::Default => &mut self.bump_pointer,
            AllocationSemantics::ReferenceArray => &mut self.ref_bump_pointer,
            AllocationSemantics::PrimitiveArray => &mut self.non_ref_bump_pointer,
            _ => panic!("Unsupported allocation semantics: {:?}", semantics),
        }
    }

    pub(crate) fn set_limit(
        &mut self,
        start: Address,
        limit: Address,
        semantics: AllocationSemantics,
    ) {
        self.get_bump_pointer_mut(semantics).reset(start, limit);
    }

    pub(crate) fn reset(&mut self) {
        let zero = Address::ZERO;
        self.bump_pointer.reset(zero, zero);
        self.ref_bump_pointer.reset(zero, zero);
        self.non_ref_bump_pointer.reset(zero, zero);
    }
}

use crate::util::alloc::allocator::align_allocation_no_fill;
use crate::util::alloc::fill_alignment_gap;

use super::allocator::AllocatorContext;

impl<VM: VMBinding> Allocator<VM> for CompressorAllocator<VM> {
    fn get_space(&self) -> &'static dyn Space<VM> {
        self.space as _
    }

    fn get_context(&self) -> &AllocatorContext<VM> {
        &self.context
    }

    fn does_thread_local_allocation(&self) -> bool {
        true
    }

    fn get_thread_local_buffer_granularity(&self) -> usize {
        BLOCK_SIZE
    }

    fn alloc(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        semantics: AllocationSemantics,
    ) -> Address {
        trace!("alloc");

        let bump_pointer = self.get_bump_pointer_mut(semantics);
        let result = align_allocation_no_fill::<VM>(bump_pointer.cursor, align, offset);
        let new_cursor = result + size;

        if new_cursor > bump_pointer.limit {
            trace!("Thread local buffer used up, go to alloc slow path");
            self.alloc_slow(size, align, offset, semantics)
        } else {
            fill_alignment_gap::<VM>(bump_pointer.cursor, result);
            bump_pointer.cursor = new_cursor;
            trace!(
                "Bump allocation size: {}, result: {}, new_cursor: {}, limit: {}",
                size,
                result,
                bump_pointer.cursor,
                bump_pointer.limit
            );
            result
        }
    }

    fn alloc_slow_once(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        semantics: AllocationSemantics,
    ) -> Address {
        trace!("alloc_slow");
        self.acquire_block(size, align, offset, semantics, false)
    }

    /// Slow path for allocation if precise stress testing has been enabled.
    /// It works by manipulating the limit to be always below the cursor.
    /// Can have three different cases:
    ///  - acquires a new block if the hard limit has been met;
    ///  - allocates an object using the bump pointer semantics from the
    ///    fastpath if there is sufficient space; and
    ///  - does not allocate an object but forces a poll for GC if the stress
    ///    factor has been crossed.
    fn alloc_slow_once_precise_stress(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        semantics: AllocationSemantics,
        need_poll: bool,
    ) -> Address {
        if need_poll {
            return self.acquire_block(size, align, offset, semantics, true);
        }

        trace!("alloc_slow stress_test");
        let bump_pointer = self.get_bump_pointer_mut(semantics);
        let result = align_allocation_no_fill::<VM>(bump_pointer.cursor, align, offset);
        let new_cursor = result + size;

        // For stress test, limit is [0, block_size) to artificially make the
        // check in the fastpath (alloc()) fail. The real limit is recovered by
        // adding it to the current cursor.
        if new_cursor > bump_pointer.cursor + bump_pointer.limit.as_usize() {
            self.acquire_block(size, align, offset, semantics, true)
        } else {
            fill_alignment_gap::<VM>(bump_pointer.cursor, result);
            bump_pointer.limit -= new_cursor - bump_pointer.cursor;
            bump_pointer.cursor = new_cursor;
            trace!(
                "alloc_slow: Bump allocation size: {}, result: {}, new_cursor: {}, limit: {}",
                size,
                result,
                bump_pointer.cursor,
                bump_pointer.limit
            );
            result
        }
    }

    fn get_tls(&self) -> VMThread {
        self.tls
    }
}

impl<VM: VMBinding> CompressorAllocator<VM> {
    pub(crate) fn new(
        tls: VMThread,
        space: &'static CompressorSpace<VM>,
        context: Arc<AllocatorContext<VM>>,
    ) -> Self {
        CompressorAllocator {
            tls,
            bump_pointer: BumpPointer::default(),
            ref_bump_pointer: BumpPointer::default(),
            non_ref_bump_pointer: BumpPointer::default(),
            space,
            context,
        }
    }

    fn acquire_block(
        &mut self,
        size: usize,
        align: usize,
        offset: usize,
        semantics: AllocationSemantics,
        stress_test: bool,
    ) -> Address {
        if self.handle_obvious_oom_request(self.tls, size) {
            return Address::ZERO;
        }

        let mut options = self.get_context().get_alloc_options();
        options.semantics = semantics;
        self.get_context().set_alloc_options(options);

        let block_size = (size + BLOCK_MASK) & (!BLOCK_MASK);
        let acquired_hole = match self.space.acquire_hole(size, Some(block_size), semantics) {
            Some(hole) => Some(hole),
            None => {
                let region = self.space.acquire(
                    self.tls,
                    CompressorRegion::BYTES >> LOG_BYTES_IN_PAGE,
                    self.get_context().get_alloc_options(),
                );
                if region.is_zero() {
                    None
                } else {
                    // Take a block out of the region, and give the rest to the space.
                    self.space.add_hole(
                        semantics,
                        (region + block_size)..(region + CompressorRegion::BYTES)
                    );
                    Some(region..(region + block_size))
                }
            },
        };
        self.get_context()
            .set_alloc_options(crate::util::alloc::AllocationOptions::default());
        match acquired_hole {
            None => {
                trace!("Failed to acquire a new block");
                Address::ZERO
            },
            Some(hole) => {
                let start = hole.start;
                let end = hole.end;
                let size = end - start;
                trace!("Acquired a new block from {start} to {end}");
                #[cfg(feature = "object_pinning")]
                self.space.touch_pages(start, size);
                if !stress_test {
                    self.set_limit(start, end, semantics);
                    self.alloc(size, align, offset, semantics)
                } else {
                    // For a stress test, we artificially make the fastpath fail by
                    // manipulating the limit as below.
                    // The assumption here is that we use an address range such that
                    // cursor > block_size always.
                    self.set_limit(
                        start,
                        unsafe { Address::from_usize(size) },
                        semantics,
                    );
                    // Note that we have just acquired a new block so we know that we don't have to go
                    // through the entire allocation sequence again, we can directly call the slow path
                    // allocation.
                    self.alloc_slow_once_precise_stress(size, align, offset, semantics, false)
                }
            }
        }
    }
}
