use crate::policy::compressor::GC_MARK_BIT_MASK;
use crate::util::constants::{BYTES_IN_WORD, BITS_IN_BYTE};
use crate::util::linear_scan::{Region, RegionIterator};
use crate::util::metadata::side_metadata::spec_defs::{
    COMPRESSOR_MARK, COMPRESSOR_OFFSET_VECTOR, COMPRESSOR_SELECTED,
};
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::{Address, ObjectReference};
use crate::vm::object_model::ObjectModel;
use crate::vm::VMBinding;
use atomic::Ordering;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;

/// A [`CompressorRegion`] is the granularity at which [`super::CompressorSpace`]
/// compacts the heap. Objects are allocated inside one region, and are only ever
/// moved *within* that region.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct CompressorRegion(Address);
impl Region for CompressorRegion {
    const LOG_BYTES: usize = 18; // 1 MiB
    fn from_aligned_address(address: Address) -> Self {
        assert!(
            address.is_aligned_to(Self::BYTES),
            "{address} is not aligned"
        );
        CompressorRegion(address)
    }
    fn start(&self) -> Address {
        self.0
    }
}

/// A finite-state machine which visits the positions of marked bits in
/// the mark bitmap, and accumulates the size of live data that it has
/// seen between marked bits.
///
/// The Compressor caches the state of the transducer at the start of
/// each block by serialising the state using [`Transducer::encode`], and
/// then deserialises the state whilst computing forwarding pointers
/// using [`Transducer::decode`].
#[derive(Debug)]
struct Transducer {
    /// The offset from the start of the region at which the Compressor
    /// will move the next object to, following preceding objects which
    /// were visited by the transducer.
    offset: u32,
    /// The address of the last mark bit which the transducer visited.
    last_bit_visited: Address,
    /// Whether or not the transducer is currently inside an object
    /// (i.e. if it has seen a first bit but no matching last bit yet).
    in_object: bool,
}
type EncodedTransducer = u32;
pub(crate) const LOG_BITS_IN_ENCODED: usize = (BITS_IN_BYTE * std::mem::size_of::<EncodedTransducer>()).ilog2() as usize;
impl Transducer {
    pub fn new() -> Self {
        Self {
            offset: 0,
            last_bit_visited: Address::ZERO,
            in_object: false,
        }
    }
    pub fn visit_mark_bit(&mut self, address: Address) {
        if self.in_object {
            // The size of an object is the distance between the end and
            // start of the object, and the last word of the object is one
            // word prior to the end of the object. Thus we must add an
            // extra word, in order to compute the size of the object based
            // on the distance between its first and last words.
            let first_word = self.last_bit_visited;
            let last_word = address;
            let size = last_word - first_word + BYTES_IN_WORD;
            self.offset += size as u32;
        }
        self.in_object = !self.in_object;
        self.last_bit_visited = address;
    }

    pub fn encode(&self, current_position: Address) -> EncodedTransducer {
        if self.in_object {
            // We count the space between the last mark bit and
            // the current address as live when we stop in the
            // middle of an object.
            self.offset + (current_position - self.last_bit_visited) as u32 + 1
        } else {
            self.offset
        }
    }

    pub fn decode(value: EncodedTransducer, current_position: Address) -> Self {
        Transducer {
            offset: value & !1,
            last_bit_visited: current_position,
            in_object: (value & 1) == 1,
        }
    }
}

pub(crate) enum CompactLimit {
    AlwaysCompact,
    Percent(u32),
}

pub(crate) struct ForwardingMetadata<VM: VMBinding> {
    compact_limit: CompactLimit,
    calculated: AtomicBool,
    vm: PhantomData<VM>,
}

// A block in the Compressor is the granularity at which we cache
// the amount of live data preceding an address. We set it to 512 bytes
// following the paper.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(crate) struct Block(Address);
impl Region for Block {
    const LOG_BYTES: usize = 9;
    fn from_aligned_address(address: Address) -> Self {
        assert!(address.is_aligned_to(Self::BYTES));
        Block(address)
    }
    fn start(&self) -> Address {
        self.0
    }
}

pub(crate) const MARK_SPEC: SideMetadataSpec = COMPRESSOR_MARK;
pub(crate) const OFFSET_VECTOR_SPEC: SideMetadataSpec = COMPRESSOR_OFFSET_VECTOR;
pub(crate) const SELECTED_SPEC: SideMetadataSpec = COMPRESSOR_SELECTED;

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new(compact_limit: CompactLimit) -> ForwardingMetadata<VM> {
        ForwardingMetadata {
            compact_limit,
            calculated: AtomicBool::new(false),
            vm: PhantomData,
        }
    }

    pub fn mark_last_word_of_object(&self, object: ObjectReference) {
        let last_word_of_object = object.to_object_start::<VM>()
            + VM::VMObjectModel::get_current_size(object)
            - BYTES_IN_WORD;
        #[cfg(debug_assertions)]
        {
            // We require to be able to iterate upon first and last bits in the
            // same bitmap. Therefore the first and last bits cannot be the
            // same, else we would only encounter one of the two bits.
            // This requirement implies that objects must be at least two words
            // large.
            debug_assert!(
                MARK_SPEC.are_different_metadata_bits(
                    object.to_object_start::<VM>(),
                    last_word_of_object
                ),
                "The first and last mark bits should be different bits."
            );
        }

        MARK_SPEC.fetch_or_atomic(last_word_of_object, GC_MARK_BIT_MASK, Ordering::SeqCst);
    }

    // SAFETY: Only call this function when the processor supports pclmulqdq and popcnt.
    unsafe fn calculate_offset_vector_clmul(
        &self,
        region: CompressorRegion,
        cursor: Address,
    ) -> u32 {
        #[target_feature(enable = "pclmulqdq,popcnt")]
        unsafe fn inner(offset: &mut u32, in_object: &mut i64, word: usize, addr: Address) {
            use std::arch::x86_64;
            // encode state to offset vector
            let encoded = *offset + (((*in_object as usize) >> 63) as u32);
            if addr.is_aligned_to(Block::BYTES) {
                OFFSET_VECTOR_SPEC.store_atomic::<EncodedTransducer>(
                    addr,
                    encoded,
                    Ordering::Relaxed,
                );
            }
            // update by clmul
            let ones = unsafe { x86_64::_mm_set1_epi8(0xFFu8 as i8) };
            let vector = unsafe { x86_64::_mm_set_epi64x(0, word as i64) };
            let mask: i64 =
                unsafe { x86_64::_mm_cvtsi128_si64(x86_64::_mm_clmulepi64_si128(vector, ones, 0)) };
            let flipped = mask ^ *in_object;
            *in_object = flipped >> 63;
            *offset += (((flipped as usize | word).count_ones()) * 8) as u32;
        }

        let mut offset: u32 = 0;
        let mut in_object: i64 = 0;
        MARK_SPEC.scan_words::<u8>(
            region.start(),
            cursor.align_up(Block::BYTES),
            &mut |_, _| panic!("should be word aligned, got a bit instead"),
            &mut |_, _| panic!("should be word aligned, got a byte instead"),
            &mut |word: usize, addr: Address| {
                inner(&mut offset, &mut in_object, word, addr);
            },
        );
        offset
    }

    fn calculate_offset_vector_base(&self, region: CompressorRegion, cursor: Address) -> u32 {
        let mut state = Transducer::new();
        let first_block = Block::from_aligned_address(region.start());
        let last_block = Block::from_aligned_address(cursor);
        for block in RegionIterator::<Block>::new(first_block, last_block) {
            OFFSET_VECTOR_SPEC.store_atomic::<EncodedTransducer>(
                block.start(),
                state.encode(block.start()),
                Ordering::Relaxed,
            );
            MARK_SPEC.scan_non_zero_values::<u8>(
                block.start(),
                block.end(),
                &mut |addr: Address| {
                    state.visit_mark_bit(addr);
                },
            );
        }
        state.offset
    }

    pub fn select_region(&self, region: CompressorRegion) {
        SELECTED_SPEC.store_atomic::<u8>(region.start(), 1, Ordering::Relaxed);
    }

    pub fn calculate_offset_vector(&self, region: CompressorRegion, cursor: Address) {
        let blocks_large_enough = Block::LOG_BYTES >= 9;
        let cpu_supports_features =
            is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("popcnt");
        let used = if blocks_large_enough && cpu_supports_features {
            unsafe {
                // SAFETY: We checked the processor supports the
                // necessary instructions.
                self.calculate_offset_vector_clmul(region, cursor)
            }
        } else {
            self.calculate_offset_vector_base(region, cursor)
        };
        self.calculated.store(true, Ordering::Relaxed);
        let percent = used / ((CompressorRegion::BYTES / 100) as u32);
        let will_compact = match self.compact_limit {
            CompactLimit::AlwaysCompact => true,
            CompactLimit::Percent(limit) => percent < limit,
        };
        SELECTED_SPEC.store_atomic::<u8>(region.start(), will_compact as u8, Ordering::Relaxed);
    }

    pub fn calculate_and_walk_offset_vector(
        &self,
        region: CompressorRegion,
        cursor: Address,
        block_lock: &mut (impl FnMut(Block, &mut dyn FnMut()) + ?Sized),
        f: &mut impl FnMut(ObjectReference),
    ) {
        let mut state = Transducer::new();
        let first_block = Block::from_aligned_address(region.start());
        let last_block = Block::from_aligned_address(cursor);
        self.calculated.store(true, Ordering::Relaxed);
        for block in RegionIterator::<Block>::new(first_block, last_block) {
            OFFSET_VECTOR_SPEC.store_atomic::<EncodedTransducer>(
                block.start(),
                state.encode(block.start()),
                Ordering::Relaxed,
            );
            block_lock(block, &mut || {
                MARK_SPEC.scan_non_zero_values::<u8>(
                    block.start(),
                    block.end(),
                    &mut |addr: Address| {
                        state.visit_mark_bit(addr);
                        if state.in_object {
                            f(ObjectReference::from_raw_address(addr).unwrap())
                        }
                    },
                );
            });
        }
    }

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
    }

    pub fn is_forwarding_region(&self, region: CompressorRegion) -> bool {
        SELECTED_SPEC.load_atomic::<u8>(region.start(), Ordering::Relaxed) != 0
    }

    pub fn forward(&self, address: Address) -> Address {
        debug_assert!(
            self.calculated.load(Ordering::Relaxed),
            "forward() should only be called when we have calculated an offset vector"
        );
        if SELECTED_SPEC.load_atomic::<u8>(address, Ordering::Relaxed) == 0 {
            return address;
        }
        let region = CompressorRegion::from_unaligned_address(address);
        let block = Block::from_unaligned_address(address);
        let mut state = Transducer::decode(
            OFFSET_VECTOR_SPEC.load_atomic::<EncodedTransducer>(block.start(), Ordering::Relaxed),
            block.start(),
        );
        // The transducer in this implementation computes the distance of
        // an object from the start of the region; whereas Total-Live-Data in
        // the paper computes the distance of the object from the start of the block.
        MARK_SPEC.scan_non_zero_values::<u8>(block.start(), address, &mut |addr: Address| {
            state.visit_mark_bit(addr)
        });
        region.start() + state.offset as usize
    }

    pub fn scan_marked_objects(
        &self,
        start: Address,
        end: Address,
        f: &mut impl FnMut(ObjectReference),
    ) {
        let mut in_object = false;
        MARK_SPEC.scan_non_zero_values::<u8>(start, end, &mut |addr: Address| {
            if !in_object {
                let object = ObjectReference::from_raw_address(addr).unwrap();
                f(object);
            }
            in_object = !in_object;
        });
    }

    pub fn has_calculated_forwarding_addresses(&self) -> bool {
        self.calculated.load(Ordering::Relaxed)
    }
}
