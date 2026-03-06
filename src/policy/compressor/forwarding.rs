use crate::util::constants::BYTES_IN_WORD;
use crate::util::linear_scan::Region;
use crate::util::metadata::side_metadata::ranges::Bits;
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
    const LOG_BYTES: usize = 18; // 256 KiB
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

// We compress the offset vector to 32-bit elements on a 64-bit system, seeing
// that regions are always smaller than chunks (<= 4MiB), so an offset within
// a region only needs 22 bits at most. I've presently set the region size to 256 kiB,
// and so a Transducer state can be made to fit in 16 bits, being
//     18 bits offset - 3 bits alignment + 1 bit in_object tag
// But that arrangement wouldn't allow for any larger regions, and I would
// prefer for all the magic numbers and types in this file to not be fragile like that.
//
// We have diminishing returns from shrinking the offset vector anyway:
// both calculate_offset_vector() and forward() have to pull in a block's worth of
// mark bitmap as well as the offset for the block, so any reduction in the size of
// offsets gets Amdahl-ed by the mark bitmap.
pub(crate) type Offset = u32;
pub(crate) const LOG_BITS_IN_OFFSET: usize = Offset::BITS.ilog2() as usize;

/// A finite-state machine which visits the positions of marked bits in
/// the mark bitmap, and accumulates the size of live data that it has
/// seen between marked bits.
///
/// The Compressor caches the state of the transducer at the start of
/// each block by serialising the state using [`Transducer::encode`], and
/// then deserialises the state whilst computing forwarding pointers
/// using [`Transducer::decode`].
#[cfg(not(feature = "compressor_art_marking"))]
#[derive(Clone, Debug)]
struct Transducer {
    /// The offset from the start of the region for the next object to be copied
    /// to, following preceding objects which were visited by the transducer.
    offset: Offset,
    /// The address of the last mark bit which the transducer visited.
    last_bit_visited: Address,
    /// Whether or not the transducer is currently inside an object
    /// (i.e. if it has seen a first bit but no matching last bit yet).
    in_object: bool,
}

#[cfg(not(feature = "compressor_art_marking"))]
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
            self.offset += size as Offset;
        }
        self.in_object = !self.in_object;
        self.last_bit_visited = address;
    }

    pub fn encode(&self, current_position: Address) -> Offset {
        if self.in_object {
            // We count the space between the last mark bit and
            // the current address as live when we stop in the
            // middle of an object.
            self.offset + (current_position - self.last_bit_visited) as Offset + 1
        } else {
            self.offset
        }
    }

    pub fn decode(offset: Offset, current_position: Address) -> Self {
        Transducer {
            offset: offset & !1,
            last_bit_visited: current_position,
            in_object: (offset & 1) == 1,
        }
    }
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

pub(crate) enum CompactLimit {
    AlwaysCompact,
    Percentage(u8),
}

pub(crate) const MARK_SPEC: SideMetadataSpec = COMPRESSOR_MARK;
#[cfg(feature = "compressor_art_marking")]
const BYTES_PER_MARK_BIT: Offset = (1usize << MARK_SPEC.log_bytes_in_region) as Offset;
pub(crate) const OFFSET_VECTOR_SPEC: SideMetadataSpec = COMPRESSOR_OFFSET_VECTOR;
pub(crate) const SELECTED_SPEC: SideMetadataSpec = COMPRESSOR_SELECTED;

pub struct ForwardingMetadata<VM: VMBinding> {
    compact_limit: CompactLimit,
    calculated: AtomicBool,
    vm: PhantomData<VM>,
    supports_clmul: bool,
}

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new(compact_limit: CompactLimit, _use_clmul: bool) -> ForwardingMetadata<VM> {
        cfg_if::cfg_if! { if #[cfg(target_arch = "x86_64")] {
            let supports_clmul = _use_clmul
                && is_x86_feature_detected!("pclmulqdq")
                && is_x86_feature_detected!("popcnt");
        } else {
            let supports_clmul = false;
        }}
        ForwardingMetadata {
            compact_limit,
            calculated: AtomicBool::new(false),
            vm: PhantomData,
            supports_clmul,
        }
    }

    pub fn supports_clmul(&self) -> bool {
        self.supports_clmul
    }

    pub fn mark_rest_of_object(&self, object: ObjectReference) {
        // We implement two styles of mark bitmap:
        if cfg!(feature = "compressor_art_marking") {
            // - When `cfg(feature = "compressor_art_marking")`, we use a style as in
            //   the Android Runtime and Clozure Common Lisp, where we mark every
            //   bit corresponding to a word in each live object.
            //
            // XXX: this will SeqCst and we don't need that.
            MARK_SPEC.bset_metadata(
                object.to_object_start::<VM>(),
                VM::VMObjectModel::get_current_size(object),
            );
        } else {
            // - When `cfg(not(feature = "compressor_art_marking"))`, we follow the style
            //   in the original Compressor paper, where we mark bits corresponding to
            //   the first and last words of each live object. The live data then
            //   corresponds to the bits between and including each pair of set bits.
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
            // The original style requires fewer words to be marked, but then any
            // use of the bitmap must keep track of if a bit is inside or outside a
            // pair of mark bits, in order to determine if the bit designates a live word
            // or not. (The `Transducer` and carryless multiply-based algorithms track
            // the state in the `in_object` field and `carry` variable respectively.)
            // The Android Runtime style sets more bits, but does not require
            // any state to be tracked when using the bitmap. But most objects are
            // small, and we set a contiguous range of bits which tend to reside in the
            // same bytes, words, cache lines, etc.; so setting more bits is not actually
            // that inefficient in practice.
            MARK_SPEC.fetch_or_atomic::<u8>(last_word_of_object, 1, Ordering::Relaxed);
        }
    }

    pub fn calculate_offset_vector(&self, region: CompressorRegion, cursor: Address) {
        use crate::util::constants::LOG_BITS_IN_WORD;
        const_assert!(Block::LOG_BYTES - MARK_SPEC.log_bytes_in_region >= LOG_BITS_IN_WORD);
        cfg_if::cfg_if! { if #[cfg(feature = "compressor_art_marking")] {
            let used = self.calculate_offset_vector_art(region, cursor);
        } else {
            let used = if self.supports_clmul {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    // SAFETY: We checked the processor supports the
                    // necessary instructions.
                    self.calculate_offset_vector_clmul(region, cursor)
                }
                #[cfg(not(target_arch = "x86_64"))]
                { unreachable!("Shouldn't have self.supports_clmul = true on non-x86_64") }
            } else {
                self.calculate_offset_vector_base(region, cursor)
            };
        }}
        self.calculated.store(true, Ordering::Relaxed);
        self.select_region(region, used);
    }

    pub fn select_region(&self, region: CompressorRegion, used: Offset) {
        if cfg!(feature = "compressor_region_selection") {
            let selected = match self.compact_limit {
                CompactLimit::AlwaysCompact => true,
                CompactLimit::Percentage(limit) => {
                    let percent = (used / (CompressorRegion::BYTES / 100) as Offset) as u8;
                    percent < limit
                }
            };
            SELECTED_SPEC.store_atomic::<u8>(region.start(), selected as u8, Ordering::Relaxed);
        }
    }

    cfg_if::cfg_if! { if #[cfg(feature = "compressor_art_marking")] {
        fn calculate_offset_vector_art(&self, region: CompressorRegion, cursor: Address) -> Offset {
            let mut offset: Offset = 0;
            MARK_SPEC.scan_words(
                region.start(),
                cursor.align_up(Block::BYTES),
                &mut |word, addr, bits| match bits {
                    Bits::Range { start, end } => {
                        unreachable!("Blocks should be bitmap-word aligned, but we got a misaligned {word}[{start}:{end}] instead")
                    }
                    Bits::All => {
                        if addr.is_aligned_to(Block::BYTES) {
                            OFFSET_VECTOR_SPEC.store_atomic::<Offset>(
                                addr,
                                offset,
                                Ordering::Relaxed,
                            );
                        }
                        // The live data in a block is proportional to how many
                        // bits have been marked in the block.
                        offset += BYTES_PER_MARK_BIT * word.count_ones() as Offset
                    }
                },
            );
            offset
        }
    } else {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "pclmulqdq,popcnt")]
        fn calculate_offset_vector_clmul(&self, region: CompressorRegion, cursor: Address) -> Offset {
            // We need a local function to use #[target_feature], which in turn
            // allows rustc to inline `clmul_step` into this function, as the two
            // functions have matching #[target_feature]s.
            #[target_feature(enable = "pclmulqdq,popcnt")]
            fn inner(offset: &mut Offset, carry: &mut i64, word: usize, addr: Address) {
                // Write the state of the start of the block.
                // We extract one in-object bit from the carry, in order to
                // match the format used by  `Transducer::encode`,
                let encoded = *offset + ((*carry as Offset) & 1);
                if addr.is_aligned_to(Block::BYTES) {
                    OFFSET_VECTOR_SPEC.store_atomic::<Offset>(addr, encoded, Ordering::Relaxed);
                }
                clmul_step(offset, carry, word)
            }

            let mut offset: Offset = 0;
            let mut carry: i64 = 0;
            MARK_SPEC.scan_words(
                region.start(),
                cursor.align_up(Block::BYTES),
                &mut |word, addr, bits| match bits {
                    Bits::Range { start, end } => {
                        panic!("should be word aligned, got {word}[{start}:{end}] instead")
                    }
                    Bits::All => inner(&mut offset, &mut carry, word, addr),
                },
            );
            offset
        }

        fn calculate_offset_vector_base(&self, region: CompressorRegion, cursor: Address) -> Offset {
            use crate::util::linear_scan::RegionIterator;
            let mut state = Transducer::new();
            let first_block = Block::from_aligned_address(region.start());
            let last_block = Block::from_aligned_address(cursor);
            for block in RegionIterator::<Block>::new(first_block, last_block) {
                OFFSET_VECTOR_SPEC.store_atomic::<Offset>(
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
    }}

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
    }

    pub fn is_forwarding_region(&self, region: CompressorRegion) -> bool {
        !cfg!(feature = "compressor_region_selection")
            || SELECTED_SPEC.load_atomic::<u8>(region.start(), Ordering::Relaxed) != 0
    }

    pub fn forward<const CAN_CLMUL: bool>(&self, address: Address) -> Address {
        debug_assert!(
            self.calculated.load(Ordering::Relaxed),
            "forward() should only be called when we have calculated an offset vector"
        );
        let region = CompressorRegion::from_unaligned_address(address);
        if !self.is_forwarding_region(region) {
            address
        } else {
            // This could be less of a mess, and with more compile-time checks,
            // if enums could be used in const generics (in stable Rust). Alas.
            cfg_if::cfg_if! { if #[cfg(feature = "compressor_art_marking")] {
                let offset = self.forward_art(address);
            } else {
                let offset = if CAN_CLMUL {
                    #[cfg(target_arch = "x86_64")]
                    unsafe { self.forward_clmul(address) }
                    // In particular, I would like to make the equivalent of
                    // `CAN_CLMUL = false` on non-x86_64 unrepresentable.
                    // (The same applies for the test in `calculate_offset_vector` too.)
                    #[cfg(not(target_arch = "x86_64"))]
                    unreachable!("Shouldn't have CAN_CLMUL = true on non-x86_64")
                } else {
                    self.forward_base(address)
                };
            }}
            region.start() + offset as usize
        }
    }

    cfg_if::cfg_if! { if #[cfg(feature = "compressor_art_marking")] {
        fn forward_art(&self, address: Address) -> Offset {
            let block = Block::from_unaligned_address(address);
            let mut offset = OFFSET_VECTOR_SPEC.load_atomic::<Offset>(block.start(), Ordering::Relaxed);
            MARK_SPEC.scan_words(block.start(), address, &mut |word, _, bits| match bits {
                Bits::Range { start, end } => {
                    // The start of a block should always be bitmap-word-aligned;
                    // only the address to forward will be (very likely) bitmap-word-unaligned.
                    assert_eq!(start, 0);
                    // We count the bits preceding the bit corresponding
                    // to the address to forward.
                    let mask = (1 << end) - 1;
                    offset += BYTES_PER_MARK_BIT * (word & mask).count_ones() as Offset
                }
                Bits::All => offset += BYTES_PER_MARK_BIT * word.count_ones() as Offset,
            });
            offset
        }
    } else {
        fn forward_base(&self, address: Address) -> Offset {
            let block = Block::from_unaligned_address(address);
            let mut state = Transducer::decode(
                OFFSET_VECTOR_SPEC.load_atomic::<Offset>(block.start(), Ordering::Relaxed),
                block.start(),
            );
            // The transducer in this implementation computes the distance of
            // an object from the start of a region; whereas Total-Live-Data in the
            // paper computes the distance of the object from the start of the block.
            MARK_SPEC.scan_non_zero_values::<u8>(block.start(), address, &mut |addr: Address| {
                state.visit_mark_bit(addr)
            });
            state.offset
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "pclmulqdq,popcnt")]
        fn forward_clmul(&self, address: Address) -> Offset {
            debug_assert!(self.supports_clmul);
            let block = Block::from_unaligned_address(address);
            let (mut offset, mut carry) = {
                let state = Transducer::decode(
                    OFFSET_VECTOR_SPEC.load_atomic::<Offset>(block.start(), Ordering::Relaxed),
                    block.start(),
                );
                (state.offset, if state.in_object { -1i64 } else { 0i64 })
            };
            MARK_SPEC.scan_words(block.start(), address, &mut |word, _, bits| match bits {
                Bits::Range { start, end } => {
                    assert_eq!(start, 0);
                    let mask = (1 << end) - 1;
                    clmul_step(&mut offset, &mut carry, word & mask)
                }
                Bits::All => clmul_step(&mut offset, &mut carry, word),
            });
            offset
        }
    }}

    pub fn scan_marked_objects(
        &self,
        start: Address,
        end: Address,
        f: &mut impl FnMut(ObjectReference),
    ) {
        cfg_if::cfg_if! { if #[cfg(feature = "compressor_art_marking")] {
            let visit = &mut |start: Address, size: usize| {
                let mut cursor = start;
                while cursor < start + size {
                    let object = ObjectReference::from_raw_address(cursor).unwrap();
                    let object_size = VM::VMObjectModel::get_current_size(object);
                    f(object);
                    cursor += object_size;
                }
            };
            // This is the VisitLiveStrides algorithm in the Android Runtime, in
            // <https://cs.android.com/android/platform/superproject/main/+/main:art/runtime/gc/collector/mark_compact-inl.h>
            let mut stride_start: Option<Address> = None;
            let mut stride_size: usize = 0;
            MARK_SPEC.scan_words(
                start,
                end,
                &mut |word, addr, bits| match bits {
                    Bits::Range { start, end } => {
                        unreachable!("Regions should be bitmap-word aligned, but we got a misaligned {word}[{start}:{end}] instead")
                    }
                    Bits::All => {
                        if word == usize::MAX {
                            // All bits in the word are marked.
                            stride_start = stride_start.or(Some(addr));
                            stride_size += BYTES_PER_MARK_BIT as usize * usize::BITS as usize;
                        } else {
                            let mut word = word;
                            let mut index_in_word: usize = 0;
                            while word != 0 {
                                // Discard zeroes.
                                let shift = word.trailing_zeros();
                                index_in_word += shift as usize;
                                word >>= shift;
                                if let Some(start) = stride_start {
                                    if shift > 0 {
                                        visit(start, stride_size);
                                        stride_start = Some(addr + BYTES_PER_MARK_BIT as usize * index_in_word);
                                        stride_size = 0;
                                    }
                                } else {
                                    stride_start = Some(addr + BYTES_PER_MARK_BIT as usize * index_in_word);
                                    stride_size = 0;
                                }
                                // Now discard ones.
                                let shift = word.trailing_ones();
                                // We discarded all the trailing zeroes, and the word is
                                // non-zero, so we should have at least one trailing one.
                                debug_assert_ne!(shift, 0);
                                index_in_word += shift as usize;
                                word >>= shift;
                                stride_size += BYTES_PER_MARK_BIT as usize * shift as usize;
                            }
                            if index_in_word < usize::BITS as usize && stride_start.is_some() {
                                // We haven't consumed all of the word, and the word has
                                // zeroes at the most significant end. The stride ends here.
                                visit(stride_start.unwrap(), stride_size);
                                stride_start = None;
                                stride_size = 0;
                            }
                        }
                    }
                }
            );
            if let Some(start) = stride_start {
                visit(start, stride_size);
            }
        } else {
            // Recall that we mark the first and last words of each live object.
            // We skip over every second marked word, in order to only visit
            // the words at the starts of objects.
            let mut in_object = false;
            MARK_SPEC.scan_non_zero_values::<u8>(start, end, &mut |addr: Address| {
                if !in_object {
                    let object = ObjectReference::from_raw_address(addr).unwrap();
                    f(object);
                }
                in_object = !in_object;
            });
        }}
    }

    // The Big Loop(tm) for OnePass.
    #[cfg(feature = "compressor_art_marking")]
    pub fn calculate_and_walk_offset_vector(
        &self,
        _start: Address,
        _cursor: Address,
        _fix_threaded_pointers: &impl Fn(ObjectReference) -> usize,
        _claim_block: &impl Fn(Block),
        _move_object: &mut impl FnMut(ObjectReference, usize),
    ) {
        todo!();
    }
    #[cfg(not(feature = "compressor_art_marking"))]
    pub fn calculate_and_walk_offset_vector(
        &self,
        start: Address,
        cursor: Address,
        fix_threaded_pointers: &impl Fn(ObjectReference),
        claim_block: &impl Fn(Block),
        move_object: &mut impl FnMut(ObjectReference),
    ) {
        use crate::util::linear_scan::RegionIterator;
        let first_block = Block::from_aligned_address(start);
        let last_block = Block::from_aligned_address(cursor);
        self.calculated.store(true, Ordering::Relaxed);

        let mut state = Transducer::new();
            for block in RegionIterator::<Block>::new(first_block, last_block) {
                OFFSET_VECTOR_SPEC.store_atomic::<Offset>(
                    block.start(),
                    state.encode(block.start()),
                    Ordering::Relaxed,
                );
                // We need to visit the objects in this block twice; make a
                // temporary copy of the transducer so that we can iterate
                // a second time, without updating the liveness information.
                let mut second_state = state.clone();
                MARK_SPEC.scan_non_zero_values::<u8>(
                    block.start(),
                    block.end(),
                    &mut |addr: Address| {
                        state.visit_mark_bit(addr);
                        if state.in_object {
                            let o = ObjectReference::from_raw_address(addr).unwrap();
                            VM::VMObjectModel::finalise_threading_list(o);
                            fix_threaded_pointers(o);
                        }
                    },
                );
                claim_block(block);
                MARK_SPEC.scan_non_zero_values::<u8>(
                    block.start(),
                    block.end(),
                    &mut |addr: Address| {
                        second_state.visit_mark_bit(addr);
                        if second_state.in_object {
                            let o = ObjectReference::from_raw_address(addr).unwrap();
                            move_object(o);
                        }
                    },
                );
            }
    }

    pub fn has_calculated_forwarding_addresses(&self) -> bool {
        self.calculated.load(Ordering::Relaxed)
    }
}

// #[target_feature] allows rustc to generate the POPCNT and PCLMULQDQ
// instructions inline in this function.
#[cfg(target_arch = "x86_64")]
#[cfg(not(feature = "compressor_art_marking"))]
#[target_feature(enable = "pclmulqdq,popcnt")]
fn clmul_step(offset: &mut Offset, carry: &mut i64, word: usize) {
    use std::arch::x86_64;
    // Compute the prefix sum of this word of mark bitmap.
    let ones = x86_64::_mm_set1_epi8(-1i8);
    let vector = x86_64::_mm_set_epi64x(0, word as i64);
    let mask: i64 = x86_64::_mm_cvtsi128_si64(x86_64::_mm_clmulepi64_si128(vector, ones, 0));
    // Carry-in from the last word. If the last word ended in the
    // middle of an object, we need to invert the in/out-of-object
    // states in this word.
    let flipped = mask ^ *carry;
    // Produce the carry-out for the next word. This shift replicates
    // the most significant bit to all bit positions.
    *carry = flipped >> 63;
    // Now count the in-object bits. The marked bits on either
    // end of an object are both in an object, despite that the
    // prefix sum for the bit at the end of an object will be zero,
    // so we bitwise-or the original word with the prefix sum to
    // find all in-object bits.
    *offset += (((flipped as usize | word).count_ones()) * 8) as Offset;
}
