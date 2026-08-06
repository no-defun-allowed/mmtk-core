#[cfg(feature = "object_pinning")]
use crate::policy::compressor::stubtable;
#[cfg(feature = "object_pinning")]
use crate::util::constants::BYTES_IN_PAGE;
use crate::util::constants::BYTES_IN_WORD;
use crate::util::linear_scan::Region;
use crate::util::metadata::side_metadata::ranges::Bits;
#[cfg(feature = "object_pinning")]
use crate::util::metadata::side_metadata::spec_defs::COMPRESSOR_PAGE_PINNED;
use crate::util::metadata::side_metadata::spec_defs::{
    COMPRESSOR_MARK, COMPRESSOR_OFFSET_VECTOR, COMPRESSOR_SELECTED,
};
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::options::PinningMode;
use crate::util::{Address, ObjectReference};
use crate::vm::object_model::ObjectModel;
use crate::vm::VMBinding;
use atomic::Ordering;
use itertools::Itertools;
#[cfg(debug_assertions)]
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(debug_assertions)]
use std::sync::Mutex;
#[cfg(feature = "object_pinning")]
use std::sync::RwLock;

#[cfg(debug_assertions)]
pub(super) static COMPUTING_FORWARDING_INFO: AtomicBool = AtomicBool::new(false);

#[cfg(debug_assertions)]
lazy_static! {
    pub(super) static ref FORWARDING_MAP: Mutex<HashMap<Address, Address>> =
        Mutex::new(HashMap::new());
}

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

/// Pinned block bit in an offset. If this bit it set, then the block is pinned
/// and all objects that *start* in this block are considered pinned.
pub(crate) const PINNED_BLOCK_BIT: u32 = 1 << 2;
/// Mask for [`PINNED_BLOCK_BIT`].
pub(crate) const PINNED_BLOCK_MASK: u32 = !PINNED_BLOCK_BIT;
/// Metadata bits in an offset vector entry.
pub(crate) const OFFSET_METADATA_BITS: u32 = PINNED_BLOCK_BIT | 0b11;
/// Mask for the offset value in an offset vector entry.
pub(crate) const OFFSET_MASK: u32 = !OFFSET_METADATA_BITS;

/// The minimum size of a hole that we will consider for allocation.
pub(super) const MINIMUM_HOLE_SIZE: usize = 1;

pub(crate) type FreeList = Vec<(Address, Address)>;
pub(crate) fn singleton_free_list(r: CompressorRegion, cursor: Offset) -> FreeList {
    vec![(r.start() + cursor as usize, r.end())]
}

#[cfg(feature = "object_pinning")]
pub(super) fn does_new_address_intersect_pinned_objects<VM: VMBinding>(
    start: Address,
    size: usize,
) -> (bool, Option<ObjectReference>) {
    let start_address = start;
    let end_address = start_address + size;

    // SAFETY: This function is only called after the Closure phase, so no one
    // will be modifying the pinning bits of objects.
    let next_pinned_address = unsafe {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC
            .extract_side_spec()
            .find_next_non_zero_value::<u8>(start + BYTES_IN_WORD, CompressorRegion::BYTES)
    };

    // SAFETY: This function is only called after the Closure phase, so no one
    // will be modifying the pinning bits of objects.
    let prev_pinned_address = unsafe {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC
            .extract_side_spec()
            .find_prev_non_zero_value::<u8>(start, CompressorRegion::BYTES)
    };
    // SAFETY: We are only creating ObjectReferences from addresses within the MMTk heap
    let prev_pinned_object = prev_pinned_address
        .map(|addr| unsafe { ObjectReference::from_raw_address_unchecked(addr) });
    let prev_pinned_object_end =
        prev_pinned_address.map(|addr| addr + get_object_size_from_mark_bits(addr));

    if let Some(prev_pinned_object_end) = prev_pinned_object_end {
        if prev_pinned_object_end > start_address {
            return (true, prev_pinned_object);
        }
    }

    if let Some(next_pinned_address) = next_pinned_address {
        if next_pinned_address < end_address {
            // SAFETY: We are only creating ObjectReferences from addresses within the MMTk heap
            let next_pinned_object =
                unsafe { ObjectReference::from_raw_address_unchecked(next_pinned_address) };
            return (true, Some(next_pinned_object));
        }
    }

    (false, None)
}

#[cfg(feature = "object_pinning")]
pub(super) fn does_new_address_intersect_pinned_pages(
    start: Address,
    size: usize,
) -> (bool, Option<Address>) {
    let start_address = start;
    let end_address = start_address + size;
    let mut current_address = start_address.align_down(BYTES_IN_PAGE);
    while current_address < end_address {
        if is_page_pinned(current_address) {
            return (true, Some(current_address));
        }
        current_address += BYTES_IN_PAGE;
    }
    (false, None)
}

pub(super) fn get_object_size_from_mark_bits(start: Address) -> usize {
    debug_assert!(
        is_address_marked(start, Ordering::Relaxed),
        "The start address {} should have its mark bit set when calculating object size from mark bits.",
        start
    );
    let region = CompressorRegion::from_unaligned_address(start);
    let search_start = start + BYTES_IN_WORD;
    // SAFETY: This is called for either pinned objects or when we're in the Forward phase.
    // Either way, no one will be modifying the mark bits for that object.
    let end = unsafe {
        MARK_SPEC
            .find_next_non_zero_value::<u8>(search_start, region.end() - search_start + 1usize)
            .expect("Failed to find first non-zero bit")
    };
    debug_assert_ne!(
        start, end,
        "Object start and end should be different: {:#x}",
        start
    );
    let size = end - start + BYTES_IN_WORD;
    size
}

pub(super) fn get_object_size<VM: VMBinding>(
    object: ObjectReference,
    stub_table: &RwLock<stubtable::StubTable<VM>>,
) -> usize {
    if let Some(size) = stub_table.read().unwrap().get_size(object) {
        debug_assert!(
            is_object_pinned::<VM>(object),
            "Object {:?} in stub table is not pinned!",
            object,
        );
        size.get()
    } else {
        VM::VMObjectModel::get_current_size(object)
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
    /// Whether or not the transducer is currently inside a pinned object
    /// or not. We use this to skip over pinned objects when calculating the
    /// offset-vector or forwarding addresses
    #[cfg(feature = "object_pinning")]
    in_pinned_object: bool,
    /// Whether or not the transducer is currently inside a pinned block
    /// or not.
    #[cfg(feature = "object_pinning")]
    pinned_block: bool,
}

#[cfg(not(feature = "compressor_art_marking"))]
impl Transducer {
    pub fn new() -> Self {
        Self {
            offset: 0,
            last_bit_visited: Address::ZERO,
            in_object: false,
            #[cfg(feature = "object_pinning")]
            in_pinned_object: false,
            #[cfg(feature = "object_pinning")]
            pinned_block: false,
        }
    }
    pub fn visit_mark_bit<VM: VMBinding>(
        &mut self,
        address: Address,
        _add_to_forwarding_map: bool,
        stub_table: &RwLock<stubtable::StubTable<VM>>,
    ) {
        if _add_to_forwarding_map {
            debug!(
                "Visiting mark bit at address {}, in_object: {}, last_bit_visited: {}",
                address, self.in_object, self.last_bit_visited
            );
        }

        // Skip if this is the same address as the last one visited. This happens
        // when we chase the end of an object while calculating the live data that
        // *starts* in an unpinned block.
        if self.last_bit_visited == address {
            if _add_to_forwarding_map {
                debug!(
                    "Skipping mark bit at {address} because it is the same as the last bit visited"
                );
            }
            debug_assert!(
                self.in_object,
                "If we are skipping a mark bit, we should be in an object. last_bit_visited: {}, address: {}",
                self.last_bit_visited, address
            );
            self.in_object = false;
            self.in_pinned_object = false;
            return;
        }

        if self.in_object {
            // The size of an object is the distance between the end and
            // start of the object, and the last word of the object is one
            // word prior to the end of the object. Thus we must add an
            // extra word, in order to compute the size of the object based
            // on the distance between its first and last words.
            let first_word = self.last_bit_visited;
            let last_word = address;
            let size = last_word - first_word + BYTES_IN_WORD;
            #[cfg(debug_assertions)]
            let region = CompressorRegion::from_unaligned_address(first_word);

            if !self.in_pinned_object {
                self.offset += size as Offset;
                #[cfg(debug_assertions)]
                if _add_to_forwarding_map {
                    let mut map = FORWARDING_MAP.lock().unwrap();
                    if map.contains_key(&first_word) {
                        debug_assert_eq!(
                            map[&first_word],
                            region.start() + self.offset as usize - size,
                            "If the object has already been forwarded, it should have the same forwarding address as before. Old: 0x{:x}, new: 0x{:x}",
                            map[&first_word],
                            region.start() + self.offset as usize - size,
                        );
                    } else {
                        map.insert(first_word, region.start() + self.offset as usize - size);
                    }
                    drop(map);
                    debug!(
                        "Move object at {first_word} -> {} (size {size}): {:#x}",
                        region.start() + self.offset as usize - size,
                        self.offset
                    );
                    let (intersects_pinned, pinned_object) =
                        does_new_address_intersect_pinned_objects::<VM>(
                            region.start() + self.offset as usize - size,
                            size,
                        );
                    debug_assert!(
                        !intersects_pinned,
                        "Object at {first_word} moved to {} (size {size}) which intersects with a pinned object {:?}!",
                        region.start() + self.offset as usize - size,
                        pinned_object.unwrap()
                    );
                    debug_assert!(
                        !does_new_address_intersect_pinned_pages(region.start() + self.offset as usize - size, size).0,
                        "Object at {first_word} moved to {} (size {size}) which intersects a pinned page!",
                        region.start() + self.offset as usize - size,
                    );
                    debug_assert!(
                        first_word >= region.start() + self.offset as usize - size,
                        "Object {first_word} moved to {} (size {size}) which is after its original location, potentially overwriting live data!",
                        region.start() + self.offset as usize - size,
                    )
                }
            } else {
                #[cfg(debug_assertions)]
                if _add_to_forwarding_map {
                    debug!("Skip pinned object at 0x{first_word:#x} -> 0x{first_word:#x} (size {size}): {:#x}", self.offset);
                    let mut map = FORWARDING_MAP.lock().unwrap();
                    if map.contains_key(&first_word) {
                        debug_assert_eq!(
                            map[&first_word],
                            first_word,
                            "Pinned object at 0x{first_word:#x} should have been forwarded to itself. Old: 0x{:x}, new: 0x{:x}",
                            map[&first_word],
                            first_word,
                        );
                    } else {
                        map.insert(first_word, first_word);
                    }
                }
            }
        }
        self.in_object = !self.in_object;
        #[cfg(feature = "object_pinning")]
        if self.in_object {
            // SAFETY: If we're currently within an object, we have just found the starting mark-bit
            // of the next live object. Hence, the address is a valid ObjectReference.
            let object = unsafe { ObjectReference::from_raw_address_unchecked(address) };
            self.in_pinned_object =
                is_object_pinned::<VM>(object) || is_object_in_pinned_block::<VM>(object);
        } else {
            self.in_pinned_object = false;
        }
        self.last_bit_visited = address;
    }

    pub fn visit_mark_bit_forwarding<VM: VMBinding>(&mut self, address: Address) {
        debug!(
            "Visiting mark bit at address {}, in_object: {}, last_bit_visited: {}, offset: 0x{:x}",
            address, self.in_object, self.last_bit_visited, self.offset
        );

        // Skip if this is the same address as the last one visited. This happens
        // when we chase the end of an object while calculating the live data that
        // *starts* in an unpinned block.
        if self.last_bit_visited == address {
            debug!("Skipping mark bit at {address} because it is the same as the last bit visited");
            debug_assert!(
                self.in_object,
                "If we are skipping a mark bit, we should be in an object. last_bit_visited: {}, address: {}",
                self.last_bit_visited, address
            );
            self.in_object = false;
            self.in_pinned_object = false;
            return;
        }

        if self.in_object {
            // The size of an object is the distance between the end and
            // start of the object, and the last word of the object is one
            // word prior to the end of the object. Thus we must add an
            // extra word, in order to compute the size of the object based
            // on the distance between its first and last words.
            let first_word = self.last_bit_visited;
            let last_word = address;
            let size = last_word - first_word + BYTES_IN_WORD;
            if !self.in_pinned_object {
                self.offset += size as Offset;
            }
        }
        self.in_object = !self.in_object;
        #[cfg(feature = "object_pinning")]
        if self.in_object {
            // SAFETY: If we're currently within an object, we have just found the starting mark-bit
            // of the next live object. Hence, the address is a valid ObjectReference.
            let object = unsafe { ObjectReference::from_raw_address_unchecked(address) };
            self.in_pinned_object =
                is_object_pinned::<VM>(object) || is_object_in_pinned_block::<VM>(object);
        } else {
            self.in_pinned_object = false;
        }
        self.last_bit_visited = address;
    }

    pub fn encode(&self, current_position: Address) -> Offset {
        debug_assert!(crate::util::conversions::raw_is_aligned(
            self.offset as usize,
            crate::util::constants::MIN_OBJECT_SIZE
        ));
        debug_assert!(
            self.offset & OFFSET_METADATA_BITS == 0,
            "The offset should have at least 3 free bits for encoding metadata."
        );

        if self.in_object {
            #[allow(unused_mut)]
            let mut offset = self.offset + 1;
            // We count the space between the last mark bit and
            // the current address as live when we stop in the
            // middle of an object. But we only add this delta
            // when we are not in a pinned block.
            let delta = (current_position - self.last_bit_visited) as Offset;
            #[cfg(feature = "object_pinning")]
            if self.in_pinned_object {
                offset += 0b10;
            }
            #[cfg(feature = "object_pinning")]
            if self.pinned_block {
                offset |= PINNED_BLOCK_BIT;
            } else {
                offset += delta;
            }
            offset
        } else {
            #[allow(unused_mut)]
            let mut offset = self.offset;
            #[cfg(feature = "object_pinning")]
            if self.pinned_block {
                offset |= PINNED_BLOCK_BIT;
            }
            offset
        }
    }

    pub fn decode(offset: Offset, _current_position: Address) -> Self {
        Transducer {
            offset: offset & OFFSET_MASK,
            last_bit_visited: Address::ZERO,
            in_object: (offset & 0b1) == 0b1,
            #[cfg(feature = "object_pinning")]
            in_pinned_object: (offset & 0b10) == 0b10,
            #[cfg(feature = "object_pinning")]
            pinned_block: (offset & PINNED_BLOCK_BIT) == PINNED_BLOCK_BIT,
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
#[cfg(feature = "object_pinning")]
pub(crate) const PINNED_PAGE_SPEC: SideMetadataSpec = COMPRESSOR_PAGE_PINNED;

pub struct ForwardingMetadata<VM: VMBinding> {
    compact_limit: CompactLimit,
    calculated: AtomicBool,
    vm: PhantomData<VM>,
    supports_clmul: bool,
    #[cfg(feature = "object_pinning")]
    pub stub_table: RwLock<stubtable::StubTable<VM>>,
    #[cfg(feature = "object_pinning")]
    pub pinning_mode: PinningMode,
    size_classes: [AtomicUsize; 32],
}

#[cfg(feature = "object_pinning")]
pub(super) fn is_page_pinned(address: Address) -> bool {
    debug_assert!(
        address.is_aligned_to(BYTES_IN_PAGE),
        "Address {} should be aligned to page size when checking for page pinning.",
        address,
    );
    PINNED_PAGE_SPEC.load_atomic::<u8>(address, Ordering::Relaxed) != 0
}

#[cfg(feature = "object_pinning")]
pub(super) fn is_object_pinned<VM: VMBinding>(object: ObjectReference) -> bool {
    VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.is_object_pinned::<VM>(object)
}

#[cfg(feature = "object_pinning")]
fn pin_object_inner<VM: VMBinding>(object: ObjectReference) -> bool {
    VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.pin_object::<VM>(object)
}

#[cfg(feature = "object_pinning")]
pub(super) fn pin_object<VM: VMBinding>(object: ObjectReference) -> bool {
    let mut pinned = false;
    while !is_object_pinned::<VM>(object) {
        pinned = pin_object_inner::<VM>(object);
    }
    pinned
}

pub(super) fn is_object_marked<VM: VMBinding>(object: ObjectReference, order: Ordering) -> bool {
    MARK_SPEC.load_atomic::<u8>(object.to_object_start::<VM>(), order) != 0
}

pub(super) fn is_address_marked(address: Address, order: Ordering) -> bool {
    MARK_SPEC.load_atomic::<u8>(address, order) != 0
}

pub(super) fn pin_block(block: Block) -> bool {
    OFFSET_VECTOR_SPEC
        .fetch_update_atomic::<Offset, _>(block.start(), Ordering::SeqCst, Ordering::Relaxed, |v| {
            if v & PINNED_BLOCK_BIT == 0 {
                Some(v | PINNED_BLOCK_BIT)
            } else {
                None
            }
        })
        .is_ok()
}

pub(super) fn is_block_pinned(block: Block) -> bool {
    OFFSET_VECTOR_SPEC.load_atomic::<Offset>(block.start(), Ordering::Relaxed) & PINNED_BLOCK_BIT
        == PINNED_BLOCK_BIT
}

pub(super) fn is_object_in_pinned_block<VM: VMBinding>(object: ObjectReference) -> bool {
    let block = Block::from_unaligned_address(object.to_object_start::<VM>());
    is_block_pinned(block)
}

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new(
        compact_limit: CompactLimit,
        _use_clmul: bool,
        _pinning_mode: PinningMode,
    ) -> ForwardingMetadata<VM> {
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
            #[cfg(feature = "object_pinning")]
            stub_table: RwLock::new(stubtable::StubTable::new()),
            #[cfg(feature = "object_pinning")]
            pinning_mode: _pinning_mode,
            size_classes: [const { AtomicUsize::new(0) }; 32],
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

        #[cfg(feature = "object_pinning")]
        self.pin_object_if_needed(object);
    }

    pub fn mark_rest_of_object_known_size(&self, object: ObjectReference, size: usize) {
        // We implement two styles of mark bitmap:
        if cfg!(feature = "compressor_art_marking") {
            // - When `cfg(feature = "compressor_art_marking")`, we use a style as in
            //   the Android Runtime and Clozure Common Lisp, where we mark every
            //   bit corresponding to a word in each live object.
            //
            // XXX: this will SeqCst and we don't need that.
            MARK_SPEC.bset_metadata(object.to_object_start::<VM>(), size);
        } else {
            // - When `cfg(not(feature = "compressor_art_marking"))`, we follow the style
            //   in the original Compressor paper, where we mark bits corresponding to
            //   the first and last words of each live object. The live data then
            //   corresponds to the bits between and including each pair of set bits.
            let last_word_of_object = object.to_object_start::<VM>() + size - BYTES_IN_WORD;
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
            MARK_SPEC.fetch_or_atomic::<u8>(last_word_of_object, 1, Ordering::Relaxed);
        }
    }

    #[cfg(feature = "object_pinning")]
    fn pin_object_if_needed(&self, object: ObjectReference) {
        match self.pinning_mode {
            PinningMode::NoPinning => {}
            PinningMode::RandomObjectPinning(fraction) => {
                // Pin the object with probability of pin_fraction
                let should_pin = rand::random_bool(fraction);
                if should_pin {
                    pin_object::<VM>(object);
                    debug!(
                        "Pinning object at {} of size {} bytes",
                        object.to_raw_address(),
                        VM::VMObjectModel::get_current_size(object)
                    );
                }
            }
            PinningMode::RandomPagePinning(..) => {
                // Any object that spans a pinned page should be pinned
                let start_page = object.to_object_start::<VM>().align_down(BYTES_IN_PAGE);
                let last_word_of_object = object.to_object_start::<VM>()
                    + VM::VMObjectModel::get_current_size(object)
                    - BYTES_IN_WORD;
                let mut should_pin = false;
                let mut current_page = start_page;
                while current_page <= last_word_of_object {
                    if is_page_pinned(current_page) {
                        should_pin = true;
                        break;
                    }
                    current_page += BYTES_IN_PAGE;
                }
                if should_pin {
                    pin_object::<VM>(object);
                    debug!(
                        "Pinning object at {} of size {} bytes",
                        object.to_raw_address(),
                        VM::VMObjectModel::get_current_size(object)
                    );
                }
                debug_assert!(
                    !should_pin || is_object_pinned::<VM>(object),
                    "Object at {} (size {}) should be pinned if we attempted to pin it!",
                    object.to_raw_address(),
                    VM::VMObjectModel::get_current_size(object),
                );
            }
        }
    }

    pub fn calculate_offset_vector(&self, region: CompressorRegion) -> FreeList {
        use crate::util::constants::LOG_BITS_IN_WORD;
        const_assert!(Block::LOG_BYTES - MARK_SPEC.log_bytes_in_region >= LOG_BITS_IN_WORD);
        #[cfg(debug_assertions)]
        COMPUTING_FORWARDING_INFO.store(true, Ordering::SeqCst);
        cfg_if::cfg_if! { if #[cfg(feature = "compressor_art_marking")] {
            let free_list = singleton_free_list(region, self.calculate_offset_vector_art(region));
        } else {
            let free_list = if self.supports_clmul {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    // SAFETY: We checked the processor supports the
                    // necessary instructions.
                    singleton_free_list(region, self.calculate_offset_vector_clmul(region))
                }
                #[cfg(not(target_arch = "x86_64"))]
                { unreachable!("Shouldn't have self.supports_clmul = true on non-x86_64") }
            } else if self.pinning_mode != PinningMode::NoPinning {
                self.calculate_offset_vector_with_pinning(region)
            } else {
                singleton_free_list(region, self.calculate_offset_vector_base(region))
            };
        }}
        self.calculated.store(true, Ordering::Relaxed);
        //self.select_region(region, used);
        free_list
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
        fn calculate_offset_vector_art(&self, region: CompressorRegion) -> Offset {
            let mut offset: Offset = 0;
            MARK_SPEC.scan_words(
                region.start(),
                region.end(),
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
        fn calculate_offset_vector_clmul(&self, region: CompressorRegion) -> Offset {
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
                *offset += clmul_step(carry, word);
            }

            let mut offset: Offset = 0;
            let mut carry: i64 = 0;
            MARK_SPEC.scan_words(
                region.start(),
                region.end(),
                &mut |word, addr, bits| match bits {
                    Bits::Range { start, end } => {
                        panic!("should be word aligned, got {word}[{start}:{end}] instead")
                    }
                    Bits::All => inner(&mut offset, &mut carry, word, addr),
                },
            );
            offset
        }

        fn calculate_offset_vector_base(&self, region: CompressorRegion) -> Offset {
            use crate::util::linear_scan::RegionIterator;
            let mut state = Transducer::new();
            let first_block = Block::from_aligned_address(region.start());
            let last_block = Block::from_aligned_address(region.end());
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
                        state.visit_mark_bit::<VM>(addr, false, &self.stub_table);
                    },
                );
            }
            state.offset
        }

        #[cfg(all(feature = "object_pinning", debug_assertions))]
        fn create_forwarding_map(&self, region: CompressorRegion) {
            debug!("Creating forwarding map for region {}-{}", region.start(), region.end());
            let first_block = Block::from_aligned_address(region.start());
            let last_block = Block::from_aligned_address(region.end());

            let mut block = first_block;
            let mut prev_block = Address::ZERO;
            let mut start = block.start();
            let end = region.end();
            let mut object_start = block.start();
            let mut found_object = false;
            let mut free = vec![(first_block.start(), last_block.start())];
            let mut pinned_ranges = vec![];
            let mut pinned_range_start = Address::ZERO;
            loop {
                if start >= end {
                    // We've reached the end of the region, so we can break out of the loop
                    debug_assert!(
                        !found_object,
                        "We should not have found an object if we reached the end of the region"
                    );
                    break;
                }
                // SAFETY: This is called in the Calculate phase, so no one else is modifying the mark bits.
                let addr = unsafe {
                    MARK_SPEC.find_next_non_zero_value::<u8>(start, end - start)
                };
                if let None = addr {
                    // We've reached the end of the region, so we can break out of the loop
                    debug_assert!(
                        !found_object,
                        "We should not have found an object if we reached the end of the region"
                    );
                    break;
                }
                // SAFETY: We just checked that addr is not None, so we can safely unwrap it
                let addr = unsafe { addr.unwrap_unchecked() };

                debug!("Visiting mark bit at address {}", addr);

                block = Block::from_unaligned_address(addr);
                // Only update the free list if we have moved to a new block.
                // if block.start() != prev_block {
                //     if is_block_pinned(block) {
                //         let hole = free.pop();
                //         // SAFETY: We should always have a hole
                //         let mut hole = unsafe { hole.unwrap_unchecked() };
                //         if hole.0 == block.start() {
                //             hole.0 = block.end();
                //             free.push(hole);
                //         } else {
                //             let free_after = (block.end(), hole.1);
                //             hole.1 = block.start();
                //             if hole.1 <= hole.0 {
                //                 // If the free range before the pinned block is
                //                 // empty, we can just remove it from the free list
                //             } else if hole.1 - hole.0 < MINIMUM_HOLE_SIZE {
                //                 // If the free range before the pinned block is too
                //                 // small, we can just remove it from the free list
                //             } else {
                //                 free.push(hole);
                //             }
                //             free.push(free_after);
                //         }
                //     }
                //     prev_block = block.start();
                // }

                if !found_object {
                    object_start = addr;
                    found_object = true;
                    if is_block_pinned(block) {
                        if pinned_range_start == Address::ZERO {
                            pinned_range_start = block.start();
                        }
                    } else {
                        if pinned_range_start != Address::ZERO {
                            pinned_ranges.push((pinned_range_start, block.start()));
                            pinned_range_start = Address::ZERO;
                        }
                    }
                } else {
                    let object_end = addr;
                    let object_size = object_end - object_start + BYTES_IN_WORD;
                    // SAFETY: The mark bits are only set for valid object references, so the address is a valid ObjectReference.
                    let object = unsafe { ObjectReference::from_raw_address_unchecked(object_start) };
                    let object_end_block = Block::from_unaligned_address(object_end);
                    debug!(
                        "Found {} object at {}-{} (size {}) in block {}",
                        if is_object_in_pinned_block::<VM>(object) { "pinned" } else { "movable" },
                        object_start,
                        object_start + object_size,
                        object_size,
                        Block::from_unaligned_address(object_start).start(),
                    );
                    // If we have an object that straddles from a pinned block
                    // into an unpinned block then we need to update the start
                    // of the last pushed hole in the free list
                    // if is_block_pinned(block) && !is_block_pinned(object_end_block) {
                    //     let hole = free.pop();
                    //     // SAFETY: We should always have a hole
                    //     let mut hole = unsafe { hole.unwrap_unchecked() };
                    //     free.push((object_end + BYTES_IN_WORD, hole.1));
                    // }
                    if is_block_pinned(block) {
                        if pinned_range_start == Address::ZERO {
                            pinned_range_start = block.start();
                        }
                    } else {
                        if pinned_range_start != Address::ZERO {
                            pinned_ranges.push((pinned_range_start, object_end + BYTES_IN_WORD));
                            pinned_range_start = Address::ZERO;
                        }
                    }
                    found_object = false;
                }
                start = addr + BYTES_IN_WORD;
            }

            for pinned_range in pinned_ranges.iter() {
                let hole = free.pop();
                // SAFETY: We should always have a hole
                let mut hole = unsafe { hole.unwrap_unchecked() };
                hole.1 = pinned_range.0;
                if hole.1 <= hole.0 {
                    // If the free range before the pinned block is
                    // empty, we can just remove it from the free list
                } else if hole.1 - hole.0 < MINIMUM_HOLE_SIZE {
                    // If the free range before the pinned block is too
                    // small, we can just remove it from the free list
                } else {
                    free.push(hole);
                }
                free.push((pinned_range.1, last_block.start()));
            }

            debug!(
                "Created forwarding map for region {}-{} with\n    free list: {:?}\n    pinned ranges: {:?}",
                region.start(),
                region.end(),
                free,
                pinned_ranges,
            );

            drop(pinned_ranges);

            block = first_block;
            while block < last_block {
                // SAFETY: This is called in the Calculate phase, so no one else is modifying the mark bits.
                if let Some(addr) = unsafe { MARK_SPEC.find_next_non_zero_value::<u8>(start, region.end() - start + 1_usize) } {
                    block = Block::from_unaligned_address(addr);
                    if !found_object {
                        object_start = addr;
                        found_object = true;
                    } else {
                        let object_end = addr;
                        let object_size = object_end - object_start + BYTES_IN_WORD;
                        // SAFETY: The mark bits are only set for valid object references, so the address is a valid ObjectReference.
                        let object = unsafe { ObjectReference::from_raw_address_unchecked(object_start) };
                        debug_assert!(
                            is_object_marked::<VM>(object, Ordering::Relaxed),
                            "Object at {} (size {}) should be marked",
                            object_start,
                            object_size,
                        );

                        debug!(
                            "Found object at {} (size {}) in block {}",
                            object_start,
                            object_size,
                            block.start(),
                        );

                        if is_object_in_pinned_block::<VM>(object) {
                            let mut map = FORWARDING_MAP.lock().unwrap();
                            if map.contains_key(&object_start) {
                                debug_assert_eq!(
                                    map[&object_start],
                                    object_start,
                                    "Pinned object at {object_start} should have been forwarded to itself. Old: {}, new: {}",
                                    map[&object_start],
                                    object_start,
                                );
                            } else {
                                map.insert(object_start, object_start);
                            }
                        } else {
                            debug_assert!(
                                !is_object_pinned::<VM>(object),
                                "Object at {} (size {}) should not be pinned",
                                object_start,
                                object_size,
                            );

                            let Some((hole_idx, hole)) = free.iter_mut().find_position(|(start, end)| {
                                *end - *start >= object_size
                            }) else {
                                panic!("No free hole found for object at {} (size {}) in region {}-{} with free list: {:?}", object_start, object_size, region.start(), region.end(), free);
                            };

                            // We have found a hole that is large enough for the live
                            // data that *starts* in this block. Store the offset in the
                            // offset vector and update the hole size
                            let forwarding_address = hole.0;
                            hole.0 += object_size;

                            if hole.1 - hole.0 < MINIMUM_HOLE_SIZE {
                                // If the hole is too small, remove it from the free list
                                free.remove(hole_idx);
                            }

                            let mut map = FORWARDING_MAP.lock().unwrap();
                            if map.contains_key(&object_start) {
                                debug_assert_eq!(
                                    map[&object_start],
                                    forwarding_address,
                                    "If the object has already been forwarded, it should have the same forwarding address as before. Old: 0x{:x}, new: 0x{:x}",
                                    map[&object_start],
                                    forwarding_address,
                                );
                            } else {
                                map.insert(object_start, forwarding_address);
                            }

                            let (intersects_pinned, pinned_object) =
                                does_new_address_intersect_pinned_objects::<VM>(
                                    forwarding_address,
                                    object_size,
                                );
                            debug_assert!(
                                !intersects_pinned,
                                "Object at {object_start} moved to {} (size {object_size}) which intersects with a pinned object {:?}!",
                                forwarding_address,
                                pinned_object.unwrap(),
                            );
                            debug_assert!(
                                !does_new_address_intersect_pinned_pages(forwarding_address, object_size).0,
                                "Object at {object_start} moved to {} (size {object_size}) which intersects a pinned page!",
                                forwarding_address,
                            );
                            debug_assert!(
                                object_start >= forwarding_address,
                                "Object {object_start} moved to {} (size {object_size}) which is after its original location, potentially overwriting live data!",
                                forwarding_address,
                            );
                        }

                        start = addr + BYTES_IN_WORD;
                        found_object = false;
                    }
                } else {
                    break;
                }
            }
        }

        fn calculate_offset_vector_with_pinning(&self, region: CompressorRegion) -> FreeList {
            use crate::util::linear_scan::RegionIterator;

            // #[cfg(debug_assertions)]
            // self.create_forwarding_map(region, cursor);

            let mut state = Transducer::new();
            let first_block = Block::from_aligned_address(region.start());
            let last_block = Block::from_aligned_address(region.end());

            let mut last_offset: Offset = 0;
            let mut free = vec![(first_block.start(), last_block.start())];
            let mut block = first_block;
            let mut live_datas = vec![];
            // while block < last_block {
            for b in RegionIterator::<Block>::new(first_block, last_block) {
                if b.start() < block.start() {
                    OFFSET_VECTOR_SPEC.store_atomic::<Offset>(
                        b.start(),
                        last_offset,
                        Ordering::Relaxed,
                    );
                    continue;
                }
                debug!(
                    "Calculating offset for block {}; free list: {:?}",
                    block.start(),
                    free,
                );
                // We have found a pinned block, so we need to update the free list
                if is_block_pinned(block) {
                    state.pinned_block = true;
                    debug!("Found pinned block at {}", block.start());
                    let hole = free.pop();
                    debug_assert!(
                        hole.is_some(),
                        "We should have a free range to pop for block {}",
                        block.start(),
                    );
                    // SAFETY: We should always have a hole
                    let mut hole = unsafe { hole.unwrap_unchecked() };
                    // If the pinned block is at the start of a free range, we
                    // can just move the start of the free range to the end of
                    // the pinned block
                    if hole.0 == block.start() {
                        debug_assert!(
                            hole.1 >= block.end(),
                            "Pinned block {} should not be after the free range {}-{}",
                            block.start(),
                            hole.0,
                            hole.1,
                        );
                        // Move the start of the free range to the end of the
                        // pinned block
                        hole.0 = block.end();
                        if hole.0 <= hole.1 {
                            // If the free range is not empty, we can push it back
                            free.push(hole);
                        }
                    } else {
                        // If the pinned block is in the middle of a free range,
                        // we need to split the free range into two ranges, one
                        // before the pinned block and one after it
                        let free_after = (block.end(), hole.1);
                        hole.1 = block.start();

                        debug_assert!(
                            free_after.0 <= free_after.1,
                            "Free range {}-{} should be valid",
                            free_after.0,
                            free_after.1,
                        );
                        debug_assert!(
                            free_after.1 == last_block.start()
                        );

                        if hole.1 <= hole.0 {
                            // If the free range before the pinned block is
                            // empty, we can just remove it from the free list
                        } else if hole.1 - hole.0 < MINIMUM_HOLE_SIZE {
                            // If the free range before the pinned block is too
                            // small, we can just remove it from the free list
                        } else {
                            free.push(hole);
                        }

                        free.push(free_after);
                    }

                    // XXX(kunals): This is wrong.
                    // We don't need to scan the mark bits in a pinned block as
                    // we don't move any objects inside it
                    state.offset = (block.start() - region.start()) as Offset;
                    last_offset = state.encode(state.last_bit_visited);
                    OFFSET_VECTOR_SPEC.store_atomic::<Offset>(
                        block.start(),
                        last_offset,
                        Ordering::Relaxed,
                    );

                    // Visit mark bits because we need to figure out if a pinned
                    // object straddles into an unpinned block.
                    MARK_SPEC.scan_non_zero_values::<u8>(
                        block.start(),
                        block.end(),
                        &mut |addr: Address| {
                            state.visit_mark_bit::<VM>(addr, true, &self.stub_table);
                        },
                    );
                    if state.in_object {
                        debug!(
                            "Found object {} that straddles pinned block {}",
                            state.last_bit_visited,
                            block.start()
                        );
                        // If we are in an object, we need to find the next mark
                        // bit so that we can add the object to the forwarding map
                        let object_start = state.last_bit_visited;
                        // SAFETY: This is called in the Calculate phase, so no
                        // one else is modifying the mark bits.
                        let object_end = unsafe {
                            MARK_SPEC.find_next_non_zero_value::<u8>(
                                object_start + BYTES_IN_WORD,
                                region.end() - object_start - BYTES_IN_WORD,
                            )
                            .expect("Failed to find next non-zero bit")
                        };
                        let mut new_state = state.clone();
                        new_state.visit_mark_bit(object_end, true, &self.stub_table);
                        // TODO(kunals): It's not that we need to check the end block, but we need to check if
                        // the object *ever* crosses any unpinned block. If it does, those blocks are actually
                        // pinned!
                        let object_end_block = Block::from_unaligned_address(object_end);
                        debug!(
                            "Found object {}-{} (size: {}) that straddles a pinned block {}",
                            object_start,
                            object_end + BYTES_IN_WORD,
                            object_end - object_start + BYTES_IN_WORD,
                            block.start(),
                        );
                        // We have found an object that straddles a pinned
                        // block, so the free range after the pinned block
                        // should start after the end of this object. Note that
                        // since this object could end up straddling multiple
                        // blocks, it is not sufficient to just check the last
                        // block to see if that is pinned or not. Hence just
                        // unconditionally update the free range.
                        let hole = free.pop();
                        debug_assert!(
                            hole.is_some(),
                            "We should have a free range to pop for block {}",
                            block.start(),
                        );
                        // SAFETY: We should always have a hole
                        let mut hole = unsafe { hole.unwrap_unchecked() };
                        hole.0 = object_end + BYTES_IN_WORD;
                        free.push(hole);
                        block = object_end_block;
                    } else {
                        block = block.next();
                    }
                } else {
                    state.offset = 0;
                    state.pinned_block = false;
                    let curr_block = block;
                    let mut orig_state = state.clone();

                    // FIXME(kunals): What happens when the first marked bit in
                    // a block is in the middle of a *pinned object*? We don't
                    // visit mark bits in pinned blocks, so the transducer state
                    // will be incorrect! Do we need to consult the crossing map
                    // here as well?
                    MARK_SPEC.scan_non_zero_values::<u8>(
                        block.start(),
                        block.end(),
                        &mut |addr: Address| {
                            state.visit_mark_bit::<VM>(addr, false, &self.stub_table);
                        },
                    );
                    if state.in_object {
                        debug!(
                            "Found object {} that straddles block {}",
                            state.last_bit_visited,
                            block.start()
                        );
                        // If we are in an object, we need to find the next mark
                        // bit in order to calculate the size of the object.
                        // We then add that size to the offset and set in_object
                        // to false.
                        let object_start = state.last_bit_visited;
                        // SAFETY: This is called in the Calculate phase, so no
                        // one else is modifying the mark bits.
                        let object_end = unsafe {
                            MARK_SPEC.find_next_non_zero_value::<u8>(
                                object_start + BYTES_IN_WORD,
                                region.end() - object_start - BYTES_IN_WORD + 1_usize,
                            )
                            .expect("Failed to find next non-zero bit")
                        };

                        let mut new_state = state.clone();
                        new_state.visit_mark_bit(object_end, false, &self.stub_table);

                        #[cfg(debug_assertions)]
                        {
                            let object_size = VM::VMObjectModel::get_current_size(
                                ObjectReference::from_raw_address(object_start).unwrap()
                            );
                            debug!(
                                "Straddling object {} (size: {}) starting in block {}",
                                object_start,
                                object_size,
                                block.start(),
                            );
                            debug_assert_eq!(
                                object_size,
                                (object_end - object_start + BYTES_IN_WORD) as usize,
                            );
                        }
                        state.offset = new_state.offset;
                        state.last_bit_visited = object_end;
                        // state.in_object = false;
                        // state.in_pinned_object = false;
                        block = Block::from_unaligned_address(object_end);
                    } else {
                        block = block.next();
                    }

                    let live_data = state.offset as usize;
                    live_datas.push(live_data);
                    if live_data == 0 {
                        debug!("Skipping completely dead block {}", curr_block.start());
                        continue;
                    }

                    let Some((hole_idx, hole)) = free.iter_mut().find_position(|(hole_start, hole_end)| {
                        *hole_end - *hole_start >= live_data
                    }) else {
                        panic!("No suitable hole found for live data of size {}", live_data);
                    };

                    debug!(
                        "Found hole {}-{} for live data of size {}; block {}",
                        hole.0,
                        hole.1,
                        live_data,
                        curr_block.start()
                    );

                    // We have found a hole that is large enough for the live
                    // data that *starts* in this block. Store the offset in the
                    // offset vector and update the hole size
                    state.offset = (hole.0 - region.start()) as Offset;
                    debug!(
                        "Storing offset 0x{:x} for block {} (hole {}-{}), in_object: {}, in_pinned_object: {}, last_bit_visited: {}, pinned_block: {}",
                        state.offset,
                        curr_block.start(),
                        hole.0,
                        hole.1,
                        orig_state.in_object,
                        orig_state.in_pinned_object,
                        state.last_bit_visited,
                        state.pinned_block,
                    );
                    orig_state.offset = state.offset;
                    last_offset = orig_state.encode(orig_state.last_bit_visited);
                    OFFSET_VECTOR_SPEC.store_atomic::<Offset>(
                        curr_block.start(),
                        last_offset,
                        Ordering::Relaxed,
                    );
                    hole.0 += live_data;

                    if hole.1 - hole.0 < MINIMUM_HOLE_SIZE {
                        // If the hole is too small, remove it from the free list
                        free.remove(hole_idx);
                    }

                    #[cfg(debug_assertions)]
                    {
                        let tmp_state = state.clone();
                        state = orig_state;
                        MARK_SPEC.scan_non_zero_values::<u8>(
                            curr_block.start(),
                            curr_block.end(),
                            &mut |addr: Address| {
                                state.visit_mark_bit::<VM>(addr, true, &self.stub_table);
                            },
                        );
                        if state.in_object {
                            // If we are in an object, we need to find the next mark
                            // bit in order to calculate the size of the object.
                            // We then add that size to the offset and set in_object
                            // to false.
                            let object_start = state.last_bit_visited;
                            // SAFETY: This is called in the Calculate phase, so no
                            // one else is modifying the mark bits.
                            let object_end = unsafe {
                                MARK_SPEC.find_next_non_zero_value::<u8>(
                                    object_start + BYTES_IN_WORD,
                                    region.end() - object_start - BYTES_IN_WORD + 1_usize,
                                )
                                .expect("Failed to find next non-zero bit")
                            };
                            state.visit_mark_bit(object_end, true, &self.stub_table);
                            debug_assert!(!state.in_object);
                        }
                        state = tmp_state;
                    }
                }
            }
            for (s, e) in free.iter() {
                if *e > *s {
                    self.size_classes[(*e - *s).ilog2() as usize].fetch_add(*e - *s, Ordering::Relaxed);
                }
            }
            trace!("Finished calculating offset vector for region {}: {:#x}\n", region.start(), state.offset);
            free
        }
    }}

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
        #[cfg(debug_assertions)]
        FORWARDING_MAP.lock().unwrap().clear();
        info!("hole sizes: {:?}",
               self.size_classes
               .iter()
               .enumerate()
               .map(|(i, v)| (1usize << i, v.load(Ordering::Relaxed)))
               .collect::<Vec<_>>());
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
        // SAFETY: We are creating an ObjectReference from a valid object since we call this
        // function only for objects
        let object = unsafe { ObjectReference::from_raw_address_unchecked(address) };
        if !self.is_forwarding_region(region) {
            address
        } else if is_object_pinned::<VM>(object) || is_object_in_pinned_block::<VM>(object) {
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
            let mut search_start = block.start();
            let mut state = Transducer::decode(
                OFFSET_VECTOR_SPEC.load_atomic::<Offset>(block.start(), Ordering::Relaxed),
                block.start(),
            );
            debug!(
                "Forwarding object at {address} in block {} with offset 0x{:x}, in_object {}",
                block.start(), state.offset, state.in_object,
            );
            // The transducer in this implementation computes the distance of
            // an object from the start of a region; whereas Total-Live-Data in the
            // paper computes the distance of the object from the start of the block.
            // XXX(kunals): We need to store the offset vector for all blocks to
            // prevent blocks from reading stale offset values
            if state.in_object {
                let end_addr = unsafe {
                    MARK_SPEC.find_next_non_zero_value::<u8>(block.start(), address - block.start() + 1_usize)
                        .expect("Failed to find next non-zero bit")
                };
                debug_assert!(
                    is_address_marked(end_addr, Ordering::Relaxed),
                    "The next non-zero mark bit should be marked in the bitmap; end_addr: {}",
                    end_addr,
                );
                debug_assert!(
                    end_addr < address,
                    "The next non-zero mark bit should be before the address to forward; end_addr: {}, address: {}",
                    end_addr,
                    address,
                );
                debug_assert_eq!(
                    block, Block::from_unaligned_address(end_addr)
                );
                search_start = end_addr + BYTES_IN_WORD;
                state.last_bit_visited = end_addr;
                state.in_object = false;
                state.in_pinned_object = false;
                debug!(
                    "Finishing off the object that straddles the block; last_bit_visited: {}, offset: 0x{:x}",
                    state.last_bit_visited, state.offset,
                );
            }
            MARK_SPEC.scan_non_zero_values::<u8>(
                search_start,
                address,
                &mut |addr: Address| {
                    state.visit_mark_bit_forwarding::<VM>(addr);
                },
            );
            debug_assert!(
                !state.in_object,
                "We should not be in an object after visiting all mark bits up to the address {} to forward; state: offset 0x{:x}, last_bit_visited: {}, in_pinned_object: {}, pinned_block: {}",
                address,
                state.offset,
                state.last_bit_visited,
                state.in_pinned_object,
                state.pinned_block,
            );
            let region = CompressorRegion::from_unaligned_address(address);
            debug_assert!(
                is_address_marked(address, Ordering::Relaxed),
                "The address to forward should be marked in the bitmap."
            );
            debug!("Forwarding object at {address} -> {}: {:#x}\n", region.start() + state.offset as usize, state.offset);
            #[cfg(debug_assertions)]
            {
                let map = FORWARDING_MAP.lock().unwrap();
                debug_assert_eq!(
                    region.start() + state.offset as usize,
                    map[&address],
                    "The forwarding address should match the one in the forwarding map. Expected: {}, actual: {}; block start: {}",
                    map[&address],
                    region.start() + state.offset as usize,
                    block.start(),
                );
            }
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
                    offset += clmul_step(&mut carry, word & mask);
                }
                Bits::All => offset += clmul_step(&mut carry, word),
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
        _end: Address,
        _fix_threaded_pointers: &impl Fn(ObjectReference),
        _claim_block: &impl Fn(Block),
        _move_object: &mut impl FnMut(ObjectReference),
    ) {
        todo!();
    }
    #[cfg(not(feature = "compressor_art_marking"))]
    pub fn calculate_and_walk_offset_vector(
        &self,
        start: Address,
        end: Address,
        fix_threaded_pointers: &impl Fn(ObjectReference),
        claim_block: &impl Fn(Block),
        move_object: &mut impl FnMut(ObjectReference),
    ) {
        use crate::util::linear_scan::RegionIterator;
        let first_block = Block::from_aligned_address(start);
        let last_block = Block::from_aligned_address(end);
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
                    state.visit_mark_bit::<VM>(addr, true, &self.stub_table);
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
                    second_state.visit_mark_bit::<VM>(addr, true, &self.stub_table);
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
#[target_feature(enable = "pclmulqdq,popcnt")]
pub(crate) fn clmul_step(carry: &mut i64, word: usize) -> Offset {
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
    (((flipped as usize | word).count_ones()) * 8) as Offset
}
