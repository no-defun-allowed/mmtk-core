use crate::policy::compressor::forwarding::clmul_step;
use crate::util::constants::BYTES_IN_WORD;
use crate::util::heap::MonotonePageResource;
use crate::util::linear_scan::{Region, RegionIterator};
use crate::util::metadata::side_metadata::ranges::Bits;
use crate::util::metadata::side_metadata::spec_defs::{COMPRESSOR_MARK, OLD_PASS_OFFSET_VECTOR};
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::{Address, ObjectReference};
use crate::vm::object_model::ObjectModel;
use crate::vm::VMBinding;
use atomic::Ordering;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;

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
    // The total live data visited by the transducer.
    live: usize,
    // The address of the last mark bit which the transducer visited.
    last_bit_visited: Address,
    // Whether or not the transducer is currently inside an object
    // (i.e. if it has seen a first bit but no matching last bit yet).
    in_object: bool,
}
impl Transducer {
    pub fn new() -> Self {
        Self {
            live: 0,
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
            self.live += size;
        }
        self.in_object = !self.in_object;
        self.last_bit_visited = address;
    }

    pub fn encode(&self, current_position: Address) -> usize {
        if self.in_object {
            // We count the space between the last mark bit and
            // the current address as live when we stop in the
            // middle of an object.
            self.live + (current_position - self.last_bit_visited) + 1
        } else {
            self.live
        }
    }

    pub fn decode(offset: usize, current_position: Address) -> Self {
        Transducer {
            live: offset & !1,
            last_bit_visited: current_position,
            in_object: (offset & 1) == 1,
        }
    }
}

pub struct ForwardingMetadata<VM: VMBinding> {
    pub(crate) first_address: Address,
    calculated: AtomicBool,
    vm: PhantomData<VM>,
    supports_clmul: bool,
}

// A block in the Compressor is the granularity at which we cache
// the amount of live data preceding an address. We set it to 512 bytes
// following the paper.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
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

#[inline(always)]
pub fn block_number(a: ObjectReference) -> usize {
    a.to_raw_address().as_usize() >> Block::LOG_BYTES
}

pub(crate) const MARK_SPEC: SideMetadataSpec = COMPRESSOR_MARK;
pub(crate) const OFFSET_VECTOR_SPEC: SideMetadataSpec = OLD_PASS_OFFSET_VECTOR;

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new(start: Address, _use_clmul: bool) -> ForwardingMetadata<VM> {
        cfg_if::cfg_if! { if #[cfg(target_arch = "x86_64")] {
            let supports_clmul = _use_clmul
                && is_x86_feature_detected!("pclmulqdq")
                && is_x86_feature_detected!("popcnt");
        } else {
            let supports_clmul = false;
        }}
        ForwardingMetadata {
            first_address: start,
            calculated: AtomicBool::new(false),
            vm: PhantomData,
            supports_clmul,
        }
    }

    pub fn supports_clmul(&self) -> bool {
        self.supports_clmul
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

        MARK_SPEC.fetch_or_atomic::<u8>(last_word_of_object, 1, Ordering::SeqCst);
    }

    pub fn calculate_offset_vector(
        &self,
        pr: &MonotonePageResource<VM>,
        f: &mut impl FnMut(ObjectReference),
    ) {
        let mut state = Transducer::new();
        let first_block = Block::from_aligned_address(self.first_address);
        let last_block = Block::from_aligned_address(pr.cursor());
        self.calculated.store(true, Ordering::Relaxed);
        for block in RegionIterator::<Block>::new(first_block, last_block) {
            OFFSET_VECTOR_SPEC.store_atomic::<usize>(
                block.start(),
                state.encode(block.start()),
                Ordering::Relaxed,
            );
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
        }
    }

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
    }

    pub fn forward<const CAN_CLMUL: bool>(&self, address: Address) -> Address {
        debug_assert!(
            self.calculated.load(Ordering::Relaxed),
            "forward() should only be called when we have calculated an offset vector"
        );
        if CAN_CLMUL {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.forward_clmul(address)
            }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!("Shouldn't have CAN_CLMUL = true on non-x86_64")
        } else {
            self.forward_base(address)
        }
    }

    fn forward_base(&self, address: Address) -> Address {
        let block = Block::from_unaligned_address(address);
        let mut state = Transducer::decode(
            OFFSET_VECTOR_SPEC.load_atomic::<usize>(block.start(), Ordering::Relaxed),
            block.start(),
        );
        // The transducer in this implementation computes the offset
        // relative to the start of the heap; whereas Total-Live-Data in
        // the paper computes the offset relative to the start of the block.
        MARK_SPEC.scan_non_zero_values::<u8>(block.start(), address, &mut |addr: Address| {
            state.visit_mark_bit(addr)
        });
        self.first_address + state.live
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "pclmulqdq,popcnt")]
    fn forward_clmul(&self, address: Address) -> Address {
        debug_assert!(self.supports_clmul);
        let block = Block::from_unaligned_address(address);
        let (mut live, mut carry) = {
            let state = Transducer::decode(
                OFFSET_VECTOR_SPEC.load_atomic::<usize>(block.start(), Ordering::Relaxed),
                block.start(),
            );
            (state.live, if state.in_object { -1i64 } else { 0i64 })
        };
        MARK_SPEC.scan_words(block.start(), address, &mut |word, _, bits| match bits {
            Bits::Range { start, end } => {
                assert_eq!(start, 0);
                let mask = (1 << end) - 1;
                live += clmul_step(&mut carry, word & mask) as usize;
            }
            Bits::All => live += clmul_step(&mut carry, word) as usize,
        });
        self.first_address + live
    }

    #[cfg(all(debug_assertions, feature = "vo_bit"))]
    pub fn scan_marked_objects(
        &self,
        start: Address,
        end: Address,
        f: &mut impl FnMut(ObjectReference),
    ) {
        let mut in_object = false;
        MARK_SPEC.scan_non_zero_values::<u8>(start, end, &mut |addr: Address| {
            in_object = !in_object;
            if in_object {
                f(ObjectReference::from_raw_address(addr).unwrap())
            }
        });
    }

    pub fn has_calculated_forwarding_addresses(&self) -> bool {
        self.calculated.load(Ordering::Relaxed)
    }
}
