use crate::policy::compressor::forwarding;
use crate::util::linear_scan::Region;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::{Address, ObjectReference};
use atomic::Ordering;

pub(crate) const STATUS_SPEC: SideMetadataSpec =
    crate::util::metadata::side_metadata::spec_defs::ONE_PASS_STATUS;

/// The status of a One Pass block. If the status of a block is `Forwarded`,
/// a reference into the block should be forwarded by using the offset vector
/// and mark bitmap (like the normal Compressor). Else the reference should be
/// put on the threading list of the referent object (after incrementing the
/// thread count, and before decrementing the count).
///
/// The status contains the "threading count" and status bit of a block as presented
/// in the One Pass Compactor paper
/// <https://dl.acm.org/doi/pdf/10.1145/3652024.3665513#page=9>.
/// The `INITIAL` state in the paper is `Status::Threading(0)`
/// and the `FINAL` state in the paper is `Status::Forwarded`.
#[derive(Debug)]
pub(crate) enum Status {
    Forwarded,
    Threading(u8),
}

impl Status {
    pub const MAX_WORKERS: usize = 254;
    const FORWARDED: u8 = 255;
    const fn encode(&self) -> u8 {
        match self {
            Status::Forwarded => Status::FORWARDED,
            Status::Threading(n) => {
                debug_assert_ne!(n, FORWARDED);
                *n
            }
        }
    }
    const fn decode(n: u8) -> Self {
        match n {
            Status::FORWARDED => Status::Forwarded,
            _ => Status::Threading(n),
        }
    }
}

pub(crate) fn reset_metadata(start: Address, size: usize) {
    const_assert_eq!(Status::Threading(0).encode(), 0);
    STATUS_SPEC.bzero_metadata(start, size);
}

pub(crate) fn status(block: forwarding::Block) -> Status {
    Status::decode(STATUS_SPEC.load_atomic::<u8>(block.start(), Ordering::Relaxed))
}

pub(crate) fn cas_status(block: forwarding::Block, from: Status, to: Status) -> bool {
    let from = from.encode();
    let to = to.encode();
    STATUS_SPEC
        .compare_exchange_atomic::<u8>(block.start(), from, to, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

pub(crate) enum ThreadOrForward {
    Thread,
    Forward,
}

pub(crate) fn thread_or_forward(target: ObjectReference, body: &mut impl FnMut(ThreadOrForward)) {
    let target_block = forwarding::Block::from_unaligned_address(target.to_raw_address());
    loop {
        match status(target_block) {
            Status::Forwarded => {
                body(ThreadOrForward::Forward);
                return;
            }
            Status::Threading(n) => {
                // Try to increment the threading worker count.
                if cas_status(target_block, Status::Threading(n), Status::Threading(n + 1)) {
                    body(ThreadOrForward::Thread);
                    // Now decrement the threading worker count.
                    loop {
                        let Status::Threading(n) = status(target_block) else {
                            panic!("should not see Forwarded before we've finished threading");
                        };
                        assert_ne!(
                            n, 0,
                            "should not see Threading(0) before we've finished threading"
                        );
                        if cas_status(target_block, Status::Threading(n), Status::Threading(n - 1))
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn lock_for_forwarding(block: forwarding::Block, body: &mut (impl FnMut() + ?Sized)) {
    loop {
        let s = status(block);
        match s {
            Status::Forwarded => {
                panic!("already forwarded {block:?}, in status {s:?}")
            }
            Status::Threading(0) => {
                if cas_status(block, Status::Threading(0), Status::Forwarded) {
                    body();
                    return;
                }
            }
            Status::Threading(_) => { /* spin until no more threading */ }
        }
    }
}
