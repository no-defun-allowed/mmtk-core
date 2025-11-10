use crate::policy::compressor::forwarding;
use crate::util::linear_scan::Region;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::{Address, ObjectReference};
use atomic::Ordering;

pub(crate) const STATUS_SPEC: SideMetadataSpec =
    crate::util::metadata::side_metadata::spec_defs::ONE_PASS_STATUS;

#[derive(Debug)]
pub(crate) enum Status {
    Forwarded,
    Forwarding,
    Threading(u8),
}

impl Status {
    pub const MAX_WORKERS: usize = 253;
    const FORWARDED: u8 = 254;
    const FORWARDING: u8 = 255;
    fn encode(&self) -> u8 {
        match self {
            Status::Forwarded => Status::FORWARDED,
            Status::Forwarding => Status::FORWARDING,
            Status::Threading(n) => *n,
        }
    }
    fn decode(n: u8) -> Self {
        match n {
            Status::FORWARDED => Status::Forwarded,
            Status::FORWARDING => Status::Forwarding,
            _ => Status::Threading(n),
        }
    }
}

pub(crate) fn reset_metadata(start: Address, size: usize) {
    // Status::Threading(0).encode() == 0
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

pub(crate) fn thread_or_forward(
    source: ObjectReference,
    target: ObjectReference,
    body: &mut impl FnMut(ThreadOrForward),
) {
    if forwarding::block_number(source.to_raw_address())
        == forwarding::block_number(target.to_raw_address())
    {
        body(ThreadOrForward::Forward);
        return;
    }
    let target_block = forwarding::Block::from_unaligned_address(target.to_raw_address());
    loop {
        match status(target_block) {
            Status::Forwarded => {
                body(ThreadOrForward::Forward);
                return;
            }
            Status::Forwarding => { /* spin until forwarded */ }
            Status::Threading(n) => {
                // increment threading worker count
                if cas_status(target_block, Status::Threading(n), Status::Threading(n + 1)) {
                    body(ThreadOrForward::Thread);
                    // decrement threading worker count
                    loop {
                        let Status::Threading(n) = status(target_block) else {
                            panic!("Threading(n > 0) => Forwarded|Forwarding transition");
                        };
                        assert!(n > 0, "we're still threading, but saw Threading(0)");
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
            Status::Forwarded | Status::Forwarding => {
                panic!("already forwarded {block:?}, in status {s:?}")
            }
            Status::Threading(0) => {
                if cas_status(block, Status::Threading(0), Status::Forwarding) {
                    body();
                    assert!(cas_status(block, Status::Forwarding, Status::Forwarded));
                    return;
                }
            }
            Status::Threading(_) => { /* spin until no more threading */ }
        }
    }
}
