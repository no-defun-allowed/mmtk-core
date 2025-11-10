use crate::policy::compressor::forwarding;
use crate::util::{Address, ObjectReference};

enum Status {
    Forwarded,
    Forwarding,
    Threading(u8),
}

impl Status {
    pub const MAX_THREADS: usize = 253;
    const FORWARDED: u8 = 254;
    const FORWARDING: u8 = 255;
    fn encode(&self) -> u8 {
        match self {
            Forwarded => FORWARDED,
            Forwarding => FORWARDING,
            Threading(n) => n,
        }
    }
    fn decode(n: u8) -> Self {
        match n {
            FORWARDED => Forwarded,
            FORWARDING => Forwarding,
            _ => Threading(n)
        }
    }
}

fn status(block: forwarding::Block) {
    todo!();
}

fn cas_status(block: forwarding::Block, from: Status, to: Status) {
    todo!();
}

enum ThreadOrForward {
    Thread,
    Forward,
}

fn thread_or_forward(
    source: ObjectReference,
    target: ObjectReference,
    body: &mut impl FnMut(ThreadOrForward),
) {
    if forwarding::block_number(source) == forwarding::block_number(target) {
        body(Forward);
        return;
    }
    let target_block = Block::from_unaligned_address(target.to_raw_address());
    loop {
        match status(target_block) {
            Forwarded => {
                body(Forward);
                return;
            }
            Forwarding => { /* spin until forwarded */ }
            Threading(n) => {
                // increment threading thread count
                if cas_status(target_block, Threading(n), Threading(n + 1)) {
                    body(Thread);
                    // decrement threading thread count
                    loop {
                        let Threading(n) = status(target_block) else {
                            panic!("Threading(n > 0) => Forwarded|Forwarding transition?");
                        };
                        assert!(n > 0, "we're still threading, but saw Threading(0)");
                        if cas_status(target_block, Threading(n), Threading(n - 1)) {
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn forward(block: forwarding::Block, body: &mut impl FnMut()) {
    loop {
        let s = status(block);
        match s {
            Forwarded | Forwarding => {
                panic!("already forwarded {block}, in status {s}")
            }
            Threading(0) => {
                if cas_status(target, Threading(0), Forwarding) {
                    body();
                    assert!(cas_status(target, Forwarding, Forwarding));
                    return;
                }
            }
            Threading(_) => { /* spin until no more threading */ }
        }
    }
}
