use crate::util::{ObjectReference};
use crate::scheduler::GCWork;
use crate::vm::VMBinding;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

const CAPACITY: usize = 1024;

#[repr(C)]
pub(super) struct RemSetEntry<VM: VMBinding>(VM::VMSlot, ObjectReference);

impl<VM: VMBinding> RemSetEntry<VM> {
    fn encode(slot: VM::VMSlot, o: ObjectReference) -> Self {
        Self(slot, o)
    }
    pub fn decode(&self) -> (VM::VMSlot, ObjectReference) {
        (self.0, self.1)
    }
}

type PacketGenerator<VM> = Box<dyn Fn(Vec<RemSetEntry<VM>>) -> Box<dyn GCWork<VM>>>;

pub struct RemSet<VM: VMBinding> {
    pub(super) gc_buffers: Vec<UnsafeCell<Vec<RemSetEntry<VM>>>>,
    local_packets: Vec<UnsafeCell<Vec<Box<dyn GCWork<VM>>>>>,
    _p: PhantomData<VM>,
    size: AtomicUsize,
    packet_generator: PacketGenerator<VM>,
}

impl<VM: VMBinding> RemSet<VM> {
    pub fn new(workers: usize, packet_generator: PacketGenerator<VM>) -> Self {
        let mut rs = Self {
            gc_buffers: vec![],
            local_packets: vec![],
            _p: PhantomData,
            size: AtomicUsize::new(0),
            packet_generator,
        };
        rs.gc_buffers
            .resize_with(workers, || UnsafeCell::new(vec![]));
        rs.local_packets
            .resize_with(workers, || UnsafeCell::new(vec![]));
        rs
    }

    fn gc_buffer(&self, id: usize) -> &mut Vec<RemSetEntry<VM>> {
        unsafe { &mut *self.gc_buffers[id].get() }
    }

    fn flush_all(&self, packet_consumer: &mut impl FnMut(Box<dyn GCWork<VM>>)) {
        self.size.store(0, Ordering::SeqCst);
        for id in 0..self.gc_buffers.len() {
            if self.gc_buffer(id).len() > 0 {
                let remset = std::mem::take(self.gc_buffer(id));
                packet_consumer((self.packet_generator)(remset));
            }
        }
        for id in 0..self.local_packets.len() {
            let buf = unsafe { &mut *self.local_packets[id].get() };
            if buf.len() > 0 {
                let packets = std::mem::take(buf);
                for p in packets {
                    packet_consumer(p);
                }
            }
        }
    }

    #[cold]
    fn flush(&self, id: usize) {
        if self.gc_buffer(id).len() > 0 {
            let remset = std::mem::take(self.gc_buffer(id));
            self.size.fetch_add(remset.len(), Ordering::SeqCst);
            let w = (self.packet_generator)(remset);
            let packet_buffer = unsafe { &mut *self.local_packets[id].get() };
            packet_buffer.push(w);
        }
    }

    pub fn record(&self, s: VM::VMSlot, o: ObjectReference) {
        let id = crate::scheduler::current_worker_ordinal();
        self.gc_buffer(id).push(RemSetEntry::encode(s, o));
        if self.gc_buffer(id).len() >= CAPACITY {
            self.flush(id)
        }
    }
}
