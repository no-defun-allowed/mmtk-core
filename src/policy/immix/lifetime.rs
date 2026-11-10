use crate::util::ObjectReference;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::statistics::counter::EventCounter;
use crate::util::statistics::stats::Stats;
use crate::vm::VMBinding;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;

pub const LIFETIME_TABLE: SideMetadataSpec =
    crate::util::metadata::side_metadata::spec_defs::IX_OBJECT_LIFETIME;
type Metadata = u8;
const LIMIT: Metadata = 15 as Metadata;

pub fn zero_lifetime<VM: VMBinding>(object: ObjectReference) {
    LIFETIME_TABLE.store_atomic::<Metadata>(object.to_object_start::<VM>(), 0, Ordering::Relaxed);
}

pub fn on_object_marked<VM: VMBinding>(object: ObjectReference) {
    // Bump the age.
    let age = LIFETIME_TABLE.load_atomic::<Metadata>(object.to_object_start::<VM>(), Ordering::Relaxed);
    let age = (age + 1).min(LIMIT);
    LIFETIME_TABLE.store_atomic::<Metadata>(object.to_object_start::<VM>(), age, Ordering::Relaxed);
}

pub fn on_object_forwarded<VM: VMBinding>(old_object: ObjectReference, new_object: ObjectReference) {
    // Copy and bump the age.
    let age = LIFETIME_TABLE.load_atomic::<Metadata>(old_object.to_object_start::<VM>(), Ordering::Relaxed);
    let age = (age + 1).min(LIMIT);
    LIFETIME_TABLE.store_atomic::<Metadata>(new_object.to_object_start::<VM>(), age, Ordering::Relaxed);
}

pub fn object_age<VM: VMBinding>(object: ObjectReference) -> usize {
    LIFETIME_TABLE.load_atomic::<Metadata>(object.to_object_start::<VM>(), Ordering::Relaxed) as usize
}

pub struct Counters {
    v: Vec<Arc<Mutex<EventCounter>>>
}

impl Counters {
    pub fn new(stats: &Stats) -> Self {
        Self {
            v: (0..=LIMIT)
                .map(|n| stats.new_event_counter(&format!("survived.{n}"), true, true, true))
                .collect(),
        }
    }
    pub fn count(&self, n: usize) {
        self.v[n].clone().lock().unwrap().inc();
    }
}
