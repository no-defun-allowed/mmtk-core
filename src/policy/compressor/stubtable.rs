use crate::policy::compressor::{forwarding, CompressorSpace};
use crate::scheduler::GCWorker;
use crate::util::constants::BYTES_IN_WORD;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::{ObjectReference, VMThread, VMWorkerThread};
use crate::vm::slot::Slot;
use crate::{vm::*, ObjectQueue};

use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;

/// A stub table keeps track of objects that live on swapped out pages. When a
/// page is swapped out, we scan all objects that potentially intersect the page
/// and add them and their [`Stub`]s to the stub table.
///
/// During a GC, we can use the stub table to mark all objects that are
/// reachable from a stubbed object. This allows us to perform a precise GC
/// without having to touch pages where a swapped out object lives at the cost
/// of increased memory usage in storing the stubs. If we are conservative and
/// mark all reachable objects from a page, then we may be keeping completely
/// unreachable objects alive.
///
/// References for every stub are packed into a single shared `stubs` arena
/// rather than each [`Stub`] owning its own allocation: each stub's
/// references occupy a contiguous `[offset, offset + len)` range of `stubs`.
/// This avoids paying a separate heap allocation per stubbed object, most of
/// which only have a handful of references.
///
/// Removing a stub (see [`Self::remove_stub`]) does not shift the arena: it
/// just drops the map entry and leaves the vacated range as garbage. The
/// arena is compacted lazily, in-place, the next time [`Self::prune_stubs`]
/// runs (once per GC), so garbage can persist for at most one GC cycle.
pub struct StubTable<VM: VMBinding> {
    /// Map from object to its [`Stub`]. Only objects with references are added
    /// to this map. Leaf objects (i.e. objects with no references) are added to
    /// [`Self::leaf_map`].
    pub stub_map: HashMap<ObjectReference, Stub<VM>>,
    /// Map from a leaf object to its size.
    pub leaf_map: HashMap<ObjectReference, u16>,
    /// Shared backing storage for every stub's references. See the
    /// struct-level docs for how stubs index into this.
    pub stubs: Vec<(u16, ObjectReference)>,
    pub num_stubs: usize,
    pub num_leaf_stubs: usize,
    pub num_references: usize,
    pub total_object_size: usize,
}

/// A stub is a representation of an object that keeps track of its references
/// and its size. Stubs live in the [`StubTable`].
pub struct Stub<VM: VMBinding> {
    /// The offset of this stub's references into [`StubTable::stubs`].
    pub offset: u32,
    /// The number of references this stub has, starting at `offset`.
    pub len: u16,
    /// The size of the object.
    pub size: u16,
    phantom: PhantomData<VM>,
}

impl<VM: VMBinding> Stub<VM> {
    pub fn new() -> Self {
        Stub {
            offset: 0,
            len: 0,
            size: 0,
            phantom: PhantomData,
        }
    }

    /// Set the size of the stub.
    pub fn set_size(&mut self, size: usize) {
        self.size = size as u16;
    }

    /// Get the size of the stub.
    pub fn get_size(&self) -> usize {
        self.size as usize
    }
}

impl<VM: VMBinding> StubTable<VM> {
    pub fn new() -> Self {
        StubTable {
            stub_map: HashMap::new(),
            leaf_map: HashMap::new(),
            stubs: Vec::new(),
            num_stubs: 0,
            num_leaf_stubs: 0,
            num_references: 0,
            total_object_size: 0,
        }
    }

    /// Get the references belonging to `stub`.
    fn references_of(&self, stub: &Stub<VM>) -> &[(u16, ObjectReference)] {
        let start = stub.offset as usize;
        let end = start + stub.len as usize;
        &self.stubs[start..end]
    }

    /// Count how many references in the stub table point at an object that is
    /// also pointed at by at least one other reference in the stub table
    /// (across the same stub or different stubs). Returns
    /// `(num_duplicate_references, num_unique_referenced_objects, sum_top_5pct_duplicates)`.
    fn count_duplicate_references(&self) -> (usize, usize, usize) {
        let mut counts: HashMap<ObjectReference, usize> = HashMap::new();
        for stub in self.stub_map.values() {
            for (_, reference) in self.references_of(stub) {
                *counts.entry(*reference).or_insert(0) += 1;
            }
        }
        let num_unique = counts.len();
        // Every occurrence past the first one for a given object is a duplicate.
        let num_duplicates = counts.values().map(|&count| count - 1).sum();
        let five_pct = (num_unique as f64 * 0.05).ceil() as usize;
        let mut top_5pct = counts.values().copied().collect::<Vec<_>>();
        top_5pct.sort_unstable_by(|a, b| b.cmp(a));
        // The sum of the top 5% of duplicate counts, minus the 1st occurrence for each reference.
        let top_5pct = top_5pct.into_iter().take(five_pct).sum::<usize>() - five_pct;
        (num_duplicates, num_unique, top_5pct)
    }

    /// Print metrics about the stub table to a file.
    pub fn print_table_metrics(&self, filename: &str, num_pinned_pages: u32) {
        use std::io::Write;

        let mut metadata_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .unwrap();

        let epoch = super::NUM_GCS.load(Ordering::SeqCst);
        writeln!(metadata_file, "GC epoch {}:", epoch).unwrap();

        let stub_size = self.num_stubs * std::mem::size_of::<Stub<VM>>();
        let leaf_stub_size = self.num_leaf_stubs * std::mem::size_of::<u16>();
        // Use the arena's actual capacity (rather than `num_references *
        // size_of`) since the arena can hold garbage left behind by
        // `remove_stub` until the next `prune_stubs` compacts it away.
        let arena_size = self.stubs.capacity() * std::mem::size_of::<(u16, ObjectReference)>();
        let moving_metadata_size = stub_size + leaf_stub_size + arena_size;
        let nonmoving_metadata_size = stub_size
            + leaf_stub_size
            + self.num_references * std::mem::size_of::<ObjectReference>();
        let (num_duplicate_references, num_unique_referenced_objects, sum_top_5pct_duplicates) =
            self.count_duplicate_references();
        writeln!(
            metadata_file,
            "  total object size: {} bytes;\n  num pinned pages: {} ({} bytes);\n  num non-leaf stubs: {} ({} metadata bytes);\n  num references: {} (average: {:.2} references per non-leaf stub; {:.2} references per page);\n  num duplicate references: {} (sum top 5%: {}; top 5% fraction: {:.2}); num unique referenced objects: {};\n  num leaf stubs: {} ({:.2} fraction; {} metadata bytes);\n  metadata size (moving): {} (average: {:.2} bytes per page);\n  metadata size (non-moving): {} (average: {:.2} bytes per page)",
            self.total_object_size,
            num_pinned_pages,
            num_pinned_pages as usize * crate::util::constants::BYTES_IN_PAGE,
            self.num_stubs,
            stub_size,
            self.num_references,
            self.num_references as f64 / self.num_stubs as f64,
            self.num_references as f64 / num_pinned_pages as f64,
            num_duplicate_references,
            sum_top_5pct_duplicates,
            sum_top_5pct_duplicates as f64 / num_duplicate_references as f64,
            num_unique_referenced_objects,
            self.num_leaf_stubs,
            self.num_leaf_stubs as f64 / (self.num_leaf_stubs + self.num_stubs) as f64,
            leaf_stub_size,
            moving_metadata_size,
            moving_metadata_size as f64 / num_pinned_pages as f64,
            nonmoving_metadata_size,
            nonmoving_metadata_size as f64 / num_pinned_pages as f64,
        ).unwrap();
    }

    /// Clear the stub table.
    pub fn clear(&mut self) {
        self.stub_map.clear();
        self.leaf_map.clear();
        self.stubs.clear();
        self.num_stubs = 0;
        self.num_leaf_stubs = 0;
        self.num_references = 0;
        self.total_object_size = 0;
    }

    /// Create a stub for the given object and add it to the stub table.
    pub fn add_stub(&mut self, object: ObjectReference) {
        debug_assert!(!self.has_stub(object));
        let object_start = object.to_raw_address();

        // Scan into a temporary buffer first: `scan_object`'s closure can't
        // borrow `self.stubs` directly while we're already borrowing `self`
        // mutably to call it.
        let mut references: Vec<(u16, ObjectReference)> = Vec::new();
        let mut closure = |slot: VM::VMSlot| {
            let Some(child_obj) = slot.load() else { return };
            let offset = slot.as_address() - object_start;
            references.push((offset as u16, child_obj));
        };
        VM::VMScanning::scan_object(
            VMWorkerThread(VMThread::UNINITIALIZED),
            object,
            &mut closure,
        );

        let len = references.len();
        assert!(
            len <= u16::MAX as usize,
            "Stub for object {:?} has too many references ({}) to fit in a u16",
            object,
            len
        );
        if len > 0 {
            let offset = self.stubs.len() as u32;
            self.stubs.extend(references);

            let mut stub = Stub::new();
            stub.offset = offset;
            stub.len = len as u16;
            stub.set_size(VM::VMObjectModel::get_current_size(object));
            debug!(
                "Adding stub for object {:?} (size {})",
                object,
                stub.get_size(),
            );

            self.num_stubs += 1;
            self.num_references += len;
            self.total_object_size += stub.get_size();
            self.stub_map.insert(object, stub);
        } else {
            debug_assert_eq!(len, 0);
            let size = VM::VMObjectModel::get_current_size(object);
            debug!("Adding leaf stub for object {:?} (size {})", object, size);
            self.num_leaf_stubs += 1;
            self.total_object_size += size;
            self.leaf_map.insert(object, size as u16);
        }
    }

    fn is_leaf_stub(&self, object: ObjectReference) -> bool {
        self.leaf_map.contains_key(&object)
    }

    /// Remove the stub for the given object from the stub table.
    ///
    /// This does not reclaim the object's references from the shared arena:
    /// they become garbage that is only reclaimed the next time
    /// [`Self::prune_stubs`] compacts the arena.
    pub fn remove_stub(&mut self, object: ObjectReference) {
        debug_assert!(self.has_stub(object));
        let is_leaf = self.is_leaf_stub(object);
        if is_leaf {
            let size = self.leaf_map.remove(&object).unwrap();
            self.num_leaf_stubs -= 1;
            self.total_object_size -= size as usize;
        } else {
            let stub = self.stub_map.remove(&object).unwrap();
            self.num_stubs -= 1;
            self.num_references -= stub.len as usize;
            self.total_object_size -= stub.get_size();
        }
    }

    /// Remove all unmarked stubs from the stub table. This is called after the
    /// transitive closure phase of a GC to remove stubs for objects that are no
    /// longer reachable.
    ///
    /// Note that we call this because a moveable object may get moved to an
    /// address that happens to be in the stub table. If we don't clean the stub
    /// table, we may incorrectly think a moveable object is actually stubbed.
    ///
    /// This also compacts the shared reference arena in place: any garbage
    /// left behind by `remove_stub` calls since the last GC is squeezed out
    /// in the same pass, so the arena never grows unboundedly relative to
    /// the number of live references.
    pub fn prune_stubs(&mut self) {
        let mut to_remove = Vec::new();
        // (offset, address) for every surviving stub, so we can walk the
        // arena left-to-right and compact it in a single pass.
        let mut live = Vec::with_capacity(self.stub_map.len());
        for (&object, stub) in &self.stub_map {
            if forwarding::MARK_SPEC
                .load_atomic::<u8>(object.to_object_start::<VM>(), Ordering::SeqCst)
                == 0
            {
                to_remove.push(object);
            } else {
                live.push((stub.offset, object));
            }
        }

        for (&object, _) in &self.leaf_map {
            if forwarding::MARK_SPEC
                .load_atomic::<u8>(object.to_object_start::<VM>(), Ordering::SeqCst)
                == 0
            {
                to_remove.push(object);
            }
        }

        for object in to_remove {
            self.remove_stub(object);
        }

        live.sort_unstable_by_key(|&(offset, _)| offset);

        let mut write = 0_usize;
        for (offset, address) in live {
            let stub = self.stub_map.get_mut(&address).unwrap();
            let len = stub.len as usize;
            let read = offset as usize;
            if read != write {
                self.stubs.copy_within(read..read + len, write);
            }
            stub.offset = write as u32;
            write += len;
        }
        self.stubs.truncate(write);
    }

    /// Check if the stub table has a stub for the given object.
    pub fn has_stub(&self, object: ObjectReference) -> bool {
        self.stub_map.contains_key(&object) || self.leaf_map.contains_key(&object)
    }

    /// Mark the given stub object and its references. This is called during the
    /// transitive closure phase of a GC.
    pub fn mark_object_stub<Q: ObjectQueue>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        forwarding: &forwarding::ForwardingMetadata<VM>,
        worker: &mut GCWorker<VM>,
    ) {
        use crate::plan::PlanTraceObject;

        debug_assert!(self.has_stub(object));
        if self.is_leaf_stub(object) {
            if CompressorSpace::<VM>::test_and_mark(object) {
                let size = self.leaf_map.get(&object).copied().unwrap() as usize;
                forwarding.mark_rest_of_object_known_size(object, size);
                while !forwarding::is_object_pinned::<VM>(object) {
                    forwarding::pin_object::<VM>(object);
                }

                debug_assert!(
                    forwarding::is_object_pinned::<VM>(object),
                    "Object {:?} in stub table is not pinned!",
                    object
                );
                debug_assert!(
                    forwarding::MARK_SPEC
                        .load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst)
                        != 0,
                    "Object {:?} in stub table is not marked!",
                    object
                );
                debug_assert!(
                    forwarding::MARK_SPEC.load_atomic::<u8>(
                        object.to_raw_address() + size - BYTES_IN_WORD,
                        Ordering::SeqCst
                    ) != 0,
                    "Object end {:?} in stub table is not marked!",
                    object
                );
            }
            return;
        }

        // If we are the first the mark the object, then go and mark its
        // children as well
        let stub = self.stub_map.get(&object).unwrap();
        if CompressorSpace::<VM>::test_and_mark(object) {
            let size = stub.get_size();
            forwarding.mark_rest_of_object_known_size(object, size);
            while !forwarding::is_object_pinned::<VM>(object) {
                forwarding::pin_object::<VM>(object);
            }

            debug_assert!(
                forwarding::is_object_pinned::<VM>(object),
                "Object {:?} in stub table is not pinned!",
                object
            );
            debug_assert!(
                forwarding::MARK_SPEC.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst)
                    != 0,
                "Object {:?} in stub table is not marked!",
                object
            );
            debug_assert!(
                forwarding::MARK_SPEC.load_atomic::<u8>(
                    object.to_raw_address() + size - BYTES_IN_WORD,
                    Ordering::SeqCst
                ) != 0,
                "Object end {:?} in stub table is not marked!",
                object
            );

            let compressor = worker
                .mmtk
                .get_plan()
                .downcast_ref::<crate::plan::compressor::Compressor<VM>>()
                .unwrap();
            for (_, reference) in self.references_of(stub) {
                debug!(
                    "Marking reference {:?} from stub object {:?}",
                    reference, object
                );
                // XXX(kunals): Be careful. We actually need to mark the referent objects.
                compressor.trace_object::<Q, { super::TRACE_KIND_MARK }>(queue, *reference, worker);
                debug_assert!(
                    reference.is_reachable(),
                    "Reference {:?} in stub object {:?} is not marked!",
                    reference,
                    object
                );
            }
        }
    }

    /// Update the references in the given stub object. This is called during the
    /// forwarding phase of a GC to update the references in the stub object to
    /// point to the correct objects.
    ///
    /// Note that this does not actually update the slots inside the object
    /// itself. It only updates the references in the stub table. The actual
    /// slots in the object will be updated when the page is swapped back in.
    pub fn update_object_stub(
        &mut self,
        object: ObjectReference,
        update_closure: &mut dyn FnMut(ObjectReference) -> ObjectReference,
    ) {
        debug_assert!(self.has_stub(object));

        if self.is_leaf_stub(object) {
            // Nothing to do here; leaf objects have no references to update.
            return;
        }

        let object_start = object.to_raw_address();
        let stub = self.stub_map.get(&object).unwrap();
        let start = stub.offset as usize;
        let end = start + stub.len as usize;

        debug_assert!(
            forwarding::MARK_SPEC.load_atomic::<u8>(object_start, Ordering::SeqCst) != 0,
            "Trying to update unmarked object {:?} in stub table!",
            object
        );

        for (_, reference) in &mut self.stubs[start..end] {
            debug_assert!(
                reference.is_reachable(),
                "Reference {:?} in stub object {:?} is not marked!",
                reference,
                object,
            );
            let new_reference = update_closure(*reference);
            if new_reference != *reference {
                debug!(
                    "Updating reference {:?} -> {:?} in stub object {:?}",
                    reference, new_reference, object
                );
                *reference = new_reference;
            }
        }
    }

    /// Regenerate the references in all stubbed objects. This is called when a
    /// page is swapped back in to update the slots in the object to point to
    /// the correct references.
    pub fn regenerate_objects(&self) {
        for &object in self.stub_map.keys() {
            if forwarding::MARK_SPEC
                .load_atomic::<u8>(object.to_object_start::<VM>(), Ordering::SeqCst)
                != 0
            {
                self.regenerate_object(object);
            }
        }
    }

    /// Regenerate the references in a given stubbed object. This is called when
    /// a page is swapped back in to update the slots in the object to point to
    /// the correct references.
    pub fn regenerate_object(&self, object: ObjectReference) {
        debug_assert!(self.has_stub(object));
        #[cfg(feature = "vo_bit")]
        debug_assert!(vo_bit::is_vo_bit_set(object));

        if self.is_leaf_stub(object) {
            // Nothing to do here; leaf objects have no references to regenerate.
            return;
        }

        let stub = self.stub_map.get(&object).unwrap();
        for (offset, reference) in self.references_of(stub) {
            debug!(
                "Regenerating reference {:?} in stub object {:?}",
                reference, object
            );
            #[cfg(feature = "vo_bit")]
            debug_assert!(
                vo_bit::is_vo_bit_set(*reference),
                "Reference {:?} in stub object {:?} does not have VO bit set!",
                reference,
                object
            );
            // SAFETY: The offset is guaranteed to be within the object size, as it was computed during the stub creation phase.
            let slot =
                unsafe { VM::VMObjectModel::slot_from_object_and_offset(object, *offset as isize) };
            slot.store(*reference);
        }
    }

    /// Get the size of the given object from the stub table. Returns `None` if
    /// the object is not in the stub table.
    pub fn get_size(&self, object: ObjectReference) -> Option<NonZeroUsize> {
        if self.is_leaf_stub(object) {
            let size = self.leaf_map.get(&object).copied()?;
            // SAFETY: The size of a leaf object is always non-zero
            return Some(unsafe { NonZeroUsize::new_unchecked(size as usize) });
        } else {
            self.stub_map
                .get(&object)
                // SAFETY: The size of an object is always non-zero
                .map(|stub| unsafe { NonZeroUsize::new_unchecked(stub.get_size()) })
        }
    }
}
