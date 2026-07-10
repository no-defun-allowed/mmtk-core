use crate::policy::compressor::{forwarding, CompressorSpace};
use crate::scheduler::GCWorker;
use crate::util::constants::BYTES_IN_WORD;
use crate::util::{Address, ObjectReference, VMThread, VMWorkerThread};
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
/// Note that we will pin/mark the directly reachable objects from the stubbed
/// object so that we can avoid updating references in the stubbed object.
pub struct StubTable<VM: VMBinding> {
    /// Map from object to its [`Stub`].
    pub stubs: HashMap<Address, Stub>,
    phantom: PhantomData<VM>,
}

/// A stub is a representation of an object that keeps track of its references
/// and its size. Stubs live in the [`StubTable`].
pub struct Stub {
    /// List of references from the object to other objects.
    pub references: Vec<ObjectReference>,
    /// The size of the object.
    pub size: usize,
}

impl Stub {
    pub fn new() -> Self {
        Stub {
            references: Vec::new(),
            size: 0,
        }
    }

    /// Add a reference to the stub.
    pub fn add_reference(&mut self, reference: ObjectReference) {
        self.references.push(reference);
    }

    /// Set the size of the stub.
    pub fn set_size(&mut self, size: usize) {
        self.size = size;
    }

    /// Get the size of the stub.
    pub fn get_size(&self) -> usize {
        self.size
    }
}

impl<VM: VMBinding> StubTable<VM> {
    pub fn new() -> Self {
        StubTable {
            stubs: HashMap::new(),
            phantom: PhantomData,
        }
    }

    /// Clear the stub table.
    pub fn clear(&mut self) {
        self.stubs.clear();
    }

    /// Create a stub for the given object and add it to the stub table.
    pub fn add_stub(&mut self, object: ObjectReference) {
        debug_assert!(!self.has_stub(object));
        let object_start = object.to_raw_address();
        let mut stub = Stub::new();
        let mut closure = |slot: VM::VMSlot| {
            let Some(child_obj) = slot.load() else { return };
            stub.add_reference(child_obj);
        };
        VM::VMScanning::scan_object(
            VMWorkerThread(VMThread::UNINITIALIZED),
            object,
            &mut closure,
        );
        stub.set_size(VM::VMObjectModel::get_current_size(object));
        self.stubs.insert(object_start, stub);
    }

    /// Remove the stub for the given object from the stub table.
    #[allow(unused)]
    pub fn remove_stub(&mut self, object: ObjectReference) {
        debug_assert!(self.has_stub(object));
        let object_start = object.to_raw_address();
        self.stubs.remove(&object_start);
    }

    /// Check if the stub table has a stub for the given object.
    pub fn has_stub(&self, object: ObjectReference) -> bool {
        let object_start = object.to_raw_address();
        self.stubs.contains_key(&object_start)
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
        debug_assert!(self.has_stub(object));
        let object_start = object.to_raw_address();
        let Some(stub) = self.stubs.get(&object_start) else {
            unreachable!()
        };

        // If we are the first the mark the object, then go and mark its
        // children as well
        if CompressorSpace::<VM>::test_and_mark(object) {
            use crate::plan::PlanTraceObject;

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
            for reference in &stub.references {
                while !forwarding::is_object_pinned::<VM>(*reference) {
                    forwarding::pin_object::<VM>(*reference);
                }
                debug_assert!(forwarding::is_object_pinned::<VM>(*reference));
                // XXX(kunals): Be careful. We actually need to mark the referent objects.
                compressor.trace_object::<Q, { super::TRACE_KIND_MARK }>(queue, *reference, worker);
                debug!(
                    "Marking reference {:?} from stub object {:?}",
                    reference, object
                );
            }
        }
    }

    /// Get the size of the given object from the stub table. Returns `None` if
    /// the object is not in the stub table.
    pub fn get_size(&self, object: ObjectReference) -> Option<NonZeroUsize> {
        let object_start = object.to_raw_address();
        self.stubs
            .get(&object_start)
            // SAFETY: The size of an object is always non-zero
            .map(|stub| unsafe { NonZeroUsize::new_unchecked(stub.get_size()) })
    }
}
