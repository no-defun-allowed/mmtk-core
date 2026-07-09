use crate::policy::compressor::{forwarding, CompressorSpace};
use crate::scheduler::GCWorker;
// use crate::util::constants::BYTES_IN_PAGE;
use crate::util::constants::BYTES_IN_WORD;
use crate::util::{Address, ObjectReference, VMThread, VMWorkerThread};
use crate::vm::slot::Slot;
use crate::{vm::*, ObjectQueue};

use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;

pub struct StubTable<VM: VMBinding> {
    pub stubs: HashMap<Address, Stub>,
    phantom: PhantomData<VM>,
}

pub struct Stub {
    pub references: Vec<ObjectReference>,
    pub mark_word: usize,
}

impl Stub {
    pub fn new() -> Self {
        Stub {
            references: Vec::new(),
            mark_word: 0,
        }
    }

    pub fn add_reference(&mut self, reference: ObjectReference) {
        self.references.push(reference);
    }

    // pub fn mark(&mut self) {
    //     self.mark_word |= 0x1;
    // }

    // pub fn is_marked(&self) -> bool {
    //     (self.mark_word & 0x1) != 0
    // }

    // pub fn clear_mark(&mut self) {
    //     self.mark_word &= !0x1;
    // }

    pub fn set_size(&mut self, size: usize) {
        self.mark_word |= size;
    }

    pub fn get_size(&self) -> usize {
        self.mark_word & !0x1
    }
}

impl<VM: VMBinding> StubTable<VM> {
    pub fn new() -> Self {
        StubTable {
            stubs: HashMap::new(),
            phantom: PhantomData,
        }
    }

    pub fn clear(&mut self) {
        self.stubs.clear();
    }

    pub fn add_stub(&mut self, object: ObjectReference) {
        let object_start = object.to_raw_address();
        // let page_start = object_start.align_down(BYTES_IN_PAGE);
        let mut stub = Stub::new();
        let mut closure = |slot: VM::VMSlot| {
            let Some(child_obj) = slot.load() else { return };
            // let child_obj_start = child_obj.to_raw_address();
            // let child_page_start = child_obj_start.align_down(BYTES_IN_PAGE);
            // if child_page_start != page_start {
            stub.add_reference(child_obj);
            // }
        };
        VM::VMScanning::scan_object(
            VMWorkerThread(VMThread::UNINITIALIZED),
            object,
            &mut closure,
        );
        stub.set_size(VM::VMObjectModel::get_current_size(object));
        self.stubs.insert(object_start, stub);
    }

    pub fn has_stub(&self, object: ObjectReference) -> bool {
        let object_start = object.to_raw_address();
        self.stubs.contains_key(&object_start)
    }

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
        let size = stub.get_size();
        if CompressorSpace::<VM>::test_and_mark(object) {
            use crate::plan::PlanTraceObject;

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

    pub fn get_size(&self, object: ObjectReference) -> Option<NonZeroUsize> {
        let object_start = object.to_raw_address();
        self.stubs
            .get(&object_start)
            // SAFETY: The size of an object is always non-zero
            .map(|stub| unsafe { NonZeroUsize::new_unchecked(stub.get_size()) })
    }

    // pub fn get_references(&self, object: ObjectReference) -> Option<&Vec<ObjectReference>> {
    //     let object_start = object.to_object_start::<VM>();
    //     self.stubs.get(&object_start).map(|stub| &stub.references)
    // }
}
