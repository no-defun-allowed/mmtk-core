use crate::plan::barriers::SATBBarrier;
use crate::plan::concurrent::barrier::SATBBarrierSemantics;
use crate::plan::concurrent::compressor::ConcurrentCompressor;
use crate::plan::concurrent::Pause;
use crate::plan::mutator_context::create_allocator_mapping;
use crate::plan::mutator_context::create_space_mapping;

use crate::plan::mutator_context::Mutator;
use crate::plan::mutator_context::MutatorBuilder;
use crate::plan::mutator_context::MutatorConfig;
use crate::plan::mutator_context::ReservedAllocators;
use crate::plan::AllocationSemantics;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::alloc::BumpAllocator;
use crate::util::opaque_pointer::{VMMutatorThread, VMWorkerThread};
use crate::vm::VMBinding;
use crate::MMTK;
use enum_map::{enum_map, EnumMap};

type BarrierSemanticsType<VM> = SATBBarrierSemantics<
    VM,
    ConcurrentCompressor<VM>,
    { crate::policy::compressor::TRACE_KIND_MARK },
>;

type BarrierType<VM> = SATBBarrier<BarrierSemanticsType<VM>>;

pub fn concurrent_compressor_mutator_release<VM: VMBinding>(
    mutator: &mut Mutator<VM>,
    _tls: VMWorkerThread,
) {
    // Release is not scheduled for initial mark pause
    let current_pause = mutator.plan.concurrent().unwrap().current_pause().unwrap();
    debug_assert_ne!(current_pause, Pause::InitialMark);

    let bump_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
    }
    .downcast_mut::<BumpAllocator<VM>>()
    .unwrap();
    bump_allocator.reset();

    // Deactivate SATB
    if current_pause == Pause::Full || current_pause == Pause::FinalMark {
        debug!("Deactivate SATB barrier active for {:?}", mutator as *mut _);
        mutator
            .barrier
            .downcast_mut::<BarrierType<VM>>()
            .unwrap()
            .set_weak_ref_barrier_enabled(false);
    }
}

pub fn concurent_compressor_mutator_prepare<VM: VMBinding>(
    mutator: &mut Mutator<VM>,
    _tls: VMWorkerThread,
) {
    // Prepare is not scheduled for final mark pause
    let current_pause = mutator.plan.concurrent().unwrap().current_pause().unwrap();
    debug_assert_ne!(current_pause, Pause::FinalMark);

    let bump_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
    }
    .downcast_mut::<BumpAllocator<VM>>()
    .unwrap();
    bump_allocator.reset();

    // Activate SATB
    if current_pause == Pause::InitialMark {
        debug!("Activate SATB barrier active for {:?}", mutator as *mut _);
        mutator
            .barrier
            .downcast_mut::<BarrierType<VM>>()
            .unwrap()
            .set_weak_ref_barrier_enabled(true);
    }
}

pub(in crate::plan) const RESERVED_ALLOCATORS: ReservedAllocators = ReservedAllocators {
    n_bump_pointer: 1,
    ..ReservedAllocators::DEFAULT
};

lazy_static! {
    /// When compressor_single_space is enabled, force all allocations to go to the default allocator and space.
    static ref ALLOCATOR_MAPPING_SINGLE_SPACE: EnumMap<AllocationSemantics, AllocatorSelector> = enum_map! {
        _ => AllocatorSelector::BumpPointer(0),
    };
    pub static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
        if cfg!(feature = "compressor_single_space") {
            *ALLOCATOR_MAPPING_SINGLE_SPACE
        } else {
            let mut map = create_allocator_mapping(RESERVED_ALLOCATORS, true);
            map[AllocationSemantics::Default] = AllocatorSelector::BumpPointer(0);
            map
        }
    };
}

pub fn create_concurrent_compressor_mutator<VM: VMBinding>(
    mutator_tls: VMMutatorThread,
    mmtk: &'static MMTK<VM>,
) -> Mutator<VM> {
    let compressor = mmtk
        .get_plan()
        .downcast_ref::<ConcurrentCompressor<VM>>()
        .unwrap();
    let config = MutatorConfig {
        allocator_mapping: &ALLOCATOR_MAPPING,
        space_mapping: Box::new({
            let mut vec = create_space_mapping(
                RESERVED_ALLOCATORS,
                !cfg!(feature = "compressor_single_space"),
                compressor,
            );
            vec.push((
                AllocatorSelector::BumpPointer(0),
                &compressor.compressor_space,
            ));
            vec
        }),

        prepare_func: &concurent_compressor_mutator_prepare,
        release_func: &concurrent_compressor_mutator_release,
    };

    let builder = MutatorBuilder::new(mutator_tls, mmtk, config);
    let mut mutator = builder
        .barrier(Box::new(SATBBarrier::new(BarrierSemanticsType::<VM>::new(
            mmtk,
            mutator_tls,
        ))))
        .build();

    // Set barrier active, based on whether concurrent marking is in progress
    mutator
        .barrier
        .downcast_mut::<BarrierType<VM>>()
        .unwrap()
        .set_weak_ref_barrier_enabled(compressor.is_concurrent_marking_active());

    mutator
}
