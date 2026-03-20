//! Plan: concurrent Compressor

pub(in crate::plan) mod gc_work;
pub(in crate::plan) mod global;
pub(in crate::plan) mod mutator;
pub(in crate::plan) mod concurrent_marking_work;

pub use global::ConcurrentCompressor;
