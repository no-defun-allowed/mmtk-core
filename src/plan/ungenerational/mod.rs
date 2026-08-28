//! Plan: ungenerational

pub(super) mod gc_work;
pub(super) mod global;
pub(super) mod mutator;

pub use self::global::Ungenerational;
pub use self::global::UG_CONSTRAINTS;
