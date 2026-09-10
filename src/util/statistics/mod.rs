pub use self::counter::Timer;

pub mod counter;
#[cfg(feature = "pause_time_stats")]
pub mod pause_time;
pub mod stats;
