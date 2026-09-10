//! A histogram of pause times, one per pause kind.  `pause_time_stats` feature only.
//!
//! Every pause is measured by the scheduler from the moment a GC worker picks up the GC request
//! to the moment it resumes the mutators -- the same window the "End of GC (took N ms)" log line
//! reports, and the same one the `time.stw` column of the MMTk statistics adds up.  While
//! statistics are being gathered (between `harness_begin` and `harness_end`) each pause is
//! counted into one of four buckets by its duration: under 1 ms, 1 to 10 ms, 10 to 100 ms, and
//! 100 ms or more.  The result is printed with the MMTk statistics at `harness_end`.
//!
//! Full GCs are counted as well as the SATB concurrent collector's `InitialMark` and `FinalMark`
//! pauses.  With the thread-local GC every pause is a full GC: a local GC runs on the mutator
//! thread that requested it and stops nobody else, so it is not a pause and is not counted here.

use crate::plan::Pause;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Upper bounds of the bounded buckets, in nanoseconds.  A pause shorter than the first bound
/// goes into the first bucket, and so on; a pause of at least the last bound goes into the
/// unbounded final bucket.
const BUCKET_UPPER_BOUNDS_NS: [u64; 3] = [1_000_000, 10_000_000, 100_000_000];
const NUM_BUCKETS: usize = BUCKET_UPPER_BOUNDS_NS.len() + 1;
const BUCKET_LABELS: [&str; NUM_BUCKETS] = ["<1ms", "1-10ms", "10-100ms", ">=100ms"];

/// Every pause kind, in the order the rows are printed.
const ALL_PAUSES: [Pause; 3] = [Pause::Full, Pause::InitialMark, Pause::FinalMark];

/// The pause kinds that always get a row, even with no pause counted.  The concurrent
/// collector's pauses are printed only once something was counted, so a plan whose every GC is a
/// single full pause prints a single row.
const ALWAYS_REPORTED: [Pause; 1] = [Pause::Full];

/// The row of `pause` in the tables below.  A `match`, not the discriminant, so the layout of
/// [`Pause`] can change without moving anything here.
fn pause_index(pause: Pause) -> usize {
    match pause {
        Pause::Full => 0,
        Pause::InitialMark => 1,
        Pause::FinalMark => 2,
    }
}

/// The bucket a pause of `duration` is counted into.
fn bucket_index(duration: Duration) -> usize {
    let ns = duration.as_nanos();
    BUCKET_UPPER_BOUNDS_NS
        .iter()
        .position(|&bound| ns < u128::from(bound))
        .unwrap_or(NUM_BUCKETS - 1)
}

/// Per-pause-kind pause-time histogram.  Lock-free: a pause is recorded by the single worker
/// that finishes the GC, and the counters are only read at `harness_end`.
#[derive(Default)]
pub struct PauseTimeStats {
    /// `counts[pause][bucket]`: how many pauses of that kind fell into that bucket.
    counts: [[AtomicUsize; NUM_BUCKETS]; ALL_PAUSES.len()],
    /// The longest pause of each kind, in nanoseconds.  Says how far past the last bound the
    /// pauses in the unbounded bucket went.
    max_ns: [AtomicU64; ALL_PAUSES.len()],
}

impl PauseTimeStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one pause of kind `pause` that took `duration`.
    pub fn record(&self, pause: Pause, duration: Duration) {
        let row = pause_index(pause);
        self.counts[row][bucket_index(duration)].fetch_add(1, Ordering::Relaxed);
        let ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.max_ns[row].fetch_max(ns, Ordering::Relaxed);
    }

    /// How many pauses of kind `pause` have been counted, over all buckets.
    fn total(&self, pause: Pause) -> usize {
        self.counts[pause_index(pause)]
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }

    /// Print the histogram: one row per pause kind, one column per bucket, plus the longest
    /// pause of that kind in milliseconds.
    pub fn print(&self) {
        println!(
            "------------------------------ MMTk Pause Times --------------------------------"
        );
        print!("{:<18}", "pause");
        for label in BUCKET_LABELS {
            print!("{:>10}", label);
        }
        println!("{:>10}", "max(ms)");
        for pause in ALL_PAUSES {
            if !ALWAYS_REPORTED.contains(&pause) && self.total(pause) == 0 {
                continue;
            }
            let row = pause_index(pause);
            print!("{:<18}", format!("{:?}", pause));
            for bucket in &self.counts[row] {
                print!("{:>10}", bucket.load(Ordering::Relaxed));
            }
            println!(
                "{:>10.2}",
                self.max_ns[row].load(Ordering::Relaxed) as f64 / 1e6
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_split_at_1_10_and_100_ms() {
        assert_eq!(bucket_index(Duration::from_nanos(0)), 0);
        assert_eq!(bucket_index(Duration::from_micros(999)), 0);
        assert_eq!(bucket_index(Duration::from_millis(1)), 1);
        assert_eq!(bucket_index(Duration::from_micros(9_999)), 1);
        assert_eq!(bucket_index(Duration::from_millis(10)), 2);
        assert_eq!(bucket_index(Duration::from_micros(99_999)), 2);
        assert_eq!(bucket_index(Duration::from_millis(100)), 3);
        assert_eq!(bucket_index(Duration::from_secs(5)), 3);
    }

    #[test]
    fn records_per_pause_kind() {
        let stats = PauseTimeStats::new();
        stats.record(Pause::InitialMark, Duration::from_micros(400));
        stats.record(Pause::InitialMark, Duration::from_micros(600));
        stats.record(Pause::FinalMark, Duration::from_millis(25));
        stats.record(Pause::Full, Duration::from_millis(250));

        let counts = |pause: Pause| -> Vec<usize> {
            stats.counts[pause_index(pause)]
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect()
        };
        assert_eq!(counts(Pause::InitialMark), vec![2, 0, 0, 0]);
        assert_eq!(counts(Pause::FinalMark), vec![0, 0, 1, 0]);
        assert_eq!(counts(Pause::Full), vec![0, 0, 0, 1]);
        assert_eq!(stats.total(Pause::InitialMark), 2);
        assert_eq!(
            stats.max_ns[pause_index(Pause::InitialMark)].load(Ordering::Relaxed),
            600_000
        );
        assert_eq!(
            stats.max_ns[pause_index(Pause::Full)].load(Ordering::Relaxed),
            250_000_000
        );
    }
}
