use crate::util::constants::BYTES_IN_PAGE;
use crate::util::heap::vm_layout::BYTES_IN_CHUNK;
use crate::util::heap::PageAccounting;
use crate::util::statistics::counter::EventCounter;
use crate::util::statistics::stats::Stats;
use crate::util::Address;
use crate::vm::VMBinding;
use crate::AllocationSemantics;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};

const MINIMUM_HOLE_BYTES: usize = 128;
const MAXIMUM_HOLE_BYTES: usize = BYTES_IN_CHUNK;
pub(crate) type Hole = Range<Address>;

pub(crate) struct HoleList {
    // Supposing we push all holes in address-order,
    // a VecDeque will pop in that address order too.
    pub holes: Mutex<VecDeque<Hole>>,
    wasted: Arc<Mutex<EventCounter>>,
    used: Arc<Mutex<EventCounter>>,
}

pub(crate) fn size(h: &Hole) -> usize {
    h.end - h.start
}
fn pages(h: &Hole) -> usize {
    let bytes = h.end.align_up(BYTES_IN_PAGE) - h.start.align_up(BYTES_IN_PAGE);
    bytes / BYTES_IN_PAGE
}

impl HoleList {
    pub fn new(name: AllocationSemantics, stats: &Stats) -> Self {
        Self {
            holes: Mutex::new(VecDeque::new()),
            wasted: stats.new_event_counter(&format!("wasted-{:?}", name), true, true),
            used: stats.new_event_counter(&format!("used-{:?}", name), true, true),
        }
    }
    /// Allocate a hole of at least `minimum_size` bytes, and targetting `maximum_size`
    /// bytes as a maximum size hint (if non-`None`). The returned hole may be slightly larger
    /// than `maximum_size` bytes due to aligning where we split a hole in half,
    /// but will always be at least `minimum_size` bytes.
    pub fn acquire(
        &self,
        acc: &PageAccounting,
        minimum_size: usize,
        maximum_size: Option<usize>,
    ) -> Option<Hole> {
        let maximum_size = maximum_size.unwrap_or(MAXIMUM_HOLE_BYTES);
        let mut holes = self.holes.lock().unwrap();
        loop {
            match holes.pop_front() {
                None => return None,
                // XXX(hayleyp): This wastes space if we usually serve small
                // allocations, but then end up doing a larger allocation once
                // in a while. Then we will miss holes which are useful for the
                // smaller allocations, even if they're useless for the larger
                // allocations.
                Some(hole) => {
                    if size(&hole) >= minimum_size {
                        // We can use this hole. Can we cut the hole to about maximum_size?
                        // We cut holes only on page boundaries, so that we correctly count
                        // the number of whole free pages at any time.
                        let cut = (hole.start + maximum_size).align_up(BYTES_IN_PAGE);
                        // Would we make a positive-sized hole larger than our
                        // threshold after the cut?
                        let carved = if cut <= (hole.end - MINIMUM_HOLE_BYTES) {
                            // We have a viable hole after the cut.
                            holes.push_back(cut..hole.end);
                            hole.start..cut
                        } else {
                            // We don't have a viable hole after the break; just use the whole hole.
                            hole
                        };
                        acc.reserve_and_commit(pages(&carved));
                        self.used.lock().unwrap().inc_by(size(&carved) as u64);
                        return Some(carved);
                    } else {
                        // We can't use this hole; skip it.
                        self.wasted.lock().unwrap().inc_by(size(&hole) as u64);
                        acc.reserve_and_commit(pages(&hole));
                    }
                }
            }
        }
    }
    pub fn clear(&self, acc: &PageAccounting) {
        let mut holes = self.holes.lock().unwrap();
        let p = holes.iter().map(pages).sum::<usize>();
        acc.reserve_and_commit(p);
        holes.clear();
    }
    pub fn add_holes(&self, acc: &PageAccounting, holes: &[(Address, Address)]) {
        let filtered = holes
            .iter()
            .map(|(s, e)| *s..*e)
            .filter(|h| size(&h) >= MINIMUM_HOLE_BYTES);
        let p = filtered.clone().map(|r| pages(&r)).sum::<usize>();
        acc.release(p);
        let mut holes = self.holes.lock().unwrap();
        holes.extend(filtered);
    }
    pub fn add_hole(&self, acc: &PageAccounting, hole: Range<Address>) {
        if size(&hole) >= MINIMUM_HOLE_BYTES {
            acc.release(pages(&hole));
            let mut holes = self.holes.lock().unwrap();
            holes.push_back(hole);
        }
    }
    pub fn free_bytes(&self) -> usize {
        let holes = self.holes.lock().unwrap();
        holes.iter().map(size).sum()
    }
}
