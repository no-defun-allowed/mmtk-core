use crate::util::heap::vm_layout::BYTES_IN_CHUNK;
use crate::util::Address;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Mutex;

const MINIMUM_HOLE_BYTES: usize = 128;
const MAXIMUM_HOLE_BYTES: usize = BYTES_IN_CHUNK;
pub(crate) type Hole = Range<Address>;

pub(crate) struct HoleList {
    // Supposing we push all holes in address-order,
    // a VecDeque will pop in that address order too.
    holes: Mutex<VecDeque<Hole>>
}

fn size(h: &Hole) -> usize { h.end - h.start }

impl HoleList {
    pub fn new() -> Self {
        Self {
            holes: Mutex::new(VecDeque::new()),
        }
    }
    pub fn acquire(&self, minimum_size: usize, maximum_size: Option<usize>) -> Option<Hole> {
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
                Some(hole) => if size(&hole) >= minimum_size {
                    let size_taken = std::cmp::min(maximum_size, size(&hole));
                   let end = hole.start + size_taken;
                    let remaining = end..hole.end;
                    if size(&remaining) >= MINIMUM_HOLE_BYTES {
                        holes.push_back(remaining);
                    }
                    return Some(hole.start..end);
                }
            }
        }
    }
    pub fn clear(&self) {
        let mut holes = self.holes.lock().unwrap();
        holes.clear();
    }
    pub fn add_holes(&self, holes: &[(Address, Address)]) {
        let filtered = holes.iter().map(|(s, e)| *s..*e).filter(|h| size(&h) >= MINIMUM_HOLE_BYTES);
        let mut holes = self.holes.lock().unwrap();
        holes.extend(filtered);
    }
    pub fn add_hole(&self, hole: Range<Address>) {
        if size(&hole) >= MINIMUM_HOLE_BYTES {
            let mut holes = self.holes.lock().unwrap();
            holes.push_back(hole);
        }
    }
    pub fn free_bytes(&self) -> usize {
        let mut holes = self.holes.lock().unwrap();
        holes.iter().map(size).sum()
    }
}
