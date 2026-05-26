use crate::util::Address;

/// Set a range of memory to 0.
pub fn zero(start: Address, len: usize) {
    set(start, 0, len);
}

/// Set a range of memory to the given value. Similar to memset.
pub fn set(start: Address, val: u8, len: usize) {
    unsafe {
        std::ptr::write_bytes(start.to_mut_ptr::<u8>(), val, len);
    }
}

/// Dump RAM around a given address. Note that be careful when using this function as it may
/// segfault for unmapped memory. ONLY use it for locations that are KNOWN to be broken AND
/// allocated by MMTk.
///
/// # Safety
/// This function is unsafe because it may read from unmapped memory, which can cause undefined behavior.
/// The caller must ensure that the address range being read is valid and mapped to avoid potential crashes.
#[allow(unused)]
pub unsafe fn dump_ram_around_address(addr: Address, bytes: usize) -> String {
    let mut string: String = String::new();
    let end_addr = (addr + bytes).to_ptr::<usize>();
    let mut current = (addr - bytes).to_ptr::<usize>();
    while current < end_addr {
        if current == addr.to_ptr::<usize>() {
            string.push_str(" | ");
        } else {
            string.push_str(" ");
        }
        let s = current.read();
        #[cfg(target_pointer_width = "64")]
        string.push_str(format!("{:#018x}", s).as_str());
        #[cfg(target_pointer_width = "32")]
        string.push_str(format!("{:#010x}", s).as_str());
        current = current.add(1);
    }
    string
}
