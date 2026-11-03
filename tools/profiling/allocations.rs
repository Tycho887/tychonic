//! Enabled only in the allocation-counting build, never the timing build.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

pub struct CountingAllocator;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

fn allocated(size: usize, realloc: bool) {
    if ACTIVE.load(Ordering::Relaxed) {
        if realloc {
            REALLOCS.fetch_add(1, Ordering::Relaxed);
        } else {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

// SAFETY: all allocation operations delegate to System with unchanged layouts;
// bookkeeping uses atomics and never allocates or alters the returned memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size(), false);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size(), false);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ACTIVE.load(Ordering::Relaxed) {
            FREES.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            allocated(size, true);
        }
        new
    }
}

pub fn start() {
    for counter in [&ALLOCS, &REALLOCS, &FREES, &BYTES] {
        counter.store(0, Ordering::Relaxed);
    }
    ACTIVE.store(true, Ordering::Relaxed);
}
pub fn stop() -> [u64; 4] {
    ACTIVE.store(false, Ordering::Relaxed);
    [&ALLOCS, &REALLOCS, &FREES, &BYTES].map(|c| c.load(Ordering::Relaxed))
}
