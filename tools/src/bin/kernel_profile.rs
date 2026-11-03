//! A fixed-work profiling runner; setup and warm-up are outside reported measurements.
#[cfg(feature = "profile-allocations")]
#[path = "../../profiling/allocations.rs"]
mod allocations;
#[path = "../../../benches/support/mod.rs"]
#[allow(dead_code)]
mod support;
#[cfg(feature = "profile-allocations")]
#[global_allocator]
static ALLOCATOR: allocations::CountingAllocator = allocations::CountingAllocator;

use std::{
    error::Error,
    hint::black_box,
    io::Write,
    time::{Duration, Instant},
};
use support::*;
use tychonic::kernels::{
    Evaluation, PreparedGeometry, compute_full_jacobian, precompute_station_geometry,
};

const HELP: &str = "kernel_profile --kernel NAME [--rows 1024] [--iterations 10000]
    [--mode stream|reuse] [--pattern shuffled] [--warmup-seconds 2] [--wait-for-start]

Kernels: traversal-control, range-rate, az-el, az-el-rotation, los-direct, los-rotation,
         rotate-quaternion, rotate-matrix, inverse-rotation, station-geometry, prepared-geometry, full-jacobian, batch-owned, batch-reused, batch-reference,
         az-el-prepared, los-prepared, chain-separate, chain-combined, chain-augmented
Patterns: shared-repeated, shared-unique, unique-repeated, unique-unique, shuffled
Reuse traverses at most 16 prebuilt rows. Stream traverses all requested rows.
Batch kernels require stream. Pattern applies only to geometry kernels.
--wait-for-start waits for a newline on stdin after warm-up, enabling perf attachment.
Use --features profile-allocations for counts instead of timing. Output is one JSON record.";

fn main() -> Result<(), Box<dyn Error>> {
    let mut kernel = Kernel::AzElRotation;
    let mut rows = 1024_usize;
    let mut iterations = 10000_u64;
    let mut mode = Mode::Stream;
    let mut pattern = Pattern::Shuffled;
    let mut pattern_set = false;
    let mut warmup = 2.0_f64;
    let mut wait_for_start = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        if arg == "--wait-for-start" {
            wait_for_start = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--kernel" => kernel = value.parse()?,
            "--rows" => rows = value.parse()?,
            "--iterations" => iterations = value.parse()?,
            "--mode" => mode = value.parse()?,
            "--pattern" => {
                pattern = value.parse()?;
                pattern_set = true;
            }
            "--warmup-seconds" => warmup = value.parse()?,
            _ => return Err(format!("unknown option {arg}; use --help").into()),
        }
    }
    if rows == 0 || iterations == 0 || !warmup.is_finite() || !(0.0..=3600.0).contains(&warmup) {
        return Err("rows/iterations must be positive; warm-up must be finite in [0, 3600]".into());
    }
    if kernel.is_batch() && mode != Mode::Stream {
        return Err("batch kernels require --mode stream".into());
    }
    if pattern_set && !matches!(kernel, Kernel::StationGeometry | Kernel::PreparedGeometry) {
        return Err("--pattern applies only to geometry kernels".into());
    }
    let total_rows = (rows as u64)
        .checked_mul(iterations)
        .ok_or("row count overflow")?;
    let geometry = matches!(kernel, Kernel::StationGeometry | Kernel::PreparedGeometry)
        .then(|| GeometryInput::new(rows, pattern));
    let fixture = if geometry.is_none() {
        Some(Fixture::new(mode.fixture_rows(rows))?)
    } else {
        None
    };
    let mut output = Evaluation::default();
    let mut run = || -> tychonic::kernels::Result<()> {
        match kernel {
            Kernel::StationGeometry => {
                let g = geometry.as_ref().unwrap();
                black_box(precompute_station_geometry(
                    black_box(&g.times),
                    black_box(&g.sites),
                )?);
            }
            Kernel::PreparedGeometry => {
                let g = geometry.as_ref().unwrap();
                black_box(PreparedGeometry::new(
                    black_box(&g.times),
                    black_box(&g.sites),
                )?);
            }
            Kernel::BatchOwned => {
                let f = fixture.as_ref().unwrap();
                black_box(
                    f.batch
                        .evaluate_from_sensitivities(black_box(&f.sensitivities))?,
                );
            }
            Kernel::BatchReference => {
                black_box(fixture.as_ref().unwrap().reference_evaluation()?);
            }
            Kernel::BatchReused => {
                let f = fixture.as_ref().unwrap();
                f.batch
                    .evaluate_from_sensitivities_into(black_box(&f.sensitivities), &mut output)?;
                black_box(&output);
            }
            Kernel::FullJacobian => {
                let f = fixture.as_ref().unwrap();
                black_box(compute_full_jacobian(
                    black_box(&f.sensitivities),
                    black_box(&f.station),
                    black_box(&f.enu),
                )?);
            }
            _ => run_scalar(
                kernel,
                black_box(fixture.as_ref().unwrap()),
                black_box(rows),
            )?,
        }
        Ok(())
    };
    // Always execute at least once, even with zero timed warm-up, to prime lazy data.
    let begin = Instant::now();
    let mut warmup_batches = 0_u64;
    loop {
        run()?;
        warmup_batches += 1;
        if begin.elapsed() >= Duration::from_secs_f64(warmup) {
            break;
        }
    }
    if wait_for_start {
        eprintln!("READY pid={}", std::process::id());
        std::io::stderr().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Err("stdin closed before start".into());
        }
    }
    #[cfg(feature = "profile-allocations")]
    allocations::start();
    #[cfg(not(feature = "profile-allocations"))]
    let begin = Instant::now();
    for _ in 0..iterations {
        run()?;
    }
    #[cfg(feature = "profile-allocations")]
    let [allocs, reallocs, frees, bytes] = allocations::stop();
    #[cfg(not(feature = "profile-allocations"))]
    let elapsed_ns = begin.elapsed().as_nanos();
    print!(
        "{{\"kernel\":\"{}\",\"mode\":\"{}\",\"pattern\":{},\"rows\":{rows},\"iterations\":{iterations},\"warmup_batches\":{warmup_batches},\"seed\":{SEED}",
        kernel.name(),
        mode.name(),
        if matches!(kernel, Kernel::StationGeometry | Kernel::PreparedGeometry) {
            format!("\"{}\"", pattern.name())
        } else {
            "null".to_string()
        }
    );
    #[cfg(feature = "profile-allocations")]
    println!(
        ",\"measurement\":\"allocations\",\"allocations\":{allocs},\"reallocations\":{reallocs},\"deallocations\":{frees},\"requested_bytes\":{bytes},\"bytes_per_row\":{}}}",
        bytes as f64 / total_rows as f64
    );
    #[cfg(not(feature = "profile-allocations"))]
    println!(
        ",\"measurement\":\"timing\",\"elapsed_ns\":{elapsed_ns},\"ns_per_row\":{},\"rows_per_second\":{}}}",
        elapsed_ns as f64 / total_rows as f64,
        total_rows as f64 * 1e9 / elapsed_ns as f64
    );
    Ok(())
}
