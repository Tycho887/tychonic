#[path = "support/mod.rs"]
#[allow(dead_code)]
mod support;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use numeris::{DynMatrix, DynVector};
use std::hint::black_box;
use support::*;
use tychonic::{
    kernels::{Evaluation, PreparedGeometry, compute_full_jacobian, precompute_station_geometry},
    optim::ContactBiasMap,
};

// Variable projection operates on prebuilt whitened residuals and Jacobians;
// no propagation or Jacobian assembly is included in these measurements.
const VARPRO_ROWS: [usize; 2] = [1024, 8192];
const VARPRO_CONTACTS: [usize; 3] = [1, 8, 32];
const VARPRO_COLUMNS: usize = 7;

struct VarproFixture {
    map: ContactBiasMap,
    evaluation: Evaluation,
    residual: DynVector<f64>,
}

fn varpro_fixture(rows: usize, contacts: usize) -> VarproFixture {
    assert!(contacts > 0 && rows >= contacts);
    let mut rng = Rng(SEED);
    let row_contact: Vec<Option<usize>> = (0..rows).map(|row| Some(row % contacts)).collect();
    let row_weight: Vec<f64> = (0..rows)
        .map(|_| 1.0 / (0.02 + 0.08 * rng.unit()))
        .collect();
    let mut residual = DynVector::zeros(rows);
    let mut jacobian = DynMatrix::zeros(rows, VARPRO_COLUMNS);
    for row in 0..rows {
        residual[row] = 4.0 * rng.unit() - 2.0;
        for column in 0..VARPRO_COLUMNS {
            jacobian[(row, column)] = 2.0 * rng.unit() - 1.0;
        }
    }
    let map = ContactBiasMap::new(&row_contact, &row_weight, contacts).expect("valid bias map");
    VarproFixture {
        map,
        evaluation: Evaluation {
            residual: residual.clone(),
            jacobian,
        },
        residual,
    }
}

fn benchmarks(c: &mut Criterion) {
    // Optional size selection keeps an initial baseline manageable without changing defaults.
    let sizes: Vec<usize> = match std::env::var("TYCHONIC_BENCH_SIZES") {
        Ok(s) => s
            .split(',')
            .map(|x| x.parse::<usize>().expect("positive comma-separated sizes"))
            .inspect(|n| assert!(*n > 0))
            .collect(),
        Err(_) => SIZES.to_vec(),
    };
    for kernel in Kernel::SCALAR {
        for mode in [Mode::Reuse, Mode::Stream] {
            let mut group = c.benchmark_group(format!("{}/{}", kernel.name(), mode.name()));
            for &n in &sizes {
                let f = Fixture::new(mode.fixture_rows(n)).expect("offline fixture geometry");
                group.throughput(Throughput::Elements(n as u64));
                group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &rows| {
                    b.iter(|| run_scalar(kernel, black_box(&f), black_box(rows)).unwrap());
                });
            }
            group.finish();
        }
    }
    let mut group = c.benchmark_group("full-jacobian/stream");
    for &n in &sizes {
        let f = Fixture::new(n).expect("offline fixture geometry");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    compute_full_jacobian(
                        black_box(&f.sensitivities),
                        black_box(&f.station),
                        black_box(&f.enu),
                    )
                    .unwrap(),
                )
            });
        });
    }
    group.finish();
    for kernel in [
        Kernel::BatchOwned,
        Kernel::BatchReused,
        Kernel::BatchReference,
    ] {
        let mut group = c.benchmark_group(format!("{}/stream", kernel.name()));
        for &n in &sizes {
            let f = Fixture::new(n).unwrap();
            let mut output = Evaluation::default();
            f.batch
                .evaluate_from_sensitivities_into(&f.sensitivities, &mut output)
                .unwrap();
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    if kernel == Kernel::BatchReference {
                        black_box(f.reference_evaluation().unwrap());
                    } else if kernel == Kernel::BatchOwned {
                        black_box(
                            f.batch
                                .evaluate_from_sensitivities(black_box(&f.sensitivities))
                                .unwrap(),
                        );
                    } else {
                        f.batch
                            .evaluate_from_sensitivities_into(
                                black_box(&f.sensitivities),
                                &mut output,
                            )
                            .unwrap();
                        black_box(&output);
                    }
                });
            });
        }
        group.finish();
    }
    for pattern in Pattern::ALL {
        let mut group = c.benchmark_group(format!("prepared-geometry/{}", pattern.name()));
        for &n in &sizes {
            let input = GeometryInput::new(n, pattern);
            black_box(PreparedGeometry::new(&input.times, &input.sites).unwrap());
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    black_box(
                        PreparedGeometry::new(black_box(&input.times), black_box(&input.sites))
                            .unwrap(),
                    )
                });
            });
        }
        group.finish();
    }
    for pattern in Pattern::ALL {
        let mut group = c.benchmark_group(format!("station-geometry/{}", pattern.name()));
        for &n in &sizes {
            let input = GeometryInput::new(n, pattern);
            // Load data and complete satkit lazy initialization before timing.
            black_box(precompute_station_geometry(&input.times, &input.sites).unwrap());
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    black_box(
                        precompute_station_geometry(
                            black_box(&input.times),
                            black_box(&input.sites),
                        )
                        .unwrap(),
                    )
                });
            });
        }
        group.finish();
    }
}

fn varpro_benchmarks(c: &mut Criterion) {
    let rows: Vec<usize> = match std::env::var("TYCHONIC_BENCH_SIZES") {
        Ok(s) => s
            .split(',')
            .map(|x| x.parse::<usize>().expect("positive comma-separated sizes"))
            .inspect(|n| assert!(*n > 0))
            .collect(),
        Err(_) => VARPRO_ROWS.to_vec(),
    };
    // Every contact needs at least one row; skip contact counts above a
    // reduced row sweep.
    let fixtures: Vec<_> = rows
        .iter()
        .flat_map(|&row_count| {
            VARPRO_CONTACTS
                .iter()
                .copied()
                .filter(move |&contacts| contacts <= row_count)
                .map(move |contacts| (row_count, contacts, varpro_fixture(row_count, contacts)))
        })
        .collect();
    let mut group = c.benchmark_group("varpro/project");
    for (row_count, contacts, fixture) in &fixtures {
        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{row_count}r-{contacts}c")),
            row_count,
            |b, _| {
                b.iter_batched(
                    || fixture.evaluation.clone(),
                    |mut evaluation| {
                        fixture.map.project(&mut evaluation).unwrap();
                        black_box(evaluation)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
    let mut group = c.benchmark_group("varpro/solve");
    for (row_count, contacts, fixture) in &fixtures {
        group.throughput(Throughput::Elements(*row_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{row_count}r-{contacts}c")),
            row_count,
            |b, _| {
                b.iter(|| black_box(fixture.map.solve(black_box(&fixture.residual)).unwrap()));
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = kernels;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(30);
    targets = benchmarks, varpro_benchmarks
}
criterion_main!(kernels);
