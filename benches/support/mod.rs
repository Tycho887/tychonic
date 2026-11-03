//! Shared, deterministic inputs and dispatch for benchmarks, profiles and stress tests.
//! This module is deliberately outside the library's public API.
use std::{hint::black_box, str::FromStr};
use tychonic::{numeris, satkit};

use numeris::{Matrix, Quaternion};
use satkit::{Duration, ITRFCoord, Instant, frametransform};
use std::sync::Arc;
use tychonic::kernels::*;

pub const SEED: u64 = 0x7479_6368_6f6e_6963;
pub const SIZES: [usize; 6] = [1, 16, 256, 1024, 16384, 65536];

/// SplitMix64: independent of platform RNG libraries and dependency versions.
pub struct Rng(pub u64);
impl Rng {
    pub fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    SharedRepeated,
    SharedUnique,
    UniqueRepeated,
    UniqueUnique,
    Shuffled,
}
impl Pattern {
    pub const ALL: [Self; 5] = [
        Self::SharedRepeated,
        Self::SharedUnique,
        Self::UniqueRepeated,
        Self::UniqueUnique,
        Self::Shuffled,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::SharedRepeated => "shared-repeated",
            Self::SharedUnique => "shared-unique",
            Self::UniqueRepeated => "unique-repeated",
            Self::UniqueUnique => "unique-unique",
            Self::Shuffled => "shuffled",
        }
    }
}
impl FromStr for Pattern {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|p| p.name() == s)
            .ok_or_else(|| format!("unknown geometry pattern: {s}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Reuse,
    Stream,
}
impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reuse => "reuse",
            Self::Stream => "stream",
        }
    }
    pub fn fixture_rows(self, rows: usize) -> usize {
        match self {
            Self::Reuse => rows.min(16),
            Self::Stream => rows,
        }
    }
}
impl FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "reuse" => Ok(Self::Reuse),
            "stream" => Ok(Self::Stream),
            _ => Err(format!("unknown memory mode: {s}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    TraversalControl,
    RangeRate,
    AzEl,
    AzElRotation,
    LosDirect,
    LosRotation,
    RotateQuaternion,
    RotateMatrix,
    InverseRotation,
    StationGeometry,
    FullJacobian,
    AzElPrepared,
    LosPrepared,
    PreparedGeometry,
    BatchOwned,
    BatchReused,
    BatchReference,
    ChainSeparate,
    ChainCombined,
    ChainAugmented,
}
impl Kernel {
    pub const SCALAR: [Self; 14] = [
        Self::TraversalControl,
        Self::RangeRate,
        Self::AzEl,
        Self::AzElRotation,
        Self::LosDirect,
        Self::LosRotation,
        Self::RotateQuaternion,
        Self::RotateMatrix,
        Self::InverseRotation,
        Self::AzElPrepared,
        Self::LosPrepared,
        Self::ChainSeparate,
        Self::ChainCombined,
        Self::ChainAugmented,
    ];
    pub const ALL: [Self; 20] = [
        Self::TraversalControl,
        Self::RangeRate,
        Self::AzEl,
        Self::AzElRotation,
        Self::LosDirect,
        Self::LosRotation,
        Self::RotateQuaternion,
        Self::RotateMatrix,
        Self::InverseRotation,
        Self::AzElPrepared,
        Self::LosPrepared,
        Self::ChainSeparate,
        Self::ChainCombined,
        Self::ChainAugmented,
        Self::StationGeometry,
        Self::FullJacobian,
        Self::PreparedGeometry,
        Self::BatchOwned,
        Self::BatchReused,
        Self::BatchReference,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::TraversalControl => "traversal-control",
            Self::RangeRate => "range-rate",
            Self::AzEl => "az-el",
            Self::AzElRotation => "az-el-rotation",
            Self::LosDirect => "los-direct",
            Self::LosRotation => "los-rotation",
            Self::RotateQuaternion => "rotate-quaternion",
            Self::RotateMatrix => "rotate-matrix",
            Self::InverseRotation => "inverse-rotation",
            Self::StationGeometry => "station-geometry",
            Self::FullJacobian => "full-jacobian",
            Self::AzElPrepared => "az-el-prepared",
            Self::LosPrepared => "los-prepared",
            Self::PreparedGeometry => "prepared-geometry",
            Self::BatchOwned => "batch-owned",
            Self::BatchReused => "batch-reused",
            Self::BatchReference => "batch-reference",
            Self::ChainSeparate => "chain-separate",
            Self::ChainCombined => "chain-combined",
            Self::ChainAugmented => "chain-augmented",
        }
    }
    pub fn is_batch(self) -> bool {
        matches!(
            self,
            Self::StationGeometry
                | Self::FullJacobian
                | Self::PreparedGeometry
                | Self::BatchOwned
                | Self::BatchReused
                | Self::BatchReference
        )
    }
}
impl FromStr for Kernel {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|k| k.name() == s)
            .ok_or_else(|| format!("unknown kernel: {s}"))
    }
}

pub struct GeometryInput {
    pub times: Vec<Instant>,
    pub sites: Vec<GroundStation>,
}
impl GeometryInput {
    pub fn new(n: usize, pattern: Pattern) -> Self {
        assert!(n > 0);
        let epoch = Instant::from_unixtime(1704110400.25);
        let mut times = Vec::with_capacity(n);
        let mut sites = Vec::with_capacity(n);
        for i in 0..n {
            let t = match pattern {
                Pattern::SharedRepeated | Pattern::SharedUnique => 0,
                Pattern::UniqueRepeated | Pattern::UniqueUnique => i,
                Pattern::Shuffled => i / 16,
            };
            let site = match pattern {
                Pattern::SharedUnique | Pattern::UniqueUnique => i,
                _ => i % 16,
            };
            let count = match pattern {
                Pattern::SharedUnique | Pattern::UniqueUnique => n,
                _ => 16,
            };
            times.push(epoch + Duration::from_seconds(t as f64 * 0.125));
            sites.push(GroundStation {
                lat_deg: -80.0 + 160.0 * (site as f64 + 0.5) / count as f64,
                lon_deg: -175.0 + 350.0 * (site as f64 + 0.5) / count as f64,
                alt_m: (site % 11) as f64 * 100.0,
            });
        }
        if pattern == Pattern::Shuffled {
            let mut rng = Rng(SEED);
            for i in (1..n).rev() {
                let j = (rng.unit() * (i + 1) as f64) as usize;
                times.swap(i, j);
                sites.swap(i, j);
            }
        }
        Self { times, sites }
    }
}

pub struct Fixture {
    pub station: GCRFTrajectory,
    pub enu: EnuBasisTrajectory,
    pub sensitivities: Sensitivities,
    pub coords: Vec<ITRFCoord>,
    pub quaternions: Vec<Quaternion<f64>>,
    pub angles: Vec<[f64; 2]>,
    pub local: Vec<V3>,
    pub prepared_geometry: Arc<PreparedGeometry>,
    pub batch: PreparedBatch,
    pub partials: Vec<Matrix<f64, 3, 6>>,
}
impl Fixture {
    /// Pre-refactor observation assembly, isolated from propagation. Preserves
    /// duplicate prediction/partial calls and intermediate Jacobian buffers
    /// for a comparison with equivalent whitened residual/Jacobian outputs.
    pub fn reference_evaluation(&self) -> Result<Evaluation> {
        let jac = compute_full_jacobian(&self.sensitivities, &self.station, &self.enu)?;
        let n = self.sensitivities.states.len();
        let mut residual = numeris::DynVector::zeros(3 * n);
        let mut j = numeris::DynMatrix::zeros(3 * n, 7);
        for i in 0..n {
            let state = &self.sensitivities.states[i];
            let (rr, _) = range_rate_partial(
                &r_of(state),
                &v_of(state),
                &self.station.position[i],
                &self.station.velocity[i],
            )?;
            let ([az, el], _) = az_el_partial_with_rotation(
                &r_of(state),
                &self.station.position[i],
                &self.enu.enu_to_gcrf[i],
            )?;
            residual[i] = rr / 0.05;
            residual[n + i] = wrap_pi(az) / 1e-4;
            residual[2 * n + i] = el / 2e-4;
            for k in 0..7 {
                j[(i, k)] = jac.range_rate[(i, k)] / 0.05;
                j[(n + i, k)] = jac.azimuth[(i, k)] / 1e-4;
                j[(2 * n + i, k)] = jac.elevation[(i, k)] / 2e-4;
            }
        }
        if !residual
            .as_slice()
            .iter()
            .chain(j.as_slice())
            .all(|x| x.is_finite())
        {
            return Err(KernelError::InvalidInput(
                "reference output exceeds numeric range",
            ));
        }
        Ok(Evaluation {
            residual,
            jacobian: j,
        })
    }

    pub fn new(n: usize) -> Result<Self> {
        let input = GeometryInput::new(n, Pattern::Shuffled);
        let (station, enu) = precompute_station_geometry(&input.times, &input.sites)?;
        let mut rng = Rng(SEED);
        let mut states = Vec::with_capacity(n);
        let mut stm = Vec::with_capacity(n);
        let mut state_dcd = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n);
        let mut quaternions = Vec::with_capacity(n);
        let mut angles = Vec::with_capacity(n);
        let mut local = Vec::with_capacity(n);
        for i in 0..n {
            let site = input.sites[i];
            let coord = ITRFCoord::from_geodetic_deg(site.lat_deg, site.lon_deg, site.alt_m);
            let q = frametransform::qitrf2gcrf_approx(&input.times[i]) * coord.q_enu2itrf();
            let az = (2.0 * rng.unit() - 1.0) * std::f64::consts::PI;
            let el = (2.0 * rng.unit() - 1.0) * 1.4;
            let range = 10_f64.powf(3.0 + 4.6 * rng.unit());
            let los = v3(az.sin() * el.cos(), az.cos() * el.cos(), el.sin());
            let r = station.position[i] + enu.enu_to_gcrf[i] * los * range;
            let v = v3(
                8000.0 * rng.unit(),
                -8000.0 * rng.unit(),
                2000.0 * rng.unit(),
            );
            states.push(v6_from(&r, &v));
            let mut phi = M6::eye();
            phi[(0, 3)] = 120.0;
            phi[(1, 0)] = -0.2;
            phi[(2, 5)] = -60.0;
            phi[(4, 2)] = 0.001;
            stm.push(phi);
            state_dcd.push(numeris::vector![100.0, -70.0, 20.0, 0.3, -0.1, 0.2]);
            coords.push(coord);
            quaternions.push(q);
            angles.push([az, el]);
            local.push(los);
        }
        let prepared_geometry = Arc::new(PreparedGeometry::new(&input.times, &input.sites)?);
        let batch = PreparedBatch::combined(
            prepared_geometry.clone(),
            Observations {
                range_rate: vec![0.0; n],
                azimuth: vec![0.0; n],
                elevation: vec![0.0; n],
                sigma_rr: 0.05,
                sigma_az: 1e-4,
                sigma_el: 2e-4,
            },
        )?;
        let mut partials = Vec::with_capacity(n);
        for (i, state) in states.iter().enumerate() {
            let (_, rr) = range_rate_partial(
                &r_of(state),
                &v_of(state),
                &station.position[i],
                &station.velocity[i],
            )?;
            let (_, angles) = az_el_partial_with_rotation(
                &r_of(state),
                &station.position[i],
                &enu.enu_to_gcrf[i],
            )?;
            let mut h = Matrix::<f64, 3, 6>::zeros();
            h.set_block(0, 0, &rr);
            h.set_block(1, 0, &angles);
            partials.push(h);
        }
        Ok(Self {
            sensitivities: Sensitivities {
                timestamps: input.times,
                states,
                stm,
                state_dcd,
            },
            station,
            enu,
            coords,
            quaternions,
            angles,
            local,
            prepared_geometry,
            batch,
            partials,
        })
    }
}

/// Dispatch outside the row loop: the benchmark does not time an enum match per call.
/// Black-box every result, including Jacobians, so discarded components remain computed.
pub fn run_scalar(kernel: Kernel, f: &Fixture, rows: usize) -> Result<()> {
    let n = f.sensitivities.states.len();
    macro_rules! each {
        ($i:ident, $body:expr) => {
            for start in (0..rows).step_by(n) {
                for $i in 0..n.min(rows - start) {
                    black_box($body);
                }
            }
        };
    }
    match kernel {
        Kernel::TraversalControl => each!(
            i,
            (black_box(&f.enu.enu_to_gcrf[i]), black_box(&f.local[i]))
        ),
        Kernel::RangeRate => each!(
            i,
            range_rate_partial(
                &black_box(r_of(&f.sensitivities.states[i])),
                &black_box(v_of(&f.sensitivities.states[i])),
                black_box(&f.station.position[i]),
                black_box(&f.station.velocity[i]),
            )?
        ),
        Kernel::AzEl => each!(
            i,
            az_el_partial(
                &black_box(r_of(&f.sensitivities.states[i])),
                black_box(&f.station.position[i]),
                &black_box(f.enu.east(i)),
                &black_box(f.enu.north(i)),
                &black_box(f.enu.up(i)),
            )?
        ),
        Kernel::AzElRotation => each!(
            i,
            az_el_partial_with_rotation(
                &black_box(r_of(&f.sensitivities.states[i])),
                black_box(&f.station.position[i]),
                black_box(&f.enu.enu_to_gcrf[i]),
            )?
        ),
        Kernel::LosDirect => each!(
            i,
            convert_az_el_to_gcrf(
                black_box(f.angles[i][0]),
                black_box(f.angles[i][1]),
                black_box(&f.coords[i]),
                black_box(&f.station.timestamps[i]),
            )?
        ),
        Kernel::LosRotation => each!(
            i,
            convert_az_el_to_gcrf_with_rotation(
                black_box(f.angles[i][0]),
                black_box(f.angles[i][1]),
                black_box(&f.enu.enu_to_gcrf[i]),
            )?
        ),
        Kernel::RotateQuaternion => {
            each!(i, *black_box(&f.quaternions[i]) * *black_box(&f.local[i]))
        }
        Kernel::RotateMatrix => each!(
            i,
            *black_box(&f.enu.enu_to_gcrf[i]) * *black_box(&f.local[i])
        ),
        Kernel::InverseRotation => each!(
            i,
            black_box(&f.enu.enu_to_gcrf[i]).transpose()
                * black_box(r_of(&f.sensitivities.states[i]) - f.station.position[i])
        ),
        Kernel::AzElPrepared => each!(
            i,
            az_el_partial_prepared(
                &black_box(r_of(&f.sensitivities.states[i])),
                black_box(&f.station.position[i]),
                black_box(f.prepared_geometry.row(i).unwrap().rotation())
            )?
        ),
        Kernel::LosPrepared => each!(
            i,
            convert_az_el_to_gcrf_prepared(
                black_box(f.angles[i][0]),
                black_box(f.angles[i][1]),
                black_box(f.prepared_geometry.row(i).unwrap().rotation())
            )?
        ),
        Kernel::ChainSeparate => each!(i, {
            let h = black_box(&f.partials[i]);
            let rr = h.block::<1, 6>(0, 0);
            let angles = h.block::<2, 6>(1, 0);
            let phi = black_box(&f.sensitivities.stm[i]);
            let drag = black_box(&f.sensitivities.state_dcd[i]);
            (rr * phi, rr * drag, angles * phi, angles * drag)
        }),
        Kernel::ChainCombined => each!(i, {
            let h = black_box(&f.partials[i]);
            (
                h * black_box(&f.sensitivities.stm[i]),
                h * black_box(&f.sensitivities.state_dcd[i]),
            )
        }),
        Kernel::ChainAugmented => each!(i, {
            let mut sensitivity = Matrix::<f64, 6, 7>::zeros();
            sensitivity.set_block(0, 0, black_box(&f.sensitivities.stm[i]));
            sensitivity.set_block(0, 6, black_box(&f.sensitivities.state_dcd[i]));
            black_box(&f.partials[i]) * sensitivity
        }),
        _ => unreachable!("batch kernels use their own dispatch"),
    }
    Ok(())
}
