//! Checked rotations and immutable, batch-local station geometry.
use super::*;
use numeris::Quaternion;
use satkit::{ITRFCoord, Instant, frametransform};
use std::collections::{BTreeMap, HashMap};

/// An orthonormal, right-handed ENU→GCRF rotation, validated once.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedRotation(M3);

impl ValidatedRotation {
    pub fn new(matrix: M3) -> Result<Self> {
        check_basis(
            &matrix.block::<3, 1>(0, 0),
            &matrix.block::<3, 1>(0, 1),
            &matrix.block::<3, 1>(0, 2),
        )?;
        Ok(Self(matrix))
    }

    pub fn matrix(&self) -> &M3 {
        &self.0
    }

    pub fn to_gcrf(&self, local: &V3) -> V3 {
        self.0 * local
    }

    pub fn to_enu(&self, gcrf: &V3) -> V3 {
        self.0.transpose() * gcrf
    }
}

#[derive(Clone, Debug)]
pub struct StationGeometry {
    position: V3,
    velocity: V3,
    rotation: ValidatedRotation,
}

impl StationGeometry {
    pub fn position(&self) -> &V3 {
        &self.position
    }
    pub fn velocity(&self) -> &V3 {
        &self.velocity
    }
    pub fn rotation(&self) -> &ValidatedRotation {
        &self.rotation
    }

    fn new(position: V3, velocity: V3, matrix: M3) -> Result<Self> {
        if !finite(&position) || !finite(&velocity) {
            return Err(KernelError::InvalidInput("station states must be finite"));
        }
        Ok(Self {
            position,
            velocity,
            rotation: ValidatedRotation::new(matrix)?,
        })
    }
}

/// Immutable geometry shared by observation models. Repeated station/epoch
/// rows index the same entry; epochs retain their full `Instant` precision.
#[derive(Clone, Debug)]
pub struct PreparedGeometry {
    timestamps: Vec<Instant>,
    stations: Vec<GroundStation>,
    entries: Vec<StationGeometry>,
    rows: Vec<usize>,
}

type StationKey = (u64, u64, u64);
fn station_key(s: &GroundStation) -> StationKey {
    (s.lat_deg.to_bits(), s.lon_deg.to_bits(), s.alt_m.to_bits())
}

impl PreparedGeometry {
    pub fn new(timestamps: &[Instant], stations: &[GroundStation]) -> Result<Self> {
        Self::with_rotation(timestamps, stations, frametransform::qitrf2gcrf_approx)
    }

    fn with_rotation(
        timestamps: &[Instant],
        stations: &[GroundStation],
        mut rotation: impl FnMut(&Instant) -> Quaternion<f64>,
    ) -> Result<Self> {
        if timestamps.len() != stations.len() {
            return Err(KernelError::InvalidInput("expected one station per epoch"));
        }
        let initial_capacity = timestamps.len().min(16);
        // Strictly monotonic epochs cannot repeat a station/epoch pair. Avoid
        // maintaining two redundant lookup tables for this common OD schedule.
        let unique_epochs = timestamps.windows(2).all(|w| w[0] < w[1])
            || timestamps.windows(2).all(|w| w[0] > w[1]);
        let mut fixed = HashMap::with_capacity(initial_capacity);
        // Instant's raw value is its exact TAI microsecond count, not a rounded
        // floating-point epoch. Hashing it preserves Instant equality exactly.
        let cache_capacity = if unique_epochs { 0 } else { initial_capacity };
        let mut epochs = HashMap::with_capacity(cache_capacity);
        let mut pairs = HashMap::with_capacity(cache_capacity);
        let mut entries = Vec::with_capacity(if unique_epochs {
            timestamps.len()
        } else {
            initial_capacity
        });
        let mut rows = Vec::with_capacity(timestamps.len());
        let omega = v3(0.0, 0.0, satkit::consts::OMEGA_EARTH);
        for (t, station) in timestamps.iter().zip(stations) {
            let key = station_key(station);
            let station_index = fixed.len();
            let (station_index, r, v, enu) = *match fixed.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let coord = station_coord(station.lat_deg, station.lon_deg, station.alt_m)?;
                    let radius = coord.itrf.norm();
                    if !finite(&coord.itrf) || !radius.is_finite() || radius == 0.0 {
                        return Err(KernelError::InvalidInput("invalid station ITRF position"));
                    }
                    e.insert((
                        station_index,
                        coord.itrf,
                        omega.cross(&coord.itrf),
                        coord.q_enu2itrf(),
                    ))
                }
            };
            if unique_epochs {
                let q = rotation(t);
                rows.push(entries.len());
                entries.push(StationGeometry::new(
                    q * r,
                    q * v,
                    (q * enu).to_rotation_matrix(),
                )?);
                continue;
            }
            let index = match pairs.entry((t.raw, station_index)) {
                std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let q = *epochs.entry(t.raw).or_insert_with(|| rotation(t));
                    let index = entries.len();
                    entries.push(StationGeometry::new(
                        q * r,
                        q * v,
                        (q * enu).to_rotation_matrix(),
                    )?);
                    e.insert(index);
                    index
                }
            };
            rows.push(index);
        }
        Ok(Self {
            timestamps: timestamps.to_vec(),
            stations: stations.to_vec(),
            entries,
            rows,
        })
    }

    /// Checked import of externally computed states/rotations. Inputs are
    /// consumed so subsequent mutation cannot invalidate this geometry.
    pub fn from_trajectories(station: GCRFTrajectory, enu: EnuBasisTrajectory) -> Result<Self> {
        let n = station.timestamps.len();
        if station.position.len() != n
            || station.velocity.len() != n
            || station.ground_stations.len() != n
            || enu.timestamps != station.timestamps
            || enu.enu_to_gcrf.len() != n
        {
            return Err(KernelError::InvalidInput(
                "station/ENU lengths or epochs do not match",
            ));
        }
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let s = station.ground_stations[i];
            station_coord(s.lat_deg, s.lon_deg, s.alt_m)?;
            entries.push(StationGeometry::new(
                station.position[i],
                station.velocity[i],
                enu.enu_to_gcrf[i],
            )?);
        }
        Ok(Self {
            timestamps: station.timestamps,
            stations: station.ground_stations,
            entries,
            rows: (0..n).collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }
    pub fn stations(&self) -> &[GroundStation] {
        &self.stations
    }
    pub fn unique_entries(&self) -> usize {
        self.entries.len()
    }
    /// Returns `None` for an out-of-range measurement row.
    pub fn row(&self, i: usize) -> Option<&StationGeometry> {
        self.rows.get(i).map(|&index| &self.entries[index])
    }
}

fn station_coord(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Result<ITRFCoord> {
    if !lat_deg.is_finite()
        || !(-90.0..=90.0).contains(&lat_deg)
        || !lon_deg.is_finite()
        || !alt_m.is_finite()
    {
        return Err(KernelError::InvalidInput(
            "station latitude, longitude, or altitude",
        ));
    }
    Ok(ITRFCoord::from_geodetic_deg(lat_deg, lon_deg, alt_m))
}

#[derive(Clone, Debug)]
pub struct GCRFTrajectory {
    /// Exact observation epochs, including fractional seconds.
    pub timestamps: Vec<Instant>,
    /// Ground-station coordinates associated with each measurement row.
    pub ground_stations: Vec<GroundStation>,
    pub position: Vec<V3>,
    pub velocity: Vec<V3>,
}

#[derive(Clone, Debug)]
pub struct EnuBasisTrajectory {
    pub timestamps: Vec<Instant>,
    /// R_i = R_ITRF→GCRF(t_i) R_ENU→ITRF(lat_i, lon_i).
    /// Columns are east, north and up in GCRF; R_iᵀ maps GCRF vectors to ENU.
    pub enu_to_gcrf: Vec<M3>,
}

impl EnuBasisTrajectory {
    pub fn east(&self, i: usize) -> V3 {
        self.enu_to_gcrf[i].block::<3, 1>(0, 0)
    }
    pub fn north(&self, i: usize) -> V3 {
        self.enu_to_gcrf[i].block::<3, 1>(0, 1)
    }
    pub fn up(&self, i: usize) -> V3 {
        self.enu_to_gcrf[i].block::<3, 1>(0, 2)
    }
}

/// Geodetic coordinates of a station fixed in ITRF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundStation {
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// Height above the reference ellipsoid, in metres.
    pub alt_m: f64,
}

impl GroundStation {
    /// Exact Earth-fixed chord r_self - r_other, expressed in this station's
    /// ENU frame. This includes curvature and altitude for continental baselines.
    pub fn baseline_enu_to(&self, other: &Self) -> Result<V3> {
        let first = station_coord(self.lat_deg, self.lon_deg, self.alt_m)?;
        let second = station_coord(other.lat_deg, other.lon_deg, other.alt_m)?;
        Ok(first.q_enu2itrf().to_rotation_matrix().transpose() * (first.itrf - second.itrf))
    }
}

/// Precompute both station states and ENU→GCRF matrices for a mixed-station
/// batch. One ITRF→GCRF quaternion is evaluated per distinct epoch and reused
/// for every station at that epoch. Fixed ITRF station geometry is also cached.
/// Position, velocity and ENU use the same approximate IAU-76/FK5 frame chain.
pub fn precompute_station_geometry(
    timestamps: &[Instant],
    stations: &[GroundStation],
) -> Result<(GCRFTrajectory, EnuBasisTrajectory)> {
    precompute_station_geometry_with_rotation(timestamps, stations, |t| {
        frametransform::qitrf2gcrf_approx(t)
    })
}

fn precompute_station_geometry_with_rotation(
    timestamps: &[Instant],
    stations: &[GroundStation],
    mut rotation: impl FnMut(&Instant) -> Quaternion<f64>,
) -> Result<(GCRFTrajectory, EnuBasisTrajectory)> {
    if timestamps.len() != stations.len() {
        return Err(KernelError::InvalidInput(
            "expected one ground station per measurement epoch",
        ));
    }
    let mut station_geometry = HashMap::new();
    for station in stations {
        let key = (
            station.lat_deg.to_bits(),
            station.lon_deg.to_bits(),
            station.alt_m.to_bits(),
        );
        if let std::collections::hash_map::Entry::Vacant(entry) = station_geometry.entry(key) {
            let coord = station_coord(station.lat_deg, station.lon_deg, station.alt_m)?;
            let radius = coord.itrf.norm();
            if !finite(&coord.itrf) || !radius.is_finite() || radius == 0.0 {
                return Err(KernelError::InvalidInput("invalid station ITRF position"));
            }
            entry.insert((coord.itrf, coord.q_enu2itrf()));
        }
    }
    let mut trajectory = GCRFTrajectory {
        timestamps: timestamps.to_vec(),
        ground_stations: stations.to_vec(),
        position: Vec::with_capacity(timestamps.len()),
        velocity: Vec::with_capacity(timestamps.len()),
    };
    let mut enu = EnuBasisTrajectory {
        timestamps: timestamps.to_vec(),
        enu_to_gcrf: Vec::with_capacity(timestamps.len()),
    };
    let mut rotations = BTreeMap::new();
    let omega = v3(0.0, 0.0, satkit::consts::OMEGA_EARTH);
    for (t, station) in timestamps.iter().zip(stations) {
        let q = *rotations.entry(*t).or_insert_with(|| rotation(t));
        let key = (
            station.lat_deg.to_bits(),
            station.lon_deg.to_bits(),
            station.alt_m.to_bits(),
        );
        let (itrf, enu_to_itrf) = station_geometry[&key];
        trajectory.position.push(q * itrf);
        // A fixed station has zero ITRF velocity; Earth rotation adds ω × r.
        trajectory.velocity.push(q * omega.cross(&itrf));
        enu.enu_to_gcrf.push((q * enu_to_itrf).to_rotation_matrix());
    }
    Ok((trajectory, enu))
}

/// Convenience wrapper for a single fixed station. Use
/// `precompute_station_geometry` to compute state and rotation together.
pub fn compute_station_trajectory(
    timestamps: &[Instant],
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
) -> Result<GCRFTrajectory> {
    station_coord(lat_deg, lon_deg, alt_m)?;
    let stations = vec![
        GroundStation {
            lat_deg,
            lon_deg,
            alt_m
        };
        timestamps.len()
    ];
    Ok(precompute_station_geometry(timestamps, &stations)?.0)
}

pub fn compute_enu_basis(
    timestamps: &[Instant],
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
) -> Result<EnuBasisTrajectory> {
    station_coord(lat_deg, lon_deg, alt_m)?;
    let stations = vec![
        GroundStation {
            lat_deg,
            lon_deg,
            alt_m
        };
        timestamps.len()
    ];
    Ok(precompute_station_geometry(timestamps, &stations)?.1)
}

pub(super) fn check_basis(e: &V3, n: &V3, u: &V3) -> Result<()> {
    if !finite(e)
        || !finite(n)
        || !finite(u)
        || (e.norm() - 1.0).abs() > 1e-10
        || (n.norm() - 1.0).abs() > 1e-10
        || (u.norm() - 1.0).abs() > 1e-10
        || e.dot(n).abs() > 1e-10
        || e.dot(u).abs() > 1e-10
        || n.dot(u).abs() > 1e-10
        || (e.cross(n) - u).norm() > 1e-10
    {
        return Err(KernelError::InvalidInput(
            "ENU basis must be finite, orthonormal and right-handed",
        ));
    }
    Ok(())
}

/// Unit line of sight using the same approximate frame chain as station/ENU.
pub fn convert_az_el_to_gcrf(az: f64, el: f64, coord: &ITRFCoord, time: &Instant) -> Result<V3> {
    if !az.is_finite()
        || !el.is_finite()
        || !(-PI / 2.0..=PI / 2.0).contains(&el)
        || !finite(&coord.itrf)
    {
        return Err(KernelError::InvalidInput(
            "LOS angles or station coordinate",
        ));
    }
    let rotation =
        (frametransform::qitrf2gcrf_approx(time) * coord.q_enu2itrf()).to_rotation_matrix();
    convert_az_el_to_gcrf_with_rotation(az, el, &rotation)
}

/// Unit GCRF line of sight using a stored ENU→GCRF matrix, with no frame work.
pub fn convert_az_el_to_gcrf_with_rotation(az: f64, el: f64, rotation: &M3) -> Result<V3> {
    convert_az_el_to_gcrf_prepared(az, el, &ValidatedRotation::new(*rotation)?)
}

/// Unit GCRF LOS with a rotation validated during preparation.
pub fn convert_az_el_to_gcrf_prepared(
    az: f64,
    el: f64,
    rotation: &ValidatedRotation,
) -> Result<V3> {
    if !az.is_finite() || !el.is_finite() || !(-PI / 2.0..=PI / 2.0).contains(&el) {
        return Err(KernelError::InvalidInput("LOS angles"));
    }
    Ok(rotation.to_gcrf(&v3(az.sin() * el.cos(), az.cos() * el.cos(), el.sin())))
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use satkit::Duration;

    #[test]
    fn prepared_geometry_computes_frame_rotation_once_per_exact_epoch() {
        let t = Instant::from_unixtime(1704110400.25);
        let later = t + Duration::from_seconds(0.000001);
        let a = GroundStation {
            lat_deg: 40.0,
            lon_deg: -75.0,
            alt_m: 100.0,
        };
        let b = GroundStation { alt_m: 2100.0, ..a };
        let times = [later, t, t, later, t];
        let stations = [a, b, a, a, a];
        let mut calls = Vec::new();
        let prepared = PreparedGeometry::with_rotation(&times, &stations, |time| {
            calls.push(*time);
            frametransform::qitrf2gcrf_approx(time)
        })
        .unwrap();
        assert_eq!(calls, vec![later, t]);
        assert_eq!(prepared.unique_entries(), 3);
        assert!(std::ptr::eq(
            prepared.row(0).unwrap(),
            prepared.row(3).unwrap()
        ));
        assert!(std::ptr::eq(
            prepared.row(2).unwrap(),
            prepared.row(4).unwrap()
        ));
        for ordered in [[t, later], [later, t]] {
            calls.clear();
            let geometry = PreparedGeometry::with_rotation(&ordered, &[a, a], |time| {
                calls.push(*time);
                frametransform::qitrf2gcrf_approx(time)
            })
            .unwrap();
            assert_eq!(calls, ordered);
            assert_eq!(geometry.timestamps(), ordered);
            assert_eq!(geometry.unique_entries(), 2);
        }
    }

    #[test]
    fn frame_rotation_is_computed_once_per_distinct_epoch() {
        let t = Instant::from_unixtime(1704110400.25);
        let later = t + Duration::from_seconds(0.125);
        let times = [later, t, t, later, t];
        let sites = [
            GroundStation {
                lat_deg: 40.0,
                lon_deg: -75.0,
                alt_m: 100.0,
            },
            GroundStation {
                lat_deg: -30.0,
                lon_deg: 120.0,
                alt_m: 300.0,
            },
        ];
        let stations = [sites[0], sites[1], sites[0], sites[1], sites[0]];
        let mut calls = Vec::new();
        let (trajectory, enu) =
            precompute_station_geometry_with_rotation(&times, &stations, |time| {
                calls.push(*time);
                frametransform::qitrf2gcrf_approx(time)
            })
            .unwrap();
        assert_eq!(calls, vec![later, t]);
        assert_eq!(trajectory.timestamps, times);
        assert_eq!(trajectory.ground_stations, stations);
        assert_eq!(trajectory.position[2], trajectory.position[4]);
        assert_eq!(enu.enu_to_gcrf[2], enu.enu_to_gcrf[4]);
        assert_ne!(trajectory.position[1], trajectory.position[2]);
    }
}
