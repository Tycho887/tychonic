//! Fit range rates with projected contact biases and hard pointing cones.
use super::{ContactBiasMap, ContactId, ContactRangeRate};
use crate::kernels::*;
use numeris::{DynMatrix, DynVector, Matrix};
use satkit::Instant;
use std::collections::{BTreeMap, BTreeSet};
mod initialization;
mod selection;
mod solver;
pub use ripopt::SolveStatus;
pub use solver::{ConstrainedFit, FitError, FitOptions, fit_constrained_range_rate};

/// A measured boresight in local ENU, with azimuth clockwise from north (radians).
#[derive(Debug, Clone, Copy)]
pub struct ContactPointing {
    pub contact: ContactId,
    pub station: GroundStation,
    pub time: Instant,
    pub azimuth: f64,
    pub elevation: f64,
}

/// Use 1–10 distinct epochs near closest approach, or every epoch for qualification.
#[derive(Debug, Clone, Copy)]
pub enum PointingSelection {
    NearTca(usize),
    All,
}

impl Default for PointingSelection {
    fn default() -> Self {
        Self::NearTca(3)
    }
}

pub use super::DEFAULT_SCALES;

#[derive(Debug)]
struct RateRow {
    value: f64,
    weight: f64,
    epoch: usize,
}

#[derive(Debug)]
struct Cone {
    pointing: ContactPointing,
    station_row: usize,
    direction: V3,
    epoch: usize,
}

/// Objective and derivatives in physical parameter units. Constraint Jacobian rows
/// follow `selected_pointings()`. A cone is feasible when its value is at most one.
#[derive(Debug)]
pub struct RangeRateEvaluation {
    pub rss: f64,
    /// Gradient of one half the projected RSS.
    pub gradient: [f64; 7],
    pub biases: BTreeMap<ContactId, f64>,
    pub cone_values: Vec<f64>,
    pub cone_jacobian: Vec<[f64; 7]>,
    pub worst_violation_rad: f64,
}

/// Immutable measurements, projected-bias design, and fixed constraint schedule.
#[derive(Debug)]
pub struct ConstrainedRangeRateProblem {
    config: PropagationConfig,
    seed: V7,
    geometry: PreparedGeometry,
    epochs: Vec<Instant>,
    rates: Vec<RateRow>,
    contacts: Vec<ContactId>,
    bias_map: ContactBiasMap,
    cones: Vec<Cone>,
    half_angle: f64,
    chord: f64,
}

impl ConstrainedRangeRateProblem {
    /// Each contact identifies a single station pass. Supply one common rate sigma
    /// or one per input row (m/s). FOV is the full circular diameter in radians.
    /// Duplicate pointing epochs within a contact use the first supplied reading.
    pub fn new(
        config: PropagationConfig,
        seed: V7,
        rates: &[ContactRangeRate],
        sigma_rr: &[f64],
        pointing: &[ContactPointing],
        fov_full_angle_rad: f64,
        selection: PointingSelection,
    ) -> Result<Self> {
        validate_setup(&config, &seed, fov_full_angle_rad, selection)?;
        let (contacts, bias_map, weights) = prepare_contacts(rates, sigma_rr, pointing)?;
        let times: Vec<_> = rates
            .iter()
            .map(|r| r.time)
            .chain(pointing.iter().map(|p| p.time))
            .collect();
        let sites: Vec<_> = rates
            .iter()
            .map(|r| r.station)
            .chain(pointing.iter().map(|p| p.station))
            .collect();
        let geometry = PreparedGeometry::new(&times, &sites)?;
        let mut cones =
            selection::select(&config, &seed, &geometry, rates.len(), pointing, selection)?;
        let epoch_set: BTreeSet<_> = rates
            .iter()
            .map(|r| r.time)
            .chain(cones.iter().map(|c| c.pointing.time))
            .collect();
        let epochs: Vec<_> = epoch_set.into_iter().collect();
        let indices: BTreeMap<_, _> = epochs.iter().copied().zip(0..).collect();
        let rates = rates
            .iter()
            .zip(weights)
            .map(|(r, weight)| RateRow {
                value: r.range_rate,
                weight,
                epoch: indices[&r.time],
            })
            .collect();
        for cone in &mut cones {
            cone.epoch = indices[&cone.pointing.time];
        }
        Ok(Self {
            config,
            seed,
            geometry,
            epochs,
            rates,
            contacts,
            bias_map,
            cones,
            half_angle: fov_full_angle_rad / 2.,
            chord: 2. * (fov_full_angle_rad / 4.).sin(),
        })
    }

    pub fn seed(&self) -> &V7 {
        &self.seed
    }
    pub fn epochs(&self) -> &[Instant] {
        &self.epochs
    }
    pub fn selected_pointings(&self) -> impl ExactSizeIterator<Item = &ContactPointing> {
        self.cones.iter().map(|c| &c.pointing)
    }

    /// Identify a pair of selected, simultaneous cones from the same site whose
    /// interiors and boundaries are disjoint. General infeasibility is solver-detected.
    pub fn conflicting_pointings(&self) -> Option<(&ContactPointing, &ContactPointing)> {
        let mut simultaneous = BTreeMap::new();
        for cone in &self.cones {
            let key = (cone.pointing.time, station_key(&cone.pointing.station));
            let prior = simultaneous.entry(key).or_insert_with(Vec::<&Cone>::new);
            for other in prior.iter() {
                let separation = other
                    .direction
                    .cross(&cone.direction)
                    .norm()
                    .atan2(other.direction.dot(&cone.direction));
                if separation > 2. * self.half_angle + 16. * f64::EPSILON {
                    return Some((&other.pointing, &cone.pointing));
                }
            }
            prior.push(cone);
        }
        None
    }

    /// One dense sensitivity schedule serves every rate and selected cone.
    pub fn evaluate(&self, parameters: &V7) -> Result<RangeRateEvaluation> {
        Ok(self.evaluate_with_rates(parameters)?.0)
    }

    /// Compute the evaluation and an upper-triangular physical correction
    /// basis from the projected range-rate Jacobian. The last coordinate controls drag.
    pub fn linearize(
        &self,
        parameters: &V7,
        scales: &[f64; 7],
    ) -> Result<(RangeRateEvaluation, M7)> {
        if !scales.iter().all(|v| v.is_finite() && *v > 0.) {
            return Err(KernelError::InvalidInput(
                "expected positive finite parameter scales",
            ));
        }
        let (value, rates) = self.evaluate_with_rates(parameters)?;
        let basis = coordinate_basis(&rates.jacobian, scales)?;
        Ok((value, basis))
    }

    fn evaluate_with_rates(&self, parameters: &V7) -> Result<(RangeRateEvaluation, Evaluation)> {
        let sens = self
            .config
            .sample(&DynVector::from_slice(parameters.as_slice()), &self.epochs)?;
        let mut rates = self.rate_evaluation(&sens)?;
        let biases = self
            .contacts
            .iter()
            .copied()
            .zip(self.bias_map.solve(&rates.residual)?)
            .collect();
        self.bias_map.project(&mut rates)?;
        let rss = rates.rss();
        let gradient: [f64; 7] = std::array::from_fn(|k| {
            (0..self.rates.len())
                .map(|i| rates.jacobian[(i, k)] * rates.residual[i])
                .sum()
        });
        if !rss.is_finite() || !gradient.iter().all(|v| v.is_finite()) {
            return Err(KernelError::Numerical(
                "nonfinite projected objective or gradient",
            ));
        }
        let mut out = RangeRateEvaluation {
            rss,
            gradient,
            biases,
            cone_values: Vec::with_capacity(self.cones.len()),
            cone_jacobian: Vec::with_capacity(self.cones.len()),
            worst_violation_rad: 0.,
        };
        self.evaluate_cones(&sens, &mut out)?;
        Ok((out, rates))
    }

    fn evaluate_cones(&self, sens: &Sensitivities, out: &mut RangeRateEvaluation) -> Result<()> {
        for cone in &self.cones {
            let sample = sample_at(sens, cone.epoch)?;
            let position = station_at(&self.geometry, cone.station_row)?.position();
            let (value, partial, angle) =
                cone_partial(r_of(sample.state) - position, &cone.direction, self.chord)?;
            let jac = compose_sample::<1, 7>(&partial, &sample);
            if !jac.as_slice().iter().all(|v| v.is_finite()) {
                return Err(KernelError::Numerical("nonfinite pointing Jacobian"));
            }
            out.cone_values.push(value);
            out.cone_jacobian.push(std::array::from_fn(|k| jac[(0, k)]));
            out.worst_violation_rad = out.worst_violation_rad.max(angle - self.half_angle);
        }
        Ok(())
    }

    fn rate_evaluation(&self, sens: &Sensitivities) -> Result<Evaluation> {
        let mut out = Evaluation::default();
        out.resize(self.rates.len(), 7)?;
        for (i, row) in self.rates.iter().enumerate() {
            let sample = sample_at(sens, row.epoch)?;
            let station = station_at(&self.geometry, i)?;
            let (value, partial) = range_rate_partial(
                &r_of(sample.state),
                &v_of(sample.state),
                station.position(),
                station.velocity(),
            )?;
            let jac = compose_sample::<1, 7>(&partial, &sample);
            out.residual[i] = (value - row.value) * row.weight;
            for k in 0..7 {
                out.jacobian[(i, k)] = jac[(0, k)] * row.weight;
            }
        }
        out.check_finite()?;
        Ok(out)
    }
}

fn coordinate_basis(jac: &DynMatrix<f64>, scales: &[f64; 7]) -> Result<M7> {
    let rows = jac.nrows();
    // Factor [J D; 0.1 I] once. The small diagonal bounds steps in weak
    // directions and affects only the coordinates, never the likelihood.
    let design = DynMatrix::from_fn(rows + 7, 7, |i, k| {
        if i < rows {
            jac[(i, k)] * scales[k]
        } else if i - rows == k {
            0.1
        } else {
            0.
        }
    });
    let r = design
        .qr()
        .map_err(|_| KernelError::Numerical("seed coordinate QR failed"))?
        .r();
    let upper = M7::from_fn(|i, k| r[(i, k)] * r[(i, i)].signum());
    let inverse = upper
        .inverse()
        .map_err(|_| KernelError::Numerical("singular seed coordinates"))?;
    let basis = M7::from_fn(|i, k| {
        if i <= k {
            scales[i] * inverse[(i, k)]
        } else {
            0.
        }
    });
    if !basis.as_slice().iter().all(|v| v.is_finite()) || basis[(6, 6)] <= 0. {
        return Err(KernelError::Numerical("invalid seed coordinates"));
    }
    Ok(basis)
}

fn cone_partial(line: V3, boresight: &V3, chord: f64) -> Result<(f64, Matrix<f64, 1, 6>, f64)> {
    let range = line.norm();
    if range == 0. {
        return Err(KernelError::ZeroRange);
    }
    let unit = line / range;
    let difference = (unit - boresight) / chord;
    let value = difference.dot(&difference);
    let derivative = (difference - unit * unit.dot(&difference)) * (2. / chord / range);
    let mut partial = Matrix::zeros();
    partial.set_block(0, 0, &derivative.transpose());
    let angle = unit.cross(boresight).norm().atan2(unit.dot(boresight));
    if !range.is_finite() || !value.is_finite() || !angle.is_finite() {
        return Err(KernelError::Numerical("nonfinite pointing geometry"));
    }
    Ok((value, partial, angle))
}

fn station_at(geometry: &PreparedGeometry, i: usize) -> Result<&StationGeometry> {
    geometry
        .row(i)
        .ok_or(KernelError::InvalidInput("missing station sample"))
}
fn sample_at(sens: &Sensitivities, i: usize) -> Result<SensitivityRow<'_>> {
    sens.row(i)
        .ok_or(KernelError::InvalidInput("missing orbit sample"))
}

fn validate_setup(
    config: &PropagationConfig,
    seed: &V7,
    fov: f64,
    selection: PointingSelection,
) -> Result<()> {
    if config.mode().columns() != 7
        || !seed.as_slice().iter().all(|x| x.is_finite())
        || seed[6] < 0.
    {
        return Err(KernelError::InvalidInput(
            "expected finite state-and-drag seed with nonnegative drag",
        ));
    }
    if !(fov > 0. && fov <= std::f64::consts::TAU && (fov / 4.).sin() > 0.) {
        return Err(KernelError::InvalidInput(
            "full FOV must be in (0, 2 pi] radians",
        ));
    }
    if matches!(selection, PointingSelection::NearTca(k) if !(1..=10).contains(&k)) {
        return Err(KernelError::InvalidInput(
            "pointing budget must be 1 through 10",
        ));
    }
    Ok(())
}

fn prepare_contacts(
    rates: &[ContactRangeRate],
    sigmas: &[f64],
    pointing: &[ContactPointing],
) -> Result<(Vec<ContactId>, ContactBiasMap, Vec<f64>)> {
    let weights = rate_weights(sigmas, rates.len())?;
    let mut sites = BTreeMap::new();
    let mut contacts = Vec::new();
    let mut indices = Vec::with_capacity(rates.len());
    for rate in rates {
        let (index, station) = *sites.entry(rate.contact).or_insert_with(|| {
            contacts.push(rate.contact);
            (contacts.len() - 1, rate.station)
        });
        if station != rate.station || !rate.range_rate.is_finite() {
            return Err(KernelError::InvalidInput(
                "expected finite rate and one station per contact",
            ));
        }
        indices.push(Some(index));
    }
    for p in pointing {
        if sites.get(&p.contact).map(|s| s.1) != Some(p.station) {
            return Err(KernelError::InvalidInput(
                "pointing must match a range-rate contact and station",
            ));
        }
    }
    let bias_map = ContactBiasMap::new(&indices, &weights, contacts.len())?;
    Ok((contacts, bias_map, weights))
}

fn rate_weights(sigmas: &[f64], count: usize) -> Result<Vec<f64>> {
    if count == 0 || !(sigmas.len() == 1 || sigmas.len() == count) {
        return Err(KernelError::InvalidInput(
            "expected rates and a common sigma or one sigma per rate",
        ));
    }
    Ok(sigmas.iter().cycle().take(count).map(|s| 1. / s).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cones_at_zenith_wrap_and_narrow_boundary() {
        let angle = 5e-7_f64;
        let chord = 2. * (angle / 2.).sin();
        let boresight = v3(0., 0., 1.);
        let line = v3(angle.sin(), 0., angle.cos()) * 3e5;
        let (value, partial, measured) = cone_partial(line, &boresight, chord).unwrap();
        assert!((value - 1.).abs() < 1e-12 && (measured - angle).abs() < 1e-14);
        assert!(partial.as_slice().iter().all(|v| v.is_finite()));
        assert_eq!(cone_partial(boresight, &boresight, chord).unwrap().0, 0.);
        let id = ValidatedRotation::new(M3::eye()).unwrap();
        let b = convert_az_el_to_gcrf_prepared(-angle, 0., &id).unwrap();
        let u = convert_az_el_to_gcrf_prepared(std::f64::consts::TAU - angle, 0., &id).unwrap();
        assert!(cone_partial(u, &b, chord).unwrap().0 < 1e-16);
    }
}
