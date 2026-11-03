//! Variable projection for per-contact constant range-rate biases.
use super::{FitError, ODProblem, run_lm};
use crate::kernels::{
    Evaluation, GroundStation, KernelError, MeasurementNoise, PreparedBatch, PreparedGeometry,
    PropagationConfig, Result,
};
use numeris::{DynMatrix, DynVector, optim::LmSettings};
use satkit::Instant;
use std::{cell::RefCell, collections::BTreeMap, fmt, rc::Rc, sync::Arc};

/// A contact identifier stored as a 128-bit UUID value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContactId(u128);

impl ContactId {
    pub const fn new(value: u128) -> Self {
        Self(value)
    }
    pub const fn as_u128(self) -> u128 {
        self.0
    }
    /// Accept 32 bare hex digits or the hyphenated 8-4-4-4-12 form.
    pub fn parse(text: &str) -> Result<Self> {
        let bytes = text.as_bytes();
        let hyphenated = match bytes.len() {
            32 => false,
            36 => {
                if bytes[8] != b'-' || bytes[13] != b'-' || bytes[18] != b'-' || bytes[23] != b'-' {
                    return Err(KernelError::InvalidInput(
                        "contact identifier is not a UUID",
                    ));
                }
                true
            }
            _ => {
                return Err(KernelError::InvalidInput(
                    "contact identifier must have 32 hex digits",
                ));
            }
        };
        let mut value = 0u128;
        for (index, byte) in bytes.iter().enumerate() {
            if hyphenated && matches!(index, 8 | 13 | 18 | 23) {
                continue;
            }
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => {
                    return Err(KernelError::InvalidInput(
                        "contact identifier contains a non-hex digit",
                    ));
                }
            };
            value = (value << 4) | u128::from(digit);
        }
        Ok(Self(value))
    }
}

impl From<u128> for ContactId {
    fn from(value: u128) -> Self {
        Self(value)
    }
}

impl fmt::Debug for ContactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContactId(0x{:032x})", self.0)
    }
}

impl fmt::Display for ContactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// One observed range-rate sample belonging to a contact.
#[derive(Clone, Copy, Debug)]
pub struct ContactRangeRate {
    pub time: Instant,
    pub station: GroundStation,
    pub contact: ContactId,
    pub range_rate: f64,
}

/// Weak azimuth/elevation pseudo-measurement without a bias term.
#[derive(Clone, Copy, Debug)]
pub struct PointingConstraint {
    pub time: Instant,
    pub station: GroundStation,
    pub azimuth: f64,
    pub elevation: f64,
}

/// Diagonal whitening structure of the contact bias columns.
#[derive(Clone, Debug)]
pub struct ContactBiasMap {
    row_contact: Vec<Option<usize>>,
    row_weight: Vec<f64>,
    norm: Vec<f64>,
    bias_rows: usize,
}

impl ContactBiasMap {
    /// `row_weight` is the whitening weight (1/sigma) of each bias row.
    pub fn new(row_contact: &[Option<usize>], row_weight: &[f64], contacts: usize) -> Result<Self> {
        if row_contact.len() != row_weight.len() {
            return Err(KernelError::InvalidInput(
                "bias row map lengths do not match",
            ));
        }
        let mut norm = vec![0.0; contacts];
        let mut bias_rows = 0usize;
        for (&contact, &weight) in row_contact.iter().zip(row_weight) {
            let Some(contact) = contact else { continue };
            if contact >= contacts {
                return Err(KernelError::InvalidInput("bias contact index out of range"));
            }
            if !weight.is_finite() || weight <= 0.0 {
                return Err(KernelError::InvalidInput(
                    "bias row weights must be finite and positive",
                ));
            }
            norm[contact] += weight * weight;
            if !norm[contact].is_finite() {
                return Err(KernelError::InvalidInput("bias normalisation overflows"));
            }
            bias_rows = bias_rows
                .checked_add(1)
                .ok_or(KernelError::InvalidInput("bias row count overflows"))?;
        }
        if norm.contains(&0.0) {
            return Err(KernelError::InvalidInput(
                "every contact needs at least one bias row",
            ));
        }
        Ok(Self {
            row_contact: row_contact.to_vec(),
            row_weight: row_weight.to_vec(),
            norm,
            bias_rows,
        })
    }

    pub fn contacts(&self) -> usize {
        self.norm.len()
    }

    pub fn bias_rows(&self) -> usize {
        self.bias_rows
    }

    /// Apply `P = I - A (AᵀA)⁻¹ Aᵀ` to a residual and Jacobian in place.
    pub fn project(&self, evaluation: &mut Evaluation) -> Result<()> {
        let rows = self.row_contact.len();
        let columns = evaluation.jacobian.ncols();
        if evaluation.residual.len() != rows || evaluation.jacobian.nrows() != rows || columns == 0
        {
            return Err(KernelError::InvalidInput(
                "evaluation dimensions do not match bias rows",
            ));
        }
        let contacts = self.norm.len();
        let gram_len = contacts
            .checked_mul(columns)
            .ok_or(KernelError::InvalidInput(
                "bias accumulator dimensions overflow",
            ))?;
        let mut sum = vec![0.0; contacts];
        let mut gram = vec![0.0; gram_len];
        for (row, &contact) in self.row_contact.iter().enumerate() {
            let Some(contact) = contact else { continue };
            let weight = self.row_weight[row];
            sum[contact] += weight * evaluation.residual[row];
            for k in 0..columns {
                gram[contact * columns + k] += weight * evaluation.jacobian[(row, k)];
            }
        }
        for (row, &contact) in self.row_contact.iter().enumerate() {
            let Some(contact) = contact else { continue };
            let scale = self.row_weight[row] / self.norm[contact];
            let residual = evaluation.residual[row] - scale * sum[contact];
            if !residual.is_finite() {
                return Err(KernelError::InvalidInput(
                    "projected residual is not finite",
                ));
            }
            evaluation.residual[row] = residual;
            for k in 0..columns {
                let value = evaluation.jacobian[(row, k)] - scale * gram[contact * columns + k];
                if !value.is_finite() {
                    return Err(KernelError::InvalidInput(
                        "projected Jacobian is not finite",
                    ));
                }
                evaluation.jacobian[(row, k)] = value;
            }
        }
        Ok(())
    }

    /// Optimal biases `b = -(AᵀA)⁻¹ Aᵀ r` in m/s.
    pub fn solve(&self, residual: &DynVector<f64>) -> Result<Vec<f64>> {
        if residual.len() != self.row_contact.len() {
            return Err(KernelError::InvalidInput(
                "residual length does not match bias rows",
            ));
        }
        let mut sum = vec![0.0; self.norm.len()];
        for (row, &contact) in self.row_contact.iter().enumerate() {
            if let Some(contact) = contact {
                sum[contact] += self.row_weight[row] * residual[row];
            }
        }
        Ok((0..self.norm.len())
            .map(|contact| -sum[contact] / self.norm[contact])
            .collect())
    }
}

/// Range-rate contacts with eliminated biases plus optional pointing constraints.
#[derive(Clone, Debug)]
pub struct ContactBiasProblem {
    inner: ODProblem,
    contacts: Vec<ContactId>,
    map: ContactBiasMap,
}

impl ContactBiasProblem {
    pub fn new(
        config: PropagationConfig,
        range_rates: &[ContactRangeRate],
        pointing: &[PointingConstraint],
        noise: MeasurementNoise,
    ) -> Result<Self> {
        if range_rates.is_empty() {
            return Err(KernelError::InvalidInput("empty range-rate contact set"));
        }
        if !range_rates.iter().all(|rr| rr.range_rate.is_finite()) {
            return Err(KernelError::InvalidInput(
                "range-rate observations must be finite",
            ));
        }
        let mut index = BTreeMap::new();
        let mut contacts = Vec::new();
        for rr in range_rates {
            if let std::collections::btree_map::Entry::Vacant(entry) = index.entry(rr.contact) {
                entry.insert(contacts.len());
                contacts.push(rr.contact);
            }
        }
        let times: Vec<_> = range_rates.iter().map(|rr| rr.time).collect();
        let stations: Vec<_> = range_rates.iter().map(|rr| rr.station).collect();
        let values: Vec<_> = range_rates.iter().map(|rr| rr.range_rate).collect();
        let geometry = Arc::new(PreparedGeometry::new(&times, &stations)?);
        let mut batches = vec![PreparedBatch::range_rate(geometry, values, noise.sigma_rr)?];
        if !pointing.is_empty() {
            let times: Vec<_> = pointing.iter().map(|p| p.time).collect();
            let stations: Vec<_> = pointing.iter().map(|p| p.station).collect();
            let azimuth: Vec<_> = pointing.iter().map(|p| p.azimuth).collect();
            let elevation: Vec<_> = pointing.iter().map(|p| p.elevation).collect();
            let geometry = Arc::new(PreparedGeometry::new(&times, &stations)?);
            batches.push(PreparedBatch::azimuth_elevation(
                geometry,
                azimuth,
                elevation,
                noise.sigma_az,
                noise.sigma_el,
            )?);
        }
        let inner = ODProblem::new(config, batches)?;
        let n_pointing_rows = pointing
            .len()
            .checked_mul(2)
            .ok_or(KernelError::InvalidInput("pointing row count overflows"))?;
        let rows = range_rates
            .len()
            .checked_add(n_pointing_rows)
            .ok_or(KernelError::InvalidInput("observation row count overflows"))?;
        if rows != inner.rows() {
            return Err(KernelError::InvalidInput(
                "observation rows do not match the bias map",
            ));
        }
        let mut row_contact = Vec::with_capacity(rows);
        let mut row_weight = Vec::with_capacity(rows);
        for rr in range_rates {
            let contact = index
                .get(&rr.contact)
                .copied()
                .ok_or(KernelError::InvalidInput(
                    "range-rate contact index is missing",
                ))?;
            row_contact.push(Some(contact));
            row_weight.push(1.0 / noise.sigma_rr);
        }
        for _ in 0..n_pointing_rows {
            row_contact.push(None);
            row_weight.push(0.0);
        }
        let map = ContactBiasMap::new(&row_contact, &row_weight, contacts.len())?;
        Ok(Self {
            inner,
            contacts,
            map,
        })
    }

    pub fn contacts(&self) -> &[ContactId] {
        &self.contacts
    }

    pub fn inner(&self) -> &ODProblem {
        &self.inner
    }

    pub fn bias_map(&self) -> &ContactBiasMap {
        &self.map
    }

    pub fn evaluate(&self, parameters: &DynVector<f64>) -> Result<Evaluation> {
        self.inner.evaluate(parameters)
    }

    pub fn evaluate_reduced(&self, parameters: &DynVector<f64>) -> Result<Evaluation> {
        let mut evaluation = self.evaluate(parameters)?;
        self.map.project(&mut evaluation)?;
        Ok(evaluation)
    }

    pub fn solve_biases(&self, parameters: &DynVector<f64>) -> Result<Vec<f64>> {
        self.map.solve(&self.evaluate(parameters)?.residual)
    }

    /// Schur-complement state covariance from the reduced problem.
    pub fn covariance(&self, parameters: &DynVector<f64>) -> Result<DynMatrix<f64>> {
        self.covariance_from(&self.evaluate_reduced(parameters)?)
    }

    fn covariance_from(&self, evaluation: &Evaluation) -> Result<DynMatrix<f64>> {
        let rows = evaluation.residual.len();
        let columns = evaluation.jacobian.ncols();
        if evaluation.jacobian.nrows() != rows || rows != self.inner.rows() {
            return Err(KernelError::InvalidInput(
                "evaluation dimensions do not match the problem",
            ));
        }
        let dof = rows
            .checked_sub(columns)
            .and_then(|value| value.checked_sub(self.map.contacts()))
            .filter(|value| *value > 0)
            .ok_or(KernelError::InvalidInput(
                "covariance requires positive residual degrees of freedom",
            ))?;
        let information = &evaluation.jacobian.transpose() * &evaluation.jacobian;
        let inverse = information.inverse().map_err(|_| KernelError::Singular)?;
        let scale = evaluation.rss() / dof as f64;
        if !scale.is_finite() {
            return Err(KernelError::InvalidInput("residual scale is not finite"));
        }
        let covariance = inverse * scale;
        if (0..columns).any(|i| (0..columns).any(|j| !covariance[(i, j)].is_finite())) {
            return Err(KernelError::InvalidInput("covariance is not finite"));
        }
        Ok(covariance)
    }
}

/// Fit result including eliminated biases and the state covariance.
#[derive(Clone, Debug)]
pub struct VarproFit {
    pub parameters: DynVector<f64>,
    pub biases: Vec<f64>,
    pub covariance: DynMatrix<f64>,
    pub iterations: usize,
    pub rss: f64,
}

type CachedReduction = (Vec<f64>, Rc<Evaluation>, Rc<Evaluation>);

struct ReducedProblem<'a> {
    problem: &'a ContactBiasProblem,
    cache: RefCell<Option<CachedReduction>>,
}

impl<'a> ReducedProblem<'a> {
    fn new(problem: &'a ContactBiasProblem) -> Self {
        Self {
            problem,
            cache: RefCell::new(None),
        }
    }

    fn cached(&self, parameters: &DynVector<f64>) -> Result<(Rc<Evaluation>, Rc<Evaluation>)> {
        if let Some((key, raw, reduced)) = self.cache.borrow().as_ref()
            && key == parameters.as_slice()
        {
            return Ok((Rc::clone(raw), Rc::clone(reduced)));
        }
        let raw = Rc::new(self.problem.evaluate(parameters)?);
        let mut reduced = (*raw).clone();
        self.problem.map.project(&mut reduced)?;
        let reduced = Rc::new(reduced);
        *self.cache.borrow_mut() = Some((
            parameters.as_slice().to_vec(),
            Rc::clone(&raw),
            Rc::clone(&reduced),
        ));
        Ok((raw, reduced))
    }

    fn evaluate(&self, parameters: &DynVector<f64>) -> Result<Rc<Evaluation>> {
        Ok(self.cached(parameters)?.1)
    }

    fn raw(&self, parameters: &DynVector<f64>) -> Result<Rc<Evaluation>> {
        Ok(self.cached(parameters)?.0)
    }
}

pub fn fit_varpro(
    problem: ContactBiasProblem,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<VarproFit, FitError> {
    let columns = problem.inner.config().mode().columns();
    let reduced = ReducedProblem::new(&problem);
    let outcome = run_lm(columns, guess, scales, settings, |x| reduced.evaluate(x))?;
    let raw = reduced.raw(&outcome.parameters)?;
    let biases = problem.bias_map().solve(&raw.residual)?;
    let covariance = problem.covariance_from(&outcome.evaluation)?;
    Ok(VarproFit {
        parameters: outcome.parameters,
        biases,
        covariance,
        iterations: outcome.iterations,
        rss: outcome.evaluation.rss(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn fit_bootstrap(
    config: PropagationConfig,
    range_rates: &[ContactRangeRate],
    pointing: &[PointingConstraint],
    noise: MeasurementNoise,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<VarproFit, FitError> {
    fit_varpro(
        ContactBiasProblem::new(config, range_rates, pointing, noise)?,
        guess,
        scales,
        settings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contact_id_parses_bare_and_hyphenated_forms() {
        let bare = ContactId::parse("00112233445566778899aabbccddeeff").unwrap();
        let hyphenated = ContactId::parse("00112233-4455-6677-8899-aabbccddeeff").unwrap();
        assert_eq!(bare, hyphenated);
        assert_eq!(bare.as_u128(), 0x00112233445566778899aabbccddeeff);
        assert_eq!(bare.to_string(), "00112233445566778899aabbccddeeff");
        assert!(ContactId::parse("").is_err());
        assert!(ContactId::parse("00112233-4455-6677-8899-aabbccddeefg").is_err());
    }

    #[test]
    fn projection_removes_weighted_contact_means() {
        // Rows: contact 0; contact 1; contact 1; unmodified pointing row.
        let row_contact = [Some(0), Some(1), Some(1), None];
        let row_weight = [2.0, 1.0, 3.0, 0.0];
        let map = ContactBiasMap::new(&row_contact, &row_weight, 2).unwrap();
        let mut evaluation = Evaluation {
            residual: DynVector::from_slice(&[1.0, 2.0, -1.0, 7.0]),
            jacobian: DynMatrix::from_rows(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0]),
        };
        map.project(&mut evaluation).unwrap();
        // Contact 0: norm 4, sum 2, so b = -0.5 and the row projects to zero.
        assert_eq!(evaluation.residual[0], 0.0);
        assert_eq!(evaluation.jacobian[(0, 0)], 0.0);
        assert_eq!(evaluation.jacobian[(0, 1)], 0.0);
        // Contact 1: norm 10, sum -1, so b = 0.1.
        assert!((evaluation.residual[1] - 2.1).abs() < 1e-14);
        assert!((evaluation.residual[2] + 0.7).abs() < 1e-14);
        assert_eq!(evaluation.residual[3], 7.0);
        assert_eq!(evaluation.jacobian[(3, 0)], 8.0);
        // Every bias row's projected residual has zero weighted sum.
        assert!((2.0 * evaluation.residual[0]).abs() < 1e-14);
        assert!((evaluation.residual[1] + 3.0 * evaluation.residual[2]).abs() < 1e-13);
        let biases = map
            .solve(&DynVector::from_slice(&[1.0, 2.0, -1.0, 7.0]))
            .unwrap();
        assert!((biases[0] - -0.5).abs() < 1e-14);
        assert!((biases[1] - 0.1).abs() < 1e-14);
    }
}
