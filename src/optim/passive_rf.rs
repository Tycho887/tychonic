//! Prior-inclusive RF batch fit with an analytically marginalized OU offset.
use super::{FitError, InputSource, LmOutcome, run_lm_with_drag};
use crate::kernels::*;
use numeris::{
    DynMatrix, DynVector, Matrix,
    dynmatrix::DynCholesky,
    optim::{LmSettings, OptimError},
};

/// One prepared schedule. Doppler values are carrier-relative Hz; phases are
/// unwrapped radians. None selects a Doppler-only baseline.
#[derive(Debug)]
pub struct PassiveRfData<'g> {
    pub(crate) geometry: GeometrySource<'g>,
    pub(crate) interferometer: Interferometer,
    pub(crate) doppler_hz: Vec<f64>,
    pub(crate) phase_rad: Option<Vec<f64>>,
    pub(crate) sigma_doppler_hz: f64,
    pub(crate) sigma_phase_rad: f64,
}

impl<'g> PassiveRfData<'g> {
    pub fn new(
        geometry: impl Into<GeometrySource<'g>>,
        interferometer: Interferometer,
        doppler_hz: Vec<f64>,
        phase_rad: Option<Vec<f64>>,
        sigma_doppler_hz: f64,
        sigma_phase_rad: f64,
    ) -> Result<Self> {
        let data = Self {
            geometry: geometry.into(),
            interferometer,
            doppler_hz,
            phase_rad,
            sigma_doppler_hz,
            sigma_phase_rad,
        };
        data.validate()?;
        Ok(data)
    }
    pub fn geometry(&self) -> &PreparedGeometry {
        &self.geometry
    }
    pub fn interferometer(&self) -> Interferometer {
        self.interferometer
    }
    pub fn doppler_hz(&self) -> &[f64] {
        &self.doppler_hz
    }
    pub fn phase_rad(&self) -> Option<&[f64]> {
        self.phase_rad.as_deref()
    }
    pub fn sigma_doppler_hz(&self) -> f64 {
        self.sigma_doppler_hz
    }
    pub fn sigma_phase_rad(&self) -> f64 {
        self.sigma_phase_rad
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.geometry.len();
        if n == 0
            || self.doppler_hz.len() != n
            || self
                .phase_rad
                .as_ref()
                .is_some_and(|phase| phase.len() != n)
        {
            return Err(KernelError::InvalidInput("RF data dimensions"));
        }
        if !self.doppler_hz.iter().all(|x| x.is_finite())
            || self
                .phase_rad
                .as_ref()
                .is_some_and(|phase| !phase.iter().all(|x| x.is_finite()))
        {
            return Err(KernelError::InvalidInput("RF observations must be finite"));
        }
        if !valid_sigma(self.sigma_doppler_hz) || !valid_sigma(self.sigma_phase_rad) {
            return Err(KernelError::InvalidInput(
                "RF noise sigmas must be finite, positive and representable",
            ));
        }
        Ok(())
    }
}

/// Diagonal Gaussian prior, ordered orbital parameters, frequency offset, phase
/// mean. Optional drag remains orbital parameter 6. All units are physical.
#[derive(Debug)]
pub struct RfPrior {
    pub(crate) mean: DynVector<f64>,
    pub(crate) sigma: DynVector<f64>,
}

impl RfPrior {
    pub fn new(mean: DynVector<f64>, sigma: DynVector<f64>) -> Result<Self> {
        let prior = Self { mean, sigma };
        prior.validate(prior.mean.len())?;
        Ok(prior)
    }
    pub fn mean(&self) -> &DynVector<f64> {
        &self.mean
    }
    pub fn sigma(&self) -> &DynVector<f64> {
        &self.sigma
    }

    pub fn validate(&self, columns: usize) -> Result<()> {
        if columns == 0
            || self.mean.len() != columns
            || self.sigma.len() != columns
            || !self.mean.as_slice().iter().all(|v| v.is_finite())
            || !self
                .sigma
                .as_slice()
                .iter()
                .all(|v| v.is_finite() && *v > 0.0 && v.powi(2).is_finite() && v.powi(2) > 0.0)
        {
            return Err(KernelError::InvalidInput("RF prior dimensions/values"));
        }
        Ok(())
    }
}

/// f1-f2 formed from independent antenna frequency errors. Antenna 1's noise
/// standard deviation comes from PassiveRfData::sigma_doppler_hz.
#[derive(Debug)]
pub struct DifferentialDopplerData {
    pub difference_hz: Vec<f64>,
    pub sigma_antenna2_hz: f64,
}
impl DifferentialDopplerData {
    pub fn validate(&self, data: &PassiveRfData) -> Result<()> {
        if data.phase_rad.is_none()
            || self.difference_hz.len() != data.geometry.len()
            || !self.difference_hz.iter().all(|x| x.is_finite())
            || !valid_sigma(self.sigma_antenna2_hz)
        {
            return Err(KernelError::InvalidInput(
                "differential RF data requires phase and matching finite values/noise",
            ));
        }
        Ok(())
    }
    /// Ordering f1, phase, f1-f2. Shared f1 noise gives covariance sigma1².
    pub fn covariance(&self, data: &PassiveRfData) -> Matrix<f64, 3, 3> {
        let a = data.sigma_doppler_hz.powi(2);
        Matrix::new([
            [a, 0.0, a],
            [0.0, data.sigma_phase_rad.powi(2), 0.0],
            [a, 0.0, a + self.sigma_antenna2_hz.powi(2)],
        ])
    }
}
#[derive(Debug, Default)]
pub struct RfExtensions {
    /// Independent 1σ ENU baseline errors, estimated as constant parameters.
    pub baseline_sigma_m: Option<V3>,
    pub differential_doppler: Option<DifferentialDopplerData>,
    /// Separate Doppler baseline anchored at the same antenna 1 as phase.
    /// None uses the phase baseline, including its calibration correction.
    pub differential_interferometer: Option<Interferometer>,
    /// Independent constant ENU calibration of the separate Doppler baseline.
    pub differential_baseline_sigma_m: Option<V3>,
}
fn valid_sigma(v: f64) -> bool {
    v.is_finite() && v > 0.0 && v.powi(2).is_finite() && v.powi(2) > 0.0
}
pub(crate) fn validate_baseline_sigma(sigma: &V3) -> Result<()> {
    if !sigma.as_slice().iter().all(|v| valid_sigma(*v)) {
        return Err(KernelError::InvalidInput(
            "baseline sigma must be finite and positive",
        ));
    }
    Ok(())
}

/// Shared input validation without constructing the dense batch covariance.
pub(crate) fn validate_rf_setup(
    config: &PropagationConfig,
    data: &PassiveRfData,
    prior: &RfPrior,
    extensions: &RfExtensions,
) -> Result<()> {
    data.validate()?;
    prior.validate(config.mode().columns() + 2)?;
    if let Some(sigma) = &extensions.baseline_sigma_m {
        validate_baseline_sigma(sigma)?;
    }
    if let Some(sigma) = &extensions.differential_baseline_sigma_m {
        validate_baseline_sigma(sigma)?;
        if extensions.differential_interferometer.is_none() {
            return Err(KernelError::InvalidInput(
                "separate baseline calibration requires separate Doppler geometry",
            ));
        }
    }
    if extensions
        .differential_interferometer
        .is_some_and(|int| int.carrier_hz() != data.interferometer.carrier_hz())
    {
        return Err(KernelError::InvalidInput(
            "phase and differential carriers must match",
        ));
    }
    if let Some(dd) = &extensions.differential_doppler {
        dd.validate(data)?;
    }
    Ok(())
}

#[derive(Debug)]
pub struct PassiveRfProblem<'g> {
    config: PropagationConfig,
    data: InputSource<'g, PassiveRfData<'g>>,
    prior: InputSource<'g, RfPrior>,
    ou: OuPhaseNoise,
    extensions: InputSource<'g, RfExtensions>,
    measurement_factor: DynCholesky<f64>,
}
impl<'g> PassiveRfProblem<'g> {
    pub fn new(
        config: PropagationConfig,
        data: impl Into<InputSource<'g, PassiveRfData<'g>>>,
        prior: impl Into<InputSource<'g, RfPrior>>,
        ou: OuPhaseNoise,
    ) -> Result<Self> {
        Self::new_extended(config, data, prior, ou, RfExtensions::default())
    }
    pub fn new_extended(
        config: PropagationConfig,
        data: impl Into<InputSource<'g, PassiveRfData<'g>>>,
        prior: impl Into<InputSource<'g, RfPrior>>,
        ou: OuPhaseNoise,
        extensions: impl Into<InputSource<'g, RfExtensions>>,
    ) -> Result<Self> {
        let data = data.into();
        let prior = prior.into();
        let extensions = extensions.into();
        validate_rf_setup(&config, &data, &prior, &extensions)?;
        let measurement_factor = rf_covariance(&data, &extensions, ou)?
            .cholesky()
            .map_err(|_| KernelError::Singular)?;
        let problem = Self {
            config,
            data,
            prior,
            ou,
            extensions,
            measurement_factor,
        };
        Ok(problem)
    }
    pub fn config(&self) -> &PropagationConfig {
        &self.config
    }
    pub fn data(&self) -> &PassiveRfData<'g> {
        &self.data
    }
    pub fn prior(&self) -> &RfPrior {
        &self.prior
    }
    pub fn extensions(&self) -> &RfExtensions {
        &self.extensions
    }
    pub fn columns(&self) -> usize {
        self.differential_baseline_column()
            + 3 * usize::from(self.extensions.differential_baseline_sigma_m.is_some())
    }
    fn differential_baseline_column(&self) -> usize {
        self.config.mode().columns()
            + 2
            + 3 * usize::from(self.extensions.baseline_sigma_m.is_some())
    }
    pub fn initial_parameters(&self) -> DynVector<f64> {
        let mut x = DynVector::zeros(self.columns());
        for k in 0..self.prior.mean.len() {
            x[k] = self.prior.mean[k];
        }
        x
    }
    pub fn measurement_covariance(&self) -> Result<DynMatrix<f64>> {
        rf_covariance(&self.data, &self.extensions, self.ou)
    }
    fn interferometer_at(&self, x: &DynVector<f64>) -> Result<Interferometer> {
        let p = self.config.mode().columns();
        if self.extensions.baseline_sigma_m.is_some() {
            self.data
                .interferometer
                .corrected(&v3(x[p + 2], x[p + 3], x[p + 4]))
        } else {
            Ok(self.data.interferometer)
        }
    }
    pub fn evaluate(&self, x: &DynVector<f64>) -> Result<Evaluation> {
        let mut output = Evaluation::default();
        self.evaluate_into(x, &mut output)?;
        Ok(output)
    }
    pub fn evaluate_into(&self, x: &DynVector<f64>, e: &mut Evaluation) -> Result<()> {
        self.check_parameters(x)?;
        let p = self.config.mode().columns();
        let sens = self.config.sample(
            &DynVector::from_slice(&x.as_slice()[..p]),
            self.data.geometry.timestamps(),
        )?;
        let rows = self.measurement_factor.l().nrows();
        let columns = self.columns();
        e.resize(
            rows.checked_add(columns)
                .ok_or(KernelError::InvalidInput("RF output dimensions overflow"))?,
            columns,
        )?;
        e.residual.as_mut_slice().fill(0.0);
        e.jacobian.as_mut_slice().fill(0.0);
        let interferometer = self.interferometer_at(x)?;
        let differential = self.differential_at(x, interferometer)?;
        for ((i, geometry), sample) in self
            .data
            .geometry
            .iter_rows()
            .enumerate()
            .zip(sens.iter_rows())
        {
            let pred = interferometer.evaluate(sample.state, geometry, x[p])?;
            self.write_doppler(i, &sample, &pred, e);
            self.write_phase(i, x, &sample, geometry, &pred, e);
            self.write_differential(i, &sample, geometry, differential, e)?;
        }
        whiten(e, self.measurement_factor.l(), rows, columns);
        self.write_priors(x, e, rows);
        e.check_finite()
    }
    fn check_parameters(&self, x: &DynVector<f64>) -> Result<()> {
        if x.len() != self.columns() || !x.as_slice().iter().all(|v| v.is_finite()) {
            return Err(KernelError::InvalidInput("RF parameter dimensions/values"));
        }
        Ok(())
    }
    fn differential_at(&self, x: &DynVector<f64>, local: Interferometer) -> Result<Interferometer> {
        let mut interferometer = self.extensions.differential_interferometer.unwrap_or(local);
        if self.extensions.differential_baseline_sigma_m.is_some() {
            let k = self.differential_baseline_column();
            interferometer = interferometer.corrected(&v3(x[k], x[k + 1], x[k + 2]))?;
        }
        Ok(interferometer)
    }
    fn write_doppler(
        &self,
        i: usize,
        sample: &SensitivityRow<'_>,
        pred: &RfPrediction,
        e: &mut Evaluation,
    ) {
        let h = pred.partial * sample.stm;
        for k in 0..6 {
            e.jacobian[(i, k)] = h[(0, k)];
        }
        if let Some(drag) = sample.state_dcd {
            e.jacobian[(i, 6)] = (pred.partial * drag)[0];
        }
        e.residual[i] = pred.doppler_hz - self.data.doppler_hz[i];
        e.jacobian[(i, self.config.mode().columns())] = 1.0;
    }
    fn write_phase(
        &self,
        i: usize,
        x: &DynVector<f64>,
        sample: &SensitivityRow<'_>,
        geometry: &StationGeometry,
        pred: &RfPrediction,
        e: &mut Evaluation,
    ) {
        let Some(observed) = &self.data.phase_rad else {
            return;
        };
        let row = self.data.geometry.len() + i;
        let p = self.config.mode().columns();
        let h = pred.partial * sample.stm;
        for k in 0..6 {
            e.jacobian[(row, k)] = h[(1, k)];
        }
        if let Some(drag) = sample.state_dcd {
            e.jacobian[(row, 6)] = (pred.partial * drag)[1];
        }
        e.residual[row] = pred.phase_rad + x[p + 1] - observed[i];
        e.jacobian[(row, p)] = pred.phase_frequency_bias_partial;
        e.jacobian[(row, p + 1)] = 1.0;
        if self.extensions.baseline_sigma_m.is_some() {
            let h = pred.phase_baseline_partial_gcrf * geometry.rotation().matrix();
            for k in 0..3 {
                e.jacobian[(row, p + 2 + k)] = h[(0, k)];
            }
        }
    }
    fn differential_baseline_column_for_model(&self) -> Option<usize> {
        if self.extensions.differential_baseline_sigma_m.is_some() {
            return Some(self.differential_baseline_column());
        }
        if self.extensions.differential_interferometer.is_none()
            && self.extensions.baseline_sigma_m.is_some()
        {
            return Some(self.config.mode().columns() + 2);
        }
        None
    }
    fn write_differential(
        &self,
        i: usize,
        sample: &SensitivityRow<'_>,
        geometry: &StationGeometry,
        interferometer: Interferometer,
        e: &mut Evaluation,
    ) -> Result<()> {
        let Some(observed) = &self.extensions.differential_doppler else {
            return Ok(());
        };
        let row = 2 * self.data.geometry.len() + i;
        let pred = interferometer.differential(sample.state, geometry, sample.time)?;
        let h = pred.partial * sample.stm;
        for k in 0..6 {
            e.jacobian[(row, k)] = h[(0, k)];
        }
        if let Some(drag) = sample.state_dcd {
            e.jacobian[(row, 6)] = (pred.partial * drag)[0];
        }
        e.residual[row] = pred.difference_hz - observed.difference_hz[i];
        if let Some(column) = self.differential_baseline_column_for_model() {
            for k in 0..3 {
                e.jacobian[(row, column + k)] = pred.baseline_partial[(0, k)];
            }
        }
        Ok(())
    }
    fn write_priors(&self, x: &DynVector<f64>, e: &mut Evaluation, rows: usize) {
        let p = self.config.mode().columns();
        for k in 0..p + 2 {
            e.residual[rows + k] = (x[k] - self.prior.mean[k]) / self.prior.sigma[k];
            e.jacobian[(rows + k, k)] = 1.0 / self.prior.sigma[k];
        }
        for (column, sigma) in [
            (p + 2, self.extensions.baseline_sigma_m.as_ref()),
            (
                self.differential_baseline_column(),
                self.extensions.differential_baseline_sigma_m.as_ref(),
            ),
        ] {
            if let Some(sigma) = sigma {
                for k in 0..3 {
                    e.residual[rows + column + k] = x[column + k] / sigma[k];
                    e.jacobian[(rows + column + k, column + k)] = 1.0 / sigma[k];
                }
            }
        }
    }
    /// Conditional OU drift mean given fitted orbital/offset/baseline parameters.
    pub fn conditional_drift(&self, x: &DynVector<f64>) -> Result<Vec<f64>> {
        let Some(observed) = &self.data.phase_rad else {
            return Ok(Vec::new());
        };
        if x.len() != self.columns() || !x.as_slice().iter().all(|v| v.is_finite()) {
            return Err(KernelError::InvalidInput("RF parameter dimensions/values"));
        }
        let p = self.config.mode().columns();
        let times = self.data.geometry.timestamps();
        let sens = self
            .config
            .sample(&DynVector::from_slice(&x.as_slice()[..p]), times)?;
        let interferometer = self.interferometer_at(x)?;
        let mut residual = DynVector::zeros(times.len());
        for (i, geometry) in self.data.geometry.iter_rows().enumerate() {
            let pred = interferometer.evaluate(&sens.states[i], geometry, x[p])?;
            residual[i] = observed[i] - pred.phase_rad - x[p + 1];
        }
        let c = self.ou.covariance(times, self.data.sigma_phase_rad)?;
        let weighted = c
            .cholesky()
            .map_err(|_| KernelError::Singular)?
            .solve(&residual);
        Ok((0..times.len())
            .map(|i| {
                (0..times.len())
                    .map(|j| {
                        (c[(i, j)]
                            - if i == j {
                                self.data.sigma_phase_rad.powi(2)
                            } else {
                                0.0
                            })
                            * weighted[j]
                    })
                    .sum()
            })
            .collect())
    }
}

#[derive(Debug)]
pub struct PassiveRfFit {
    pub parameters: DynVector<f64>,
    pub covariance: DynMatrix<f64>,
    pub drift_rad: Vec<f64>,
    pub rss: f64,
    pub iterations: usize,
    /// Numeris currently checks step/cost tolerances only after accepted steps.
    /// True means its iteration limit was followed by an independent undamped
    /// QR step satisfying the supplied step or predicted-cost tolerance.
    pub linearized_tolerance: bool,
}

pub fn fit_passive_rf(
    problem: &PassiveRfProblem,
    guess: &DynVector<f64>,
    settings: &LmSettings<f64>,
) -> std::result::Result<PassiveRfFit, FitError> {
    let p = problem.config.mode().columns();
    let columns = problem.columns();
    let mut scales = vec![100.0, 100.0, 100.0, 0.1, 0.1, 0.1];
    if p == 7 {
        scales.push(0.01);
    }
    scales.extend([1.0, 1.0]);
    if let Some(sigma) = &problem.extensions.baseline_sigma_m {
        scales.extend(sigma.as_slice());
    }
    if let Some(sigma) = &problem.extensions.differential_baseline_sigma_m {
        scales.extend(sigma.as_slice());
    }
    let mut best_parameters = DynVector::zeros(columns);
    let mut best_rss = f64::INFINITY;
    let outcome = run_lm_with_drag(
        columns,
        guess,
        Some(&scales),
        settings,
        (p == 7).then_some(6),
        |x, output| {
            problem.evaluate_into(x, output)?;
            let rss = output.rss();
            if rss < best_rss {
                best_parameters.as_mut_slice().copy_from_slice(x.as_slice());
                best_rss = rss;
            }
            Ok(())
        },
    );
    let (outcome, linearized_tolerance) = match outcome {
        Ok(outcome) => (outcome, false),
        Err(FitError::Solver(OptimError::MaxIterations)) => {
            let evaluation = problem.evaluate(&best_parameters)?;
            verify_linearized_tolerance(&evaluation, &best_parameters, guess, &scales, settings)?;
            (
                LmOutcome {
                    parameters: best_parameters,
                    evaluation,
                    iterations: settings.max_iter,
                },
                true,
            )
        }
        Err(error) => return Err(error),
    };
    // Form information in scaled coordinates to protect mixed-unit pivots.
    let mut j = DynMatrix::zeros(outcome.evaluation.jacobian.nrows(), columns);
    super::lm::scale_jacobian(&outcome.evaluation, &scales, &mut j);
    let mut covariance = (j.transpose() * j)
        .cholesky()
        .map_err(|_| KernelError::Singular)?
        .inverse();
    for i in 0..columns {
        for k in 0..columns {
            covariance[(i, k)] *= scales[i] * scales[k];
        }
    }
    let drift_rad = problem.conditional_drift(&outcome.parameters)?;
    Ok(PassiveRfFit {
        parameters: outcome.parameters,
        covariance,
        drift_rad,
        rss: outcome.evaluation.rss(),
        iterations: outcome.iterations,
        linearized_tolerance,
    })
}

fn verify_linearized_tolerance(
    evaluation: &Evaluation,
    parameters: &DynVector<f64>,
    guess: &DynVector<f64>,
    scales: &[f64],
    settings: &LmSettings<f64>,
) -> std::result::Result<(), FitError> {
    let mut j = DynMatrix::zeros(evaluation.jacobian.nrows(), scales.len());
    super::lm::scale_jacobian(evaluation, scales, &mut j);
    let negative = DynVector::from_slice(
        &evaluation
            .residual
            .as_slice()
            .iter()
            .map(|v| -v)
            .collect::<Vec<_>>(),
    );
    let step = j.qr().map_err(|_| KernelError::Singular)?.solve(&negative);
    let correction_norm = (0..scales.len())
        .map(|k| ((parameters[k] - guess[k]) / scales[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    let predicted_gain = 0.5
        * (0..j.nrows())
            .map(|row| {
                (0..scales.len())
                    .map(|k| j[(row, k)] * step[k])
                    .sum::<f64>()
                    .powi(2)
            })
            .sum::<f64>();
    if !step.as_slice().iter().all(|v| v.is_finite())
        || !correction_norm.is_finite()
        || !predicted_gain.is_finite()
        || !(step.norm() <= settings.x_tol * (1.0 + correction_norm)
            || predicted_gain <= settings.f_tol * (1.0 + 0.5 * evaluation.rss()))
    {
        return Err(FitError::Solver(OptimError::MaxIterations));
    }
    Ok(())
}

fn rf_covariance(
    data: &PassiveRfData,
    extensions: &RfExtensions,
    ou: OuPhaseNoise,
) -> Result<DynMatrix<f64>> {
    let n = data.geometry.len();
    let phase = data.phase_rad.is_some();
    let differential = extensions.differential_doppler.as_ref();
    let m = 1 + usize::from(phase) + usize::from(differential.is_some());
    let rows = m.checked_mul(n).ok_or(KernelError::InvalidInput(
        "RF covariance dimensions overflow",
    ))?;
    rows.checked_mul(rows).ok_or(KernelError::InvalidInput(
        "RF covariance dimensions overflow",
    ))?;
    let mut c = DynMatrix::zeros(rows, rows);
    for i in 0..n {
        c[(i, i)] = data.sigma_doppler_hz.powi(2);
    }
    if phase {
        let phase_c = ou.covariance(data.geometry.timestamps(), data.sigma_phase_rad)?;
        for i in 0..n {
            for j in 0..n {
                c[(n + i, n + j)] = phase_c[(i, j)];
            }
        }
    }
    if let Some(dd) = differential {
        for i in 0..n {
            c[(2 * n + i, 2 * n + i)] =
                data.sigma_doppler_hz.powi(2) + dd.sigma_antenna2_hz.powi(2);
            c[(i, 2 * n + i)] = data.sigma_doppler_hz.powi(2);
            c[(2 * n + i, i)] = data.sigma_doppler_hz.powi(2);
        }
    }
    Ok(c)
}

fn whiten(e: &mut Evaluation, lower: &DynMatrix<f64>, rows: usize, columns: usize) {
    for i in 0..rows {
        for j in 0..i {
            let weight = lower[(i, j)];
            if weight == 0.0 {
                continue;
            }
            e.residual[i] -= weight * e.residual[j];
            for k in 0..columns {
                e.jacobian[(i, k)] -= weight * e.jacobian[(j, k)];
            }
        }
        e.residual[i] /= lower[(i, i)];
        for k in 0..columns {
            e.jacobian[(i, k)] /= lower[(i, i)];
        }
    }
}
