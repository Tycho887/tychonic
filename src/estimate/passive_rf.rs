//! Bias-aware RF EKF. Layout: orbit (6, or 7 including drag), frequency
//! offset (Hz), unknown hardware phase mean (rad), stationary OU drift (rad).
use super::{EkfResult, SmoothedOrbit, UpdateOptions, UpdateResult, ekf::check_covariance};
use crate::{
    kernels::*,
    optim::{
        DifferentialDopplerData, PassiveRfData, RfExtensions, RfPrior,
        passive_rf::validate_baseline_sigma,
    },
};
use numeris::{
    Matrix, Vector,
    estimate::{Ekf, EkfStep, rts_smooth},
};
use satkit::Instant;

pub struct PassiveRfEkf<const N: usize> {
    config: PropagationConfig,
    ou: OuPhaseNoise,
    epoch: Instant,
    filter: Ekf<f64, N, 1>,
    timestamps: Vec<Instant>,
    steps: Vec<EkfStep<f64, N>>,
    has_baseline: bool,
    has_differential_baseline: bool,
}

impl<const N: usize> PassiveRfEkf<N> {
    pub fn new(config: PropagationConfig, ou: OuPhaseNoise, prior: &RfPrior) -> EkfResult<Self> {
        Self::new_extended(config, ou, prior, None)
    }
    /// Adds three constant ENU baseline correction states after the OU state.
    /// N=12 (fixed drag) or 13 (estimated drag) with baseline calibration.
    pub fn new_extended(
        config: PropagationConfig,
        ou: OuPhaseNoise,
        prior: &RfPrior,
        baseline_sigma_m: Option<V3>,
    ) -> EkfResult<Self> {
        Self::new_with_calibration(config, ou, prior, baseline_sigma_m, None)
    }
    /// Independent local phase and remote Doppler baseline corrections follow
    /// the OU state. N=15/16 with both calibrations (fixed/estimated drag).
    pub fn new_with_calibration(
        config: PropagationConfig,
        ou: OuPhaseNoise,
        prior: &RfPrior,
        baseline_sigma_m: Option<V3>,
        differential_baseline_sigma_m: Option<V3>,
    ) -> EkfResult<Self> {
        let p = config.mode().columns();
        if N != p
            + 3
            + 3 * usize::from(baseline_sigma_m.is_some())
            + 3 * usize::from(differential_baseline_sigma_m.is_some())
        {
            return Err(KernelError::InvalidInput(
                "RF EKF dimension must match orbit and calibration modes (9/10, 12/13 or 15/16)",
            )
            .into());
        }
        prior.validate(p + 2)?;
        let mut x = Vector::zeros();
        let mut covariance = Matrix::zeros();
        for i in 0..p + 2 {
            x[i] = prior.mean[i];
            covariance[(i, i)] = prior.sigma[i].powi(2);
        }
        covariance[(p + 2, p + 2)] = ou.stationary_sigma_rad().powi(2);
        if let Some(sigma) = &baseline_sigma_m {
            validate_baseline_sigma(sigma)?;
            for k in 0..3 {
                covariance[(p + 3 + k, p + 3 + k)] = sigma[k].powi(2);
            }
        }
        if let Some(sigma) = &differential_baseline_sigma_m {
            validate_baseline_sigma(sigma)?;
            let column = p + 3 + 3 * usize::from(baseline_sigma_m.is_some());
            for k in 0..3 {
                covariance[(column + k, column + k)] = sigma[k].powi(2);
            }
        }
        validate_state(&x, p)?;
        check_covariance(&covariance, true)?;
        let epoch = config.reference_epoch();
        Ok(Self {
            config,
            ou,
            epoch,
            filter: Ekf::new(x, covariance),
            timestamps: vec![epoch],
            steps: vec![record(x, covariance, Matrix::eye())],
            has_baseline: baseline_sigma_m.is_some(),
            has_differential_baseline: differential_baseline_sigma_m.is_some(),
        })
    }
    pub fn state(&self) -> &Vector<f64, N> {
        self.filter.state()
    }
    pub fn covariance(&self) -> &Matrix<f64, N, N> {
        self.filter.covariance()
    }
    pub fn epoch(&self) -> Instant {
        self.epoch
    }
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }
    pub fn steps(&self) -> &[EkfStep<f64, N>] {
        &self.steps
    }
    pub fn smooth(&self) -> EkfResult<SmoothedOrbit<N>> {
        Ok(rts_smooth(&self.steps)?)
    }

    /// Deterministic orbital dynamics and exact OU discretization. Optional
    /// orbital process noise is endpoint GCRF covariance, added once per epoch.
    pub fn predict_to(
        &mut self,
        time: Instant,
        orbital_q: Option<&Matrix<f64, 6, 6>>,
    ) -> EkfResult<()> {
        if time < self.epoch {
            return Err(KernelError::InvalidInput("RF epochs must be chronological").into());
        }
        if let Some(q) = orbital_q {
            check_covariance(q, false)?;
        }
        if time == self.epoch {
            if orbital_q.is_some_and(|q| q.as_slice().iter().any(|x| *x != 0.0)) {
                return Err(
                    KernelError::InvalidInput("same epoch requires zero process noise").into(),
                );
            }
            return Ok(());
        }
        let p = self.config.mode().columns();
        let state = V6::from_slice(&self.filter.x.as_slice()[..6]);
        let mut f = Matrix::eye();
        let propagated = match self.config.mode() {
            ParameterMode::StateOnly { fixed_drag } => {
                let (x, phi) = propagate_with_stm(
                    &state,
                    &self.epoch,
                    &time,
                    self.config.settings(),
                    fixed_drag,
                )?;
                f.set_block(0, 0, &phi);
                x
            }
            ParameterMode::StateAndDrag { delta_cd } => {
                let s = compute_sensitivities(
                    &state,
                    &self.epoch,
                    &[time],
                    self.config.settings(),
                    self.filter.x[6],
                    delta_cd,
                )?;
                f.set_block(0, 0, &s.augmented_transition(0)?);
                s.states[0]
            }
        };
        let (a, variance) = self.ou.transition((time - self.epoch).as_seconds())?;
        f[(p + 2, p + 2)] = a;
        let mut predicted = self.filter.x;
        predicted.set_block(0, 0, &propagated);
        predicted[p + 2] *= a;
        let mut q = Matrix::zeros();
        q[(p + 2, p + 2)] = variance;
        if let Some(orbital_q) = orbital_q {
            q.set_block(0, 0, orbital_q);
        }
        let mut candidate = Ekf::<f64, N, 1>::new(self.filter.x, self.filter.p);
        candidate.predict(|_| predicted, |_| f, Some(&q));
        validate_state(&candidate.x, p)?;
        check_covariance(&candidate.p, true)?;
        self.steps.push(record(candidate.x, candidate.p, f));
        self.timestamps.push(time);
        self.filter = candidate;
        self.epoch = time;
        Ok(())
    }

    /// Doppler (M=1), Doppler/phase (M=2), or Doppler/phase/local differential
    /// (M=3). Supply the full measurement covariance, including shared noise.
    pub fn update<const M: usize>(
        &mut self,
        observed: &Vector<f64, M>,
        geometry: &StationGeometry,
        interferometer: Interferometer,
        covariance: &Matrix<f64, M, M>,
        options: UpdateOptions,
    ) -> EkfResult<UpdateResult<M>> {
        self.update_with_differential_baseline(
            observed,
            geometry,
            interferometer,
            None,
            covariance,
            options,
        )
    }
    /// As update, with an optional separate Doppler baseline anchored at the
    /// same antenna 1. Its calibration states are independent of local phase.
    pub fn update_with_differential_baseline<const M: usize>(
        &mut self,
        observed: &Vector<f64, M>,
        geometry: &StationGeometry,
        interferometer: Interferometer,
        differential_interferometer: Option<Interferometer>,
        covariance: &Matrix<f64, M, M>,
        options: UpdateOptions,
    ) -> EkfResult<UpdateResult<M>> {
        if !(1..=3).contains(&M)
            || !observed.as_slice().iter().all(|x| x.is_finite())
            || options.gate.is_some_and(|g| !g.is_finite() || g < 0.0)
        {
            return Err(KernelError::InvalidInput("RF sensor dimension, reading or gate").into());
        }
        check_covariance(covariance, true)?;
        if differential_interferometer
            .is_some_and(|int| int.carrier_hz() != interferometer.carrier_hz())
            || (M == 3 && self.has_differential_baseline && differential_interferometer.is_none())
        {
            return Err(KernelError::InvalidInput(
                "separate differential geometry/carrier mismatch",
            )
            .into());
        }
        let p = self.config.mode().columns();
        let interferometer = if self.has_baseline {
            interferometer.corrected(&v3(
                self.filter.x[p + 3],
                self.filter.x[p + 4],
                self.filter.x[p + 5],
            ))?
        } else {
            interferometer
        };
        let pred = interferometer.evaluate(
            &V6::from_slice(&self.filter.x.as_slice()[..6]),
            geometry,
            self.filter.x[p],
        )?;
        let mut h = Matrix::<f64, M, N>::zeros();
        let mut innovation = Vector::zeros();
        innovation[0] = observed[0] - pred.doppler_hz;
        for j in 0..6 {
            h[(0, j)] = pred.partial[(0, j)];
        }
        h[(0, p)] = 1.0;
        if M >= 2 {
            innovation[1] =
                observed[1] - pred.phase_rad - self.filter.x[p + 1] - self.filter.x[p + 2];
            for j in 0..6 {
                h[(1, j)] = pred.partial[(1, j)];
            }
            h[(1, p)] = pred.phase_frequency_bias_partial;
            h[(1, p + 1)] = 1.0;
            h[(1, p + 2)] = 1.0;
            if self.has_baseline {
                let partial = pred.phase_baseline_partial_gcrf * geometry.rotation().matrix();
                for k in 0..3 {
                    h[(1, p + 3 + k)] = partial[(0, k)];
                }
            }
        }
        if M == 3 {
            let mut differential_int = differential_interferometer.unwrap_or(interferometer);
            let column = if self.has_differential_baseline {
                Some(p + 3 + 3 * usize::from(self.has_baseline))
            } else if self.has_baseline && differential_interferometer.is_none() {
                Some(p + 3)
            } else {
                None
            };
            if self.has_differential_baseline {
                let k = column.unwrap();
                differential_int = differential_int.corrected(&v3(
                    self.filter.x[k],
                    self.filter.x[k + 1],
                    self.filter.x[k + 2],
                ))?;
            }
            let pred = differential_int.differential(
                &V6::from_slice(&self.filter.x.as_slice()[..6]),
                geometry,
                self.epoch,
            )?;
            innovation[2] = observed[2] - pred.difference_hz;
            for k in 0..6 {
                h[(2, k)] = pred.partial[(0, k)];
            }
            if let Some(column) = column {
                for k in 0..3 {
                    h[(2, column + k)] = pred.baseline_partial[(0, k)];
                }
            }
        }
        let mut candidate = Ekf::<f64, N, M>::new(self.filter.x, self.filter.p);
        let nis = candidate.update(&innovation, |_| Vector::zeros(), |_| h, covariance)?;
        if !nis.is_finite() {
            return Err(KernelError::InvalidInput("RF innovation exceeds numeric range").into());
        }
        let accepted = options.gate.is_none_or(|g| nis <= g);
        if accepted {
            validate_state(&candidate.x, p)?;
            check_covariance(&candidate.p, true)?;
            self.filter.x = candidate.x;
            self.filter.p = candidate.p;
            let last = self.steps.last_mut().unwrap();
            last.x_updated = candidate.x;
            last.p_updated = candidate.p;
        }
        Ok(UpdateResult {
            innovation,
            nis,
            accepted,
        })
    }

    /// Sort exact epochs stably. On a failed reading roll back its prediction;
    /// earlier successful readings remain committed.
    pub fn filter_data(
        &mut self,
        data: &PassiveRfData,
        options: UpdateOptions,
    ) -> EkfResult<Vec<(f64, bool)>> {
        self.filter_data_extended(data, None, options)
    }
    pub fn filter_data_extended(
        &mut self,
        data: &PassiveRfData,
        differential: Option<&DifferentialDopplerData>,
        options: UpdateOptions,
    ) -> EkfResult<Vec<(f64, bool)>> {
        self.filter_data_with_extensions(
            data,
            &RfExtensions {
                differential_doppler: differential.cloned(),
                ..RfExtensions::default()
            },
            options,
        )
    }
    pub fn filter_data_with_extensions(
        &mut self,
        data: &PassiveRfData,
        extensions: &RfExtensions,
        options: UpdateOptions,
    ) -> EkfResult<Vec<(f64, bool)>> {
        data.validate()?;
        let differential = extensions.differential_doppler.as_ref();
        if let Some(dd) = differential {
            dd.validate(data)?;
        }
        let mut order: Vec<_> = (0..data.geometry.len()).collect();
        order.sort_by_key(|&i| data.geometry.timestamps()[i]);
        if data.geometry.timestamps()[order[0]] < self.epoch {
            return Err(KernelError::InvalidInput("RF data precedes filter epoch").into());
        }
        let mut results = Vec::with_capacity(order.len());
        for i in order {
            let saved = (self.filter.x, self.filter.p, self.epoch, self.steps.len());
            self.predict_to(data.geometry.timestamps()[i], None)?;
            let g = data.geometry.row(i).unwrap();
            let result = if let Some(dd) = differential {
                self.update_with_differential_baseline(
                    &Vector::from_array([
                        data.doppler_hz[i],
                        data.phase_rad.as_ref().unwrap()[i],
                        dd.difference_hz[i],
                    ]),
                    g,
                    data.interferometer,
                    extensions.differential_interferometer,
                    &dd.covariance(data),
                    options,
                )
                .map(|r| (r.nis, r.accepted))
            } else if let Some(phase) = &data.phase_rad {
                self.update(
                    &Vector::from_array([data.doppler_hz[i], phase[i]]),
                    g,
                    data.interferometer,
                    &Matrix::new([
                        [data.sigma_doppler_hz.powi(2), 0.0],
                        [0.0, data.sigma_phase_rad.powi(2)],
                    ]),
                    options,
                )
                .map(|r| (r.nis, r.accepted))
            } else {
                self.update(
                    &Vector::from_array([data.doppler_hz[i]]),
                    g,
                    data.interferometer,
                    &Matrix::new([[data.sigma_doppler_hz.powi(2)]]),
                    options,
                )
                .map(|r| (r.nis, r.accepted))
            };
            match result {
                Ok(r) => results.push(r),
                Err(e) => {
                    self.filter.x = saved.0;
                    self.filter.p = saved.1;
                    self.epoch = saved.2;
                    self.steps.truncate(saved.3);
                    self.timestamps.truncate(saved.3);
                    return Err(e);
                }
            }
        }
        Ok(results)
    }
}

fn validate_state<const N: usize>(x: &Vector<f64, N>, orbit_columns: usize) -> Result<()> {
    let radius = r_of(&V6::from_slice(&x.as_slice()[..6])).norm();
    if !x.as_slice().iter().all(|x| x.is_finite())
        || !radius.is_finite()
        || radius == 0.0
        || (orbit_columns == 7 && x[6] < 0.0)
    {
        return Err(KernelError::InvalidInput(
            "RF state must be finite with valid orbit/drag",
        ));
    }
    Ok(())
}
fn record<const N: usize>(
    x: Vector<f64, N>,
    p: Matrix<f64, N, N>,
    f: Matrix<f64, N, N>,
) -> EkfStep<f64, N> {
    EkfStep {
        x_predicted: x,
        p_predicted: p,
        x_updated: x,
        p_updated: p,
        f_jacobian: f,
    }
}
