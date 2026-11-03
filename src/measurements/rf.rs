//! Named priors and setup for the existing RF offset/drift estimators.
use crate::{
    estimate::{EkfResult, PassiveRfEkf},
    kernels::*,
    optim::*,
};
use numeris::{DynVector, optim::LmSettings};

/// Scalar Gaussian prior in the physical units of its parameter.
#[derive(Clone, Copy, Debug)]
pub struct GaussianPrior {
    pub mean: f64,
    pub sigma: f64,
}

/// Named RF priors. Orbit entries are GCRF position/velocity, optionally drag.
/// Frequency uses Hz and phase uses radians. No implicit offset or drift priors.
#[derive(Clone, Debug)]
pub struct RfPriors {
    pub orbit_mean: DynVector<f64>,
    pub orbit_sigma: DynVector<f64>,
    pub frequency_offset_hz: GaussianPrior,
    pub phase_offset_rad: GaussianPrior,
}

impl RfPriors {
    fn prepare(&self, columns: usize) -> Result<RfPrior> {
        if self.orbit_mean.len() != columns || self.orbit_sigma.len() != columns {
            return Err(KernelError::InvalidInput("RF orbit prior dimensions"));
        }
        let mut mean = self.orbit_mean.as_slice().to_vec();
        let mut sigma = self.orbit_sigma.as_slice().to_vec();
        mean.extend([self.frequency_offset_hz.mean, self.phase_offset_rad.mean]);
        sigma.extend([self.frequency_offset_hz.sigma, self.phase_offset_rad.sigma]);
        let prior = RfPrior {
            mean: DynVector::from_slice(&mean),
            sigma: DynVector::from_slice(&sigma),
        };
        prior.validate(columns + 2)?;
        Ok(prior)
    }
}

/// Setup for Doppler, Doppler/phase, or Doppler/phase/differential fitting.
/// These are the existing augmented RF combinations; calibrated arbitrary
/// combinations and angles use the ordinary measurement facade instead.
#[derive(Clone, Debug)]
pub struct RfBuilder {
    config: PropagationConfig,
    data: PassiveRfData,
    priors: RfPriors,
    drift: OuPhaseNoise,
    extensions: RfExtensions,
}

impl RfBuilder {
    pub fn new(
        config: PropagationConfig,
        data: PassiveRfData,
        priors: RfPriors,
        drift: OuPhaseNoise,
    ) -> Self {
        Self {
            config,
            data,
            priors,
            drift,
            extensions: RfExtensions::default(),
        }
    }
    /// Estimate constant ENU corrections to the local phase baseline.
    pub fn baseline_calibration(mut self, sigma_m: V3) -> Self {
        self.extensions.baseline_sigma_m = Some(sigma_m);
        self
    }
    /// Add f1 − f2 measurements; None shares the phase baseline.
    pub fn differential_doppler(
        mut self,
        data: DifferentialDopplerData,
        interferometer: Option<Interferometer>,
    ) -> Self {
        self.extensions.differential_doppler = Some(data);
        self.extensions.differential_interferometer = interferometer;
        self
    }
    /// Estimate an independent, separate Doppler baseline correction.
    pub fn differential_baseline_calibration(mut self, sigma_m: V3) -> Self {
        self.extensions.differential_baseline_sigma_m = Some(sigma_m);
        self
    }
    pub fn build(self) -> Result<PassiveRfProblem> {
        let prior = self.priors.prepare(self.config.mode().columns())?;
        PassiveRfProblem::new_extended(self.config, self.data, prior, self.drift, self.extensions)
    }
    /// Fit starting from the supplied named prior means, with numeris defaults.
    pub fn fit(self) -> std::result::Result<PassiveRfFit, FitError> {
        self.fit_with_options(&LmSettings::default())
    }
    pub fn fit_with_options(
        self,
        settings: &LmSettings<f64>,
    ) -> std::result::Result<PassiveRfFit, FitError> {
        let problem = self.build()?;
        fit_passive_rf(&problem, &problem.initial_parameters(), settings)
    }
    /// Construct the corresponding streaming estimator. N is 9/10 without
    /// calibration, plus three states per calibrated baseline. Data are validated
    /// with shared input checks, without allocating the dense batch covariance.
    pub fn filter<const N: usize>(&self) -> EkfResult<PassiveRfEkf<N>> {
        let prior = self.priors.prepare(self.config.mode().columns())?;
        crate::optim::passive_rf::validate_rf_setup(
            &self.config,
            &self.data,
            &prior,
            &self.extensions,
        )?;
        PassiveRfEkf::new_with_calibration(
            self.config.clone(),
            self.drift,
            &prior,
            self.extensions.baseline_sigma_m,
            self.extensions.differential_baseline_sigma_m,
        )
    }
    /// Feed this builder's data and extensions to its matching streaming filter.
    pub fn filter_data<const N: usize>(
        &self,
        filter: &mut PassiveRfEkf<N>,
        options: crate::estimate::UpdateOptions,
    ) -> EkfResult<Vec<(f64, bool)>> {
        filter.filter_data_with_extensions(&self.data, &self.extensions, options)
    }
}
