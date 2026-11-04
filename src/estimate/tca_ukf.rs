//! Pass-level TLE epoch correction with a nearly constant range-rate bias.
//!
//! The state is `[dT_seconds, bias_m_per_s]`. Positive `dT` moves the TLE epoch
//! later, leaving its mean elements fixed. Every sigma-point TLE is propagated
//! at the actual observation epoch; station geometry and frame conversions use
//! that epoch too. The observations are `[range_rate, azimuth, elevation]`, in
//! m/s and radians, with range-rate positive for a receding spacecraft.
//!
//! Pointing constraints are Gaussian measurements, not hard bounds. The noise
//! constructor uses independent 2.5-degree standard deviations for both angles.
//! This single-Gaussian UKF requires a prior covariance covering the epoch error;
//! it does not search for other passes or fit other orbital elements.

use super::{UpdateOptions, UpdateResult, ekf::check_covariance};
use crate::kernels::{
    KernelError, Measurement, MeasurementNoise, PreparedGeometry, Result, StationGeometry, V3,
    az_el_partial_prepared, initial_state_from_tle, r_of, range_rate_partial, v_of, wrap_pi,
};
use numeris::{Matrix, Vector};
use satkit::{Duration, Instant, TLE, sgp4::SGP4Source};

/// `[TLE epoch correction (seconds), additive range-rate bias (m/s)]`.
pub type TcaState = Vector<f64, 2>;
/// Covariance in squared seconds, squared m/s, and their cross units.
pub type TcaCovariance = Matrix<f64, 2, 2>;

// Merwe parameters alpha=1, beta=2, kappa=0, with N=2.
const MEAN_WEIGHTS: [f64; 5] = [0.0, 0.25, 0.25, 0.25, 0.25];
const COVARIANCE_WEIGHTS: [f64; 5] = [2.0, 0.25, 0.25, 0.25, 0.25];

/// Combined measurement noise with 2.5-degree az/el standard deviations.
/// `sigma_rr` is a positive range-rate standard deviation in m/s. Callers can
/// modify `sigma_az` and `sigma_el` (radians) to adjust the pointing constraint.
pub fn tca_measurement_noise(sigma_rr: f64) -> Result<MeasurementNoise> {
    let noise = MeasurementNoise {
        sigma_rr,
        sigma_az: 2.5_f64.to_radians(),
        sigma_el: 2.5_f64.to_radians(),
    };
    measurement_sigmas(&noise)?;
    Ok(noise)
}

/// Two-state unscented filter for combined range-rate and soft az/el readings.
/// One filter represents one pass and one shared bias. State dynamics are
/// constant, with optional caller-supplied discretized process covariance.
#[derive(Clone, Debug)]
pub struct TcaUkf {
    prior: TLE,
    epoch: Instant,
    state: TcaState,
    covariance: TcaCovariance,
}

impl TcaUkf {
    /// Construct at an observation epoch, independently of the prior TLE epoch.
    /// `initial` and its positive-definite covariance are caller supplied.
    /// The prior is validated by propagating its initially corrected copy.
    pub fn new(
        prior: TLE,
        epoch: Instant,
        initial: TcaState,
        covariance: TcaCovariance,
    ) -> Result<Self> {
        check_finite(&initial)?;
        check_covariance(&covariance, true)?;
        check_epoch(epoch)?;
        let corrected = shifted_tle(&prior, initial[0])?;
        check_finite(&initial_state_from_tle(&corrected, &epoch)?)?;
        Ok(Self {
            prior,
            epoch,
            state: initial,
            covariance,
        })
    }

    pub fn state(&self) -> &TcaState {
        &self.state
    }

    pub fn covariance(&self) -> &TcaCovariance {
        &self.covariance
    }

    /// Current observation epoch, not the corrected TLE epoch.
    pub fn epoch(&self) -> Instant {
        self.epoch
    }

    pub fn prior_tle(&self) -> &TLE {
        &self.prior
    }

    /// Copy the prior with `epoch = prior.epoch + dT` and a cleared SGP4 cache.
    /// Its other fields are preserved. The range-rate bias is a sensor parameter
    /// and does not enter the corrected TLE. Epochs round to satkit microseconds.
    pub fn corrected_tle(&self) -> Result<TLE> {
        shifted_tle(&self.prior, self.state[0])
    }

    /// Constant-state prediction, adding optional discretized process covariance.
    /// Q uses the same units as the state covariance; no duration scaling is
    /// applied here. None means zero noise, including a constant pass bias.
    /// Epochs must be chronological; same-epoch predictions require zero Q.
    /// Errors leave the filter unchanged.
    pub fn predict_to(&mut self, time: Instant, q: Option<&TcaCovariance>) -> Result<()> {
        check_epoch(time)?;
        if time < self.epoch {
            return Err(KernelError::InvalidInput(
                "TCA UKF epochs must be chronological",
            ));
        }
        if let Some(q) = q {
            check_covariance(q, false)?;
        }
        if time == self.epoch {
            if q.is_some_and(|q| q.as_slice().iter().any(|v| *v != 0.0)) {
                return Err(KernelError::InvalidInput(
                    "same-epoch prediction requires zero process noise",
                ));
            }
            return Ok(());
        }
        let covariance = self.covariance + q.copied().unwrap_or_else(Matrix::zeros);
        check_covariance(&covariance, true)?;
        self.covariance = covariance;
        self.epoch = time;
        Ok(())
    }

    /// Predict and jointly update range-rate, azimuth, and elevation.
    /// Noise components are independent standard deviations in m/s and radians.
    /// `options.gate` is a NIS threshold with three measurement degrees of freedom;
    /// the default disables gating so a distant prior can acquire the pass.
    /// Errors roll back prediction and update. A gated observation retains its
    /// prediction, including process noise, without applying a measurement update.
    pub fn observe(
        &mut self,
        measurement: &Measurement,
        noise: &MeasurementNoise,
        q: Option<&TcaCovariance>,
        options: UpdateOptions,
    ) -> Result<UpdateResult<3>> {
        let observed = measurement_values(measurement, options)?;
        let sigmas = measurement_sigmas(noise)?;
        let mut candidate = self.clone();
        candidate.predict_to(measurement.time, q)?;
        let geometry = PreparedGeometry::new(&[measurement.time], &[measurement.station])?;
        let result = candidate.update(
            &observed,
            geometry.row(0).expect("one prepared station"),
            &sigmas,
            options,
        )?;
        *self = candidate;
        Ok(result)
    }

    fn update(
        &mut self,
        observed: &V3,
        geometry: &StationGeometry,
        noise_sigmas: &V3,
        options: UpdateOptions,
    ) -> Result<UpdateResult<3>> {
        let sigmas = sigma_points(&self.state, &self.covariance)?;
        let mut predictions = [V3::zeros(); 5];
        for (prediction, sigma) in predictions.iter_mut().zip(&sigmas) {
            *prediction = self.predict_measurement(sigma, geometry)?;
        }
        let transform =
            measurement_transform(&self.state, &sigmas, &predictions, observed, noise_sigmas)?;
        let factor = transform
            .covariance
            .cholesky()
            .map_err(|_| KernelError::Singular)?;
        let innovation = normalize(&transform.innovation, noise_sigmas);
        let nis = innovation.dot(&factor.solve(&innovation));
        if !nis.is_finite() || nis < 0.0 {
            return Err(KernelError::InvalidInput(
                "TCA UKF NIS exceeds numeric range",
            ));
        }
        let accepted = options.gate.is_none_or(|gate| nis <= gate);
        if accepted {
            let gain = factor
                .solve_matrix(&transform.cross.transpose())
                .transpose();
            let state = self.state + gain * innovation;
            let covariance =
                posterior_covariance(&self.state, &sigmas, &transform.deviations, &gain);
            check_finite(&state)?;
            shifted_tle(&self.prior, state[0])?;
            check_covariance(&covariance, true)?;
            self.state = state;
            self.covariance = covariance;
        }
        Ok(UpdateResult {
            innovation: transform.innovation,
            nis,
            accepted,
        })
    }

    fn predict_measurement(&self, state: &TcaState, geometry: &StationGeometry) -> Result<V3> {
        let tle = shifted_tle(&self.prior, state[0])?;
        let satellite = initial_state_from_tle(&tle, &self.epoch)?;
        let r = r_of(&satellite);
        let (rr, _) = range_rate_partial(
            &r,
            &v_of(&satellite),
            geometry.position(),
            geometry.velocity(),
        )?;
        let (angles, _) = az_el_partial_prepared(&r, geometry.position(), geometry.rotation())?;
        let predicted = V3::from_array([rr + state[1], angles[0], angles[1]]);
        check_finite(&predicted)?;
        Ok(predicted)
    }
}

fn shifted_tle(prior: &TLE, seconds: f64) -> Result<TLE> {
    let microseconds = (seconds * 1e6).round();
    if !microseconds.is_finite()
        || microseconds <= i64::MIN as f64
        || microseconds >= i64::MAX as f64
    {
        return Err(KernelError::InvalidInput(
            "TCA epoch correction exceeds numeric range",
        ));
    }
    check_epoch(prior.epoch)?;
    let mut tle = prior.clone();
    tle.epoch = prior
        .epoch
        .checked_add(Duration::from_seconds(seconds))
        .ok_or(KernelError::InvalidInput(
            "corrected TLE epoch exceeds numeric range",
        ))?;
    check_epoch(tle.epoch)?;
    // Changing the public epoch does not invalidate satkit's derived cache.
    *tle.satrec_mut() = None;
    Ok(tle)
}

fn check_epoch(time: Instant) -> Result<()> {
    if time == Instant::INVALID {
        return Err(KernelError::InvalidInput("TCA UKF epoch must be valid"));
    }
    Ok(())
}

fn check_finite<const R: usize, const C: usize>(value: &Matrix<f64, R, C>) -> Result<()> {
    if !value.as_slice().iter().all(|v| v.is_finite()) {
        return Err(KernelError::InvalidInput("TCA UKF values must be finite"));
    }
    Ok(())
}

fn measurement_values(measurement: &Measurement, options: UpdateOptions) -> Result<V3> {
    let value = V3::from_array([
        measurement.range_rate,
        measurement.azimuth,
        measurement.elevation,
    ]);
    check_finite(&value)?;
    if measurement.elevation.abs() > std::f64::consts::FRAC_PI_2 {
        return Err(KernelError::InvalidInput(
            "measurement elevation must be in [-pi/2, pi/2]",
        ));
    }
    if options
        .gate
        .is_some_and(|gate| !gate.is_finite() || gate < 0.0)
    {
        return Err(KernelError::InvalidInput(
            "TCA UKF NIS gate must be finite and nonnegative",
        ));
    }
    Ok(value)
}

fn measurement_sigmas(noise: &MeasurementNoise) -> Result<V3> {
    let sigmas = V3::from_array([noise.sigma_rr, noise.sigma_az, noise.sigma_el]);
    if !sigmas
        .as_slice()
        .iter()
        .all(|s| s.is_finite() && *s > 0.0 && (s * s).is_finite() && s * s > 0.0)
    {
        return Err(KernelError::InvalidInput(
            "TCA measurement standard deviations must be finite and positive",
        ));
    }
    Ok(sigmas)
}

fn sigma_points(state: &TcaState, covariance: &TcaCovariance) -> Result<[TcaState; 5]> {
    // Factor in correlation units to avoid absolute pivot tolerances coupling
    // the different state units. Row scaling restores the physical covariance.
    let scales = TcaState::from_array([covariance[(0, 0)].sqrt(), covariance[(1, 1)].sqrt()]);
    let correlation = covariance[(0, 1)] / scales[0] / scales[1];
    let scaled = Matrix::new([[1.0, correlation], [correlation, 1.0]]);
    let mut root = scaled
        .cholesky()
        .map_err(|_| KernelError::Singular)?
        .l_full();
    for i in 0..2 {
        for j in 0..2 {
            root[(i, j)] *= std::f64::consts::SQRT_2 * scales[i];
        }
    }
    let points = [
        *state,
        state + root.block::<2, 1>(0, 0),
        state - root.block::<2, 1>(0, 0),
        state + root.block::<2, 1>(0, 1),
        state - root.block::<2, 1>(0, 1),
    ];
    for point in &points {
        check_finite(point)?;
    }
    Ok(points)
}

struct MeasurementTransform {
    innovation: V3,
    deviations: [V3; 5],
    covariance: Matrix<f64, 3, 3>,
    cross: Matrix<f64, 2, 3>,
}

fn measurement_mean(predictions: &[V3; 5]) -> Result<V3> {
    let mut mean = V3::zeros();
    let mut sin = 0.0;
    let mut cos = 0.0;
    for (prediction, weight) in predictions.iter().zip(MEAN_WEIGHTS) {
        mean[0] += weight * prediction[0];
        mean[2] += weight * prediction[2];
        sin += weight * prediction[1].sin();
        cos += weight * prediction[1].cos();
    }
    if sin.hypot(cos) <= 1e-12 {
        return Err(KernelError::InvalidInput(
            "azimuth sigma points have no defined circular mean",
        ));
    }
    mean[1] = sin.atan2(cos);
    check_finite(&mean)?;
    Ok(mean)
}

fn measurement_difference(observed: &V3, predicted: &V3) -> V3 {
    V3::from_array([
        observed[0] - predicted[0],
        wrap_pi(wrap_pi(observed[1]) - predicted[1]),
        observed[2] - predicted[2],
    ])
}

fn normalize(value: &V3, sigmas: &V3) -> V3 {
    V3::from_array([
        value[0] / sigmas[0],
        value[1] / sigmas[1],
        value[2] / sigmas[2],
    ])
}

fn measurement_transform(
    state: &TcaState,
    sigmas: &[TcaState; 5],
    predictions: &[V3; 5],
    observed: &V3,
    noise_sigmas: &V3,
) -> Result<MeasurementTransform> {
    let mean = measurement_mean(predictions)?;
    let deviations = std::array::from_fn(|i| {
        normalize(
            &measurement_difference(&predictions[i], &mean),
            noise_sigmas,
        )
    });
    // Whitening by the sensor standard deviations makes R=I and prevents mixed
    // m/s and radian units from conditioning the innovation covariance poorly.
    let mut covariance = Matrix::<f64, 3, 3>::eye();
    let mut cross = Matrix::<f64, 2, 3>::zeros();
    for i in 0..5 {
        let dx = sigmas[i] - state;
        let dz = deviations[i];
        covariance += dz * dz.transpose() * COVARIANCE_WEIGHTS[i];
        cross += dx * dz.transpose() * COVARIANCE_WEIGHTS[i];
    }
    let innovation = measurement_difference(observed, &mean);
    check_finite(&innovation)?;
    check_finite(&cross)?;
    check_covariance(&covariance, true)?;
    Ok(MeasurementTransform {
        innovation,
        deviations,
        covariance,
        cross,
    })
}

fn posterior_covariance(
    state: &TcaState,
    sigmas: &[TcaState; 5],
    deviations: &[V3; 5],
    gain: &Matrix<f64, 2, 3>,
) -> TcaCovariance {
    // Equivalent to P - K S K^T, expressed as positive covariance terms to
    // avoid subtracting nearly equal matrices after a precise range-rate update.
    let mut covariance = gain * gain.transpose();
    for i in 0..5 {
        let residual = sigmas[i] - state - gain * deviations[i];
        covariance += residual * residual.transpose() * COVARIANCE_WEIGHTS[i];
    }
    (covariance + covariance.transpose()) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn assert_close<const R: usize, const C: usize>(a: &Matrix<f64, R, C>, b: &Matrix<f64, R, C>) {
        for (actual, expected) in a.as_slice().iter().zip(b.as_slice()) {
            assert!(
                (actual - expected).abs() < 1e-11 * (1.0 + expected.abs()),
                "{a:?} != {b:?}"
            );
        }
    }

    #[test]
    fn joint_transform_matches_linear_information_update_across_azimuth_branch() {
        let state = TcaState::from_array([3.0, -2.0]);
        let p = Matrix::new([[4.0, 1.0], [1.0, 9.0]]);
        let sigmas = sigma_points(&state, &p).unwrap();
        let h = Matrix::new([[2.0, 1.0], [0.01, 0.0], [0.02, 0.01]]);
        let offset = V3::from_array([50.0, PI - 0.04, 0.6]);
        let predictions = sigmas.map(|x| {
            let mut y = h * x + offset;
            y[1] = wrap_pi(y[1]);
            y
        });
        assert!(predictions.iter().any(|y| y[1] < 0.0));
        assert!(predictions.iter().any(|y| y[1] > 0.0));
        let noise = V3::from_array([0.5, 0.05, 0.03]);
        let innovation = V3::from_array([0.4, 0.01, -0.02]);
        let observed = h * state + offset + innovation;
        let transform =
            measurement_transform(&state, &sigmas, &predictions, &observed, &noise).unwrap();
        let factor = transform.covariance.cholesky().unwrap();
        let gain = factor
            .solve_matrix(&transform.cross.transpose())
            .transpose();
        let actual_state = state + gain * normalize(&transform.innovation, &noise);
        let actual_p = posterior_covariance(&state, &sigmas, &transform.deviations, &gain);
        let inverse_r = Matrix::new([
            [4.0, 0.0, 0.0],
            [0.0, 400.0, 0.0],
            [0.0, 0.0, 1.0 / 0.03_f64.powi(2)],
        ]);
        let expected_p = (p.cholesky().unwrap().inverse() + h.transpose() * inverse_r * h)
            .cholesky()
            .unwrap()
            .inverse();
        let expected_state = state + expected_p * h.transpose() * inverse_r * innovation;
        assert_close(&actual_state, &expected_state);
        assert_close(&actual_p, &expected_p);
        assert_close(&transform.innovation, &innovation);
    }

    #[test]
    fn circular_mean_and_covariance_keep_branch_crossing_deviations_small() {
        let state = TcaState::zeros();
        let sigmas = sigma_points(&state, &Matrix::eye()).unwrap();
        let predictions = [PI, PI - 0.02, -PI + 0.02, PI - 0.03, -PI + 0.03]
            .map(|az| V3::from_array([0.0, az, 0.0]));
        let observed = V3::from_array([0.0, -PI + 0.01, 0.0]);
        let transform = measurement_transform(
            &state,
            &sigmas,
            &predictions,
            &observed,
            &V3::from_array([1.0; 3]),
        )
        .unwrap();
        assert!((transform.innovation[1] - 0.01).abs() < 1e-14);
        assert!((transform.covariance[(1, 1)] - 1.00065).abs() < 1e-14);
        for deviation in transform.deviations {
            assert!(deviation[1].abs() <= 0.031);
        }
        let mut equivalent = observed;
        equivalent[1] += 2.0 * PI;
        assert_close(
            &measurement_difference(&equivalent, &measurement_mean(&predictions).unwrap()),
            &transform.innovation,
        );
    }

    #[test]
    fn sigma_points_reconstruct_correlated_covariance_in_mixed_units() {
        let state = TcaState::from_array([2.0, 10.0]);
        for p in [
            Matrix::new([[9.0, 3.0], [3.0, 4.0]]),
            Matrix::new([[1e-12, 0.2], [0.2, 1e12]]),
        ] {
            let points = sigma_points(&state, &p).unwrap();
            let mut actual = TcaCovariance::zeros();
            for i in 0..5 {
                let delta = points[i] - state;
                actual += delta * delta.transpose() * COVARIANCE_WEIGHTS[i];
            }
            assert_close(&actual, &p);
        }
    }

    #[test]
    fn ambiguous_azimuth_sigma_points_return_error() {
        let predictions =
            [0.0, 0.0, PI, PI / 2.0, -PI / 2.0].map(|az| V3::from_array([0.0, az, 0.0]));
        assert!(measurement_mean(&predictions).is_err());
    }
}
