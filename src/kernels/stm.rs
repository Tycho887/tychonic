//! Dense state-transition sampling and optional constant-drag sensitivities.
use super::*;
use satkit::{
    Frame, Instant, TLE, frametransform,
    orbitprop::{self, CovState, Integrator, PropSettings, SatPropertiesSimple, TideModel},
    sgp4::{SGP4Error, sgp4},
};

/// Reference parameters are GCRF state, optionally followed by C_d A/m.
#[derive(Clone, Copy, Debug)]
pub enum ParameterMode {
    StateOnly { fixed_drag: f64 },
    StateAndDrag { delta_cd: f64 },
}

impl ParameterMode {
    pub fn columns(self) -> usize {
        match self {
            Self::StateOnly { .. } => 6,
            Self::StateAndDrag { .. } => 7,
        }
    }
}

/// Validated, immutable propagation configuration shared by all models.
#[derive(Clone, Debug)]
pub struct PropagationConfig {
    reference_epoch: Instant,
    settings: PropSettings,
    mode: ParameterMode,
}

impl PropagationConfig {
    pub fn new(
        reference_epoch: Instant,
        settings: PropSettings,
        mode: ParameterMode,
    ) -> Result<Self> {
        check_stm_settings(&settings)?;
        match mode {
            ParameterMode::StateOnly { fixed_drag } => check_drag_value(fixed_drag)?,
            ParameterMode::StateAndDrag { delta_cd } => check_drag(0.0, delta_cd)?,
        }
        Ok(Self {
            reference_epoch,
            settings,
            mode,
        })
    }
    pub fn reference_epoch(&self) -> Instant {
        self.reference_epoch
    }
    pub fn settings(&self) -> &PropSettings {
        &self.settings
    }
    pub fn mode(&self) -> ParameterMode {
        self.mode
    }

    pub fn sample(
        &self,
        parameters: &numeris::DynVector<f64>,
        times: &[Instant],
    ) -> Result<Sensitivities> {
        if parameters.len() != self.mode.columns()
            || !parameters.as_slice().iter().all(|x| x.is_finite())
        {
            return Err(KernelError::InvalidInput(
                "reference parameter dimensions or values",
            ));
        }
        let state = V6::from_slice(&parameters.as_slice()[..6]);
        match self.mode {
            ParameterMode::StateOnly { fixed_drag } => compute_state_transitions(
                &state,
                &self.reference_epoch,
                times,
                &self.settings,
                fixed_drag,
            ),
            ParameterMode::StateAndDrag { delta_cd } => compute_sensitivities(
                &state,
                &self.reference_epoch,
                times,
                &self.settings,
                parameters[6],
                delta_cd,
            ),
        }
    }
}

fn check_drag_value(cda: f64) -> Result<()> {
    if !cda.is_finite() || cda < 0.0 {
        return Err(KernelError::InvalidInput(
            "C_d A/m must be finite and nonnegative",
        ));
    }
    Ok(())
}

/// Defaults for the verified STM force model: spherical-harmonic gravity,
/// Sun/Moon gravity and atmospheric drag. SRP is zero in these kernels.
/// Satkit omits tide/relativity partials, so they are disabled explicitly.
/// Drag density gradients in satkit are approximate (altitude only).
pub fn od_prop_settings() -> PropSettings {
    PropSettings {
        abs_error: 1e-10,
        rel_error: 1e-13,
        tide_model: TideModel::None,
        use_relativistic_correction: false,
        ..PropSettings::default()
    }
}

pub(crate) fn check_stm_settings(settings: &PropSettings) -> Result<()> {
    if matches!(
        settings.integrator,
        Integrator::RODAS4 | Integrator::GaussJackson8
    ) {
        return Err(KernelError::InvalidInput(
            "integrator does not support STMs",
        ));
    }
    if settings.tide_model != TideModel::None || settings.use_relativistic_correction {
        return Err(KernelError::InvalidInput(
            "satkit STM lacks tide/relativity partials; use od_prop_settings()",
        ));
    }
    Ok(())
}

pub fn propagate_state(
    x: &V6,
    from: &Instant,
    to: &Instant,
    settings: &PropSettings,
    props: &SatPropertiesSimple,
) -> Result<V6> {
    check_state(x)?;
    if !props.cdaoverm.is_finite()
        || props.cdaoverm < 0.0
        || !props.craoverm.is_finite()
        || props.craoverm < 0.0
    {
        return Err(KernelError::InvalidInput(
            "drag and SRP susceptibilities must be finite and nonnegative",
        ));
    }
    let out = orbitprop::propagate(x, from, to, settings, Some(props))?.state_end;
    check_state(&out)?;
    Ok(out)
}

/// Propagate in either direction; Φ = ∂x(to)/∂x(from).
/// Only constant drag properties are supported for the verified STM model.
pub fn propagate_with_stm(
    x: &V6,
    from: &Instant,
    to: &Instant,
    settings: &PropSettings,
    cda_over_m: f64,
) -> Result<(V6, M6)> {
    check_state(x)?;
    check_stm_settings(settings)?;
    check_drag_value(cda_over_m)?;
    let cov = cov_state(x);
    let props = SatPropertiesSimple::new(cda_over_m, 0.0);
    let out = orbitprop::propagate(&cov, from, to, settings, Some(&props))?.state_end;
    let state = out.block::<6, 1>(0, 0);
    let stm = out.block::<6, 6>(0, 1);
    check_state(&state)?;
    if !finite(&stm) {
        return Err(KernelError::InvalidInput("non-finite propagated STM"));
    }
    Ok((state, stm))
}

fn cov_state(x: &V6) -> CovState {
    let mut cov = CovState::zeros();
    cov.set_block(0, 0, x);
    cov.set_block(0, 1, &M6::eye());
    cov
}

pub(crate) fn check_drag(cda: f64, step: f64) -> Result<()> {
    if !cda.is_finite() || cda < 0.0 {
        return Err(KernelError::InvalidInput(
            "C_d A/m must be finite and nonnegative",
        ));
    }
    if !step.is_finite() || step <= 0.0 || !((cda + 2.0 * step).is_finite()) || cda + step == cda {
        return Err(KernelError::InvalidInput(
            "drag difference step must be finite, positive and resolvable",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Sensitivities {
    pub timestamps: Vec<Instant>,
    pub states: Vec<V6>,
    pub stm: Vec<M6>,
    /// ∂x(t_i)/∂(C_d A/m). Empty for state-only sampling.
    pub state_dcd: Vec<V6>,
}

impl Sensitivities {
    /// Checks array dimensions; finite values are checked at composition time.
    pub fn columns(&self) -> Result<usize> {
        let n = self.timestamps.len();
        if self.states.len() != n
            || self.stm.len() != n
            || (!self.state_dcd.is_empty() && self.state_dcd.len() != n)
        {
            return Err(KernelError::InvalidInput(
                "sensitivity lengths do not match",
            ));
        }
        Ok(if self.state_dcd.is_empty() { 6 } else { 7 })
    }

    /// EKF transition for a constant drag parameter: [Φ, dx/d(C_d A/m); 0, 1].
    pub fn augmented_transition(&self, i: usize) -> Result<M7> {
        if i >= self.timestamps.len()
            || i >= self.states.len()
            || i >= self.stm.len()
            || i >= self.state_dcd.len()
        {
            return Err(KernelError::InvalidInput("sensitivity index out of bounds"));
        }
        if !finite(&self.stm[i]) || !finite(&self.state_dcd[i]) {
            return Err(KernelError::InvalidInput("sensitivities must be finite"));
        }
        let mut phi = M7::eye();
        phi.set_block(0, 0, &self.stm[i]);
        phi.set_block(0, 6, &self.state_dcd[i]);
        Ok(phi)
    }
}

/// Evaluate one or two complete arcs with dense output, then sample in the
/// caller's order. Supports unsorted, duplicate, forward and backward epochs.
fn sample_arc<const C: usize>(
    x: &Matrix<f64, 6, C>,
    epoch: &Instant,
    times: &[Instant],
    settings: &PropSettings,
    props: &SatPropertiesSimple,
) -> Result<Vec<Matrix<f64, 6, C>>> {
    let mut samples = vec![*x; times.len()];
    for forward in [false, true] {
        let end = times
            .iter()
            .copied()
            .filter(|t| if forward { *t > *epoch } else { *t < *epoch })
            .reduce(|a, b| if forward { a.max(b) } else { a.min(b) });
        if let Some(end) = end {
            let mut dense_settings = settings.clone();
            dense_settings.enable_interp = true;
            if dense_settings.integrator == Integrator::RKV98NoInterp {
                dense_settings.integrator = Integrator::RKV98;
            }
            let arc = orbitprop::propagate(x, epoch, &end, &dense_settings, Some(props))?;
            for (i, t) in times.iter().enumerate() {
                if if forward { *t > *epoch } else { *t < *epoch } {
                    samples[i] = if *t == end {
                        arc.state_end
                    } else {
                        arc.interp(t)?
                    };
                }
            }
        }
    }
    Ok(samples)
}

/// Sample one nominal state/STM arc per direction with fixed drag. No drag
/// perturbation arcs are evaluated; `state_dcd` is empty in the result.
pub fn compute_state_transitions(
    x: &V6,
    epoch: &Instant,
    times: &[Instant],
    settings: &PropSettings,
    fixed_drag: f64,
) -> Result<Sensitivities> {
    check_state(x)?;
    check_stm_settings(settings)?;
    check_drag_value(fixed_drag)?;
    let nominal = sample_arc(
        &cov_state(x),
        epoch,
        times,
        settings,
        &SatPropertiesSimple::new(fixed_drag, 0.0),
    )?;
    let mut sens = Sensitivities {
        timestamps: times.to_vec(),
        states: Vec::with_capacity(times.len()),
        stm: Vec::with_capacity(times.len()),
        state_dcd: Vec::new(),
    };
    for sample in nominal {
        let state = sample.block::<6, 1>(0, 0);
        let phi = sample.block::<6, 6>(0, 1);
        check_state(&state)?;
        if !finite(&phi) {
            return Err(KernelError::InvalidInput("non-finite propagated STM"));
        }
        sens.states.push(state);
        sens.stm.push(phi);
    }
    Ok(sens)
}

/// STM plus second-order drag differences. Uses central differences away
/// from zero and a second-order forward stencil at the nonnegative boundary.
/// `delta_cd` is an absolute step in m²/kg; check stability with smaller steps
/// for your altitude, arc length and integration tolerances.
pub fn compute_sensitivities(
    x: &V6,
    epoch: &Instant,
    times: &[Instant],
    settings: &PropSettings,
    cda_over_m: f64,
    delta_cd: f64,
) -> Result<Sensitivities> {
    check_state(x)?;
    check_stm_settings(settings)?;
    check_drag(cda_over_m, delta_cd)?;
    let props = SatPropertiesSimple::new(cda_over_m, 0.0);
    let nominal = sample_arc(&cov_state(x), epoch, times, settings, &props)?;
    let plus = sample_arc(
        x,
        epoch,
        times,
        settings,
        &SatPropertiesSimple::new(cda_over_m + delta_cd, 0.0),
    )?;
    let central = cda_over_m >= delta_cd;
    let other_cd = if central {
        cda_over_m - delta_cd
    } else {
        cda_over_m + 2.0 * delta_cd
    };
    let other = sample_arc(
        x,
        epoch,
        times,
        settings,
        &SatPropertiesSimple::new(other_cd, 0.0),
    )?;
    // A simple-state baseline avoids contamination by a different adaptive
    // step sequence in the STM propagation for the one-sided stencil.
    let baseline = if central {
        Vec::new()
    } else {
        sample_arc(x, epoch, times, settings, &props)?
    };
    let mut sens = Sensitivities {
        timestamps: times.to_vec(),
        states: Vec::with_capacity(times.len()),
        stm: Vec::with_capacity(times.len()),
        state_dcd: Vec::with_capacity(times.len()),
    };
    for i in 0..times.len() {
        let state = nominal[i].block::<6, 1>(0, 0);
        let stm = nominal[i].block::<6, 6>(0, 1);
        let derivative = if times[i] == *epoch {
            V6::zeros()
        } else if central {
            (plus[i] - other[i]) / (2.0 * delta_cd)
        } else {
            ((plus[i] - baseline[i]) * 4.0 - (other[i] - baseline[i])) / (2.0 * delta_cd)
        };
        check_state(&state)?;
        if !finite(&stm) || !finite(&derivative) {
            return Err(KernelError::InvalidInput(
                "non-finite propagated sensitivity",
            ));
        }
        sens.states.push(state);
        sens.stm.push(stm);
        sens.state_dcd.push(derivative);
    }
    Ok(sens)
}

pub fn initial_state_from_tle(tle: &TLE, epoch: &Instant) -> Result<V6> {
    let mut tle = tle.clone();
    let out = sgp4(&mut tle, &[*epoch]).map_err(|e| KernelError::Tle(e.to_string()))?;
    if out.errcode[0] != SGP4Error::SGP4Success {
        return Err(KernelError::Tle(out.errcode[0].to_string()));
    }
    let r = V3::from_slice(out.pos.col_slice(0));
    let v = V3::from_slice(out.vel.col_slice(0));
    let (r, v) = frametransform::transform_state(Frame::TEME, Frame::GCRF, epoch, &r, &v)
        .map_err(KernelError::Frame)?;
    Ok(v6_from(&r, &v))
}
