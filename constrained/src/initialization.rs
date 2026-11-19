//! A local initializer in the projected range-rate metric, with frozen cones.
use super::*;
pub(super) struct Start {
    pub origin: V7,
    pub basis: M7,
    pub initial: [f64; 7],
    pub value: RangeRateEvaluation,
    pub evaluations: usize,
}

impl Start {
    fn direction(&self) -> V7 {
        -(self.basis.transpose() * V7::from_slice(&self.value.gradient))
    }
    fn improves(&self, candidate: &RangeRateEvaluation) -> bool {
        candidate.rss < self.value.rss
            && candidate.worst_violation_rad <= self.value.worst_violation_rad
    }

    // The supplied scales express typical physical corrections. In a weak
    // direction an unconstrained Gauss–Newton step can be kilometres long.
    fn limited_step(&self, scales: &[f64; 7]) -> Option<V7> {
        let scaled = M7::from_fn(|i, k| self.basis[(i, k)] / scales[i]);
        let direction = self.direction();
        if (scaled * direction).norm() <= 5. {
            return Some(direction);
        }
        let metric = scaled.transpose() * scaled;
        let mut damping = 0.01;
        for _ in 0..18 {
            let step = (M7::eye() + metric * damping)
                .cholesky()
                .ok()?
                .solve(&direction);
            if (scaled * step).norm() <= 5. {
                return Some(step);
            }
            damping *= 10.;
        }
        None
    }

    fn recenter(&mut self, model: &ConstrainedRangeRateProblem, scales: &[f64; 7]) {
        let Some(step) = self.limited_step(scales) else {
            return;
        };
        let trial = self.origin + self.basis * step;
        if trial[6] < 0. {
            return;
        }
        self.evaluations += 1;
        if let Ok((value, basis)) = model.linearize(&trial, scales)
            && self.improves(&value)
        {
            self.origin = trial;
            self.value = value;
            self.basis = basis;
        }
    }
}

pub(super) fn prepare(
    model: &ConstrainedRangeRateProblem,
    scales: &[f64; 7],
) -> Result<Start, KernelError> {
    let (value, basis) = model.linearize(model.seed(), scales)?;
    let mut start = Start {
        origin: *model.seed(),
        basis,
        value,
        initial: [0.; 7],
        evaluations: 1,
    };
    let correction = basis * start.direction();
    let scaled_length = V7::from_fn(|i, _| correction[i] / scales[i]).norm();
    if scaled_length > 5. {
        // Refresh the local metric for up to three accepted short corrections.
        // Every evaluation uses the original station geometry and cone selection.
        for _ in 0..3 {
            let prior = start.origin;
            start.recenter(model, scales);
            if prior == start.origin {
                break;
            }
        }
    }
    let Some(step) = start.limited_step(scales) else {
        return Ok(start);
    };
    let trial = start.origin + start.basis * step;
    if step.norm() > 0. && trial[6] >= 0. {
        start.evaluations += 1;
        if let Ok(candidate) = model.evaluate(&trial)
            && start.improves(&candidate)
        {
            start.initial.copy_from_slice(step.as_slice());
            start.value = candidate;
        }
    }
    Ok(start)
}
