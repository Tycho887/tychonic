use super::{
    AcquisitionError, Action, ActionPlan, AntennaLimits, DetectionModel, EnuDirection,
    PointingGeometry, Result, geometry::check_mask, normalize_weights, positive, probability,
};

/// One cycle's moving candidates and current particle geometry. `timing`
/// supplies cycle/listen times; its direction and candidate index are ignored.
#[derive(Debug)]
pub struct PlanningCycle {
    pub timing: Action,
    pub candidates: Vec<EnuDirection>,
    pub particles: Vec<PointingGeometry>,
}

#[derive(Clone, Copy, Debug)]
pub struct PlannerSettings {
    pub horizon: usize,
    pub beam_width: usize,
    pub move_s: f64,
    pub listen_s: f64,
    pub elevation_mask_rad: f64,
    pub antenna: AntennaLimits,
}
impl PlannerSettings {
    fn validate(&self) -> Result<()> {
        if self.horizon == 0 || self.beam_width == 0 {
            return Err(AcquisitionError::InvalidInput(
                "positive horizon and beam width required",
            ));
        }
        positive(self.move_s, "move duration")?;
        positive(self.listen_s, "listen duration")?;
        check_mask(self.elevation_mask_rad)?;
        self.antenna.validate()
    }
    fn reachable(&self, from: EnuDirection, to: EnuDirection) -> Result<bool> {
        Ok(to.elevation_rad() >= self.elevation_mask_rad
            && self.antenna.reachable(from, to, self.move_s)?)
    }
}

/// Eq. (4), equivalently one minus the Eq. (10) weighted full-contact
/// miss probability. Each row contains detection probabilities per particle.
pub fn schedule_acquisition_probability(weights: &[f64], detections: &[Vec<f64>]) -> Result<f64> {
    let weights = normalize_weights(weights)?;
    let mut survival = vec![0.0; weights.len()];
    for row in detections {
        if row.len() != weights.len() {
            return Err(AcquisitionError::InvalidInput(
                "probability table dimensions",
            ));
        }
        for (total, &p) in survival.iter_mut().zip(row) {
            *total += log_miss(p)?;
        }
    }
    Ok(acquisition(log_cost(&weights, &survival)))
}

pub(crate) fn log_miss(p: f64) -> Result<f64> {
    probability(p)?;
    Ok((-p).ln_1p())
}
fn acquisition(log_cost: f64) -> f64 {
    (-log_cost.exp_m1()).clamp(0.0, 1.0)
}
fn log_cost(weights: &[f64], survival: &[f64]) -> f64 {
    // Near zero acquisition, subtracting weighted survival from one loses
    // small probabilities. expm1/log1p preserve them. Near certain acquisition,
    // use log-sum-exp to retain distinctions between tiny miss probabilities.
    let detected: f64 = weights
        .iter()
        .zip(survival)
        .map(|(w, s)| w * -s.exp_m1())
        .sum();
    if detected <= 0.5 {
        return (-detected).ln_1p();
    }
    let maximum = weights
        .iter()
        .zip(survival)
        .map(|(w, s)| w.ln() + s)
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return maximum;
    }
    let sum: f64 = weights
        .iter()
        .zip(survival)
        .map(|(w, s)| (w.ln() + s - maximum).exp())
        .sum();
    maximum + sum.ln()
}

/// Receiver evaluation with checked lengths/values and explicit overwrite.
pub(crate) fn detection_logs(
    model: &mut impl DetectionModel,
    direction: EnuDirection,
    particles: &[PointingGeometry],
    listen_s: f64,
) -> Result<Vec<f64>> {
    let mut probabilities = vec![f64::NAN; particles.len()];
    model.probabilities(direction, particles, listen_s, &mut probabilities)?;
    probabilities
        .iter()
        .zip(particles)
        .map(|(&p, geometry)| {
            if !geometry.visible && p != 0.0 {
                return Err(AcquisitionError::InvalidInput(
                    "invisible particle must have zero detection probability",
                ));
            }
            log_miss(p)
        })
        .collect()
}

fn validate_cycles(
    cycles: &[PlanningCycle],
    retained: &[Action],
    n: usize,
    settings: &PlannerSettings,
) -> Result<()> {
    if cycles.is_empty() || cycles.len() != retained.len() {
        return Err(AcquisitionError::InvalidInput(
            "nonempty matching cycles and continuation required",
        ));
    }
    for (cycle, action) in cycles.iter().zip(retained) {
        validate_cycle(cycle, action, n, settings)?;
    }
    if retained.windows(2).any(|w| {
        w[0].cycle.checked_add(1) != Some(w[1].cycle)
            || (w[1].listen_start - w[0].listen_end).as_seconds() + 1e-6 < settings.move_s
    }) {
        return Err(AcquisitionError::InvalidInput(
            "continuation cycles must be consecutive and chronological",
        ));
    }
    Ok(())
}
fn validate_cycle(
    cycle: &PlanningCycle,
    action: &Action,
    n: usize,
    settings: &PlannerSettings,
) -> Result<()> {
    if cycle.particles.len() != n || cycle.candidates.is_empty() {
        return Err(AcquisitionError::InvalidInput("planning cycle dimensions"));
    }
    validate_timing(cycle.timing, action, settings.listen_s)?;
    for particle in &cycle.particles {
        validate_particle(particle, settings.elevation_mask_rad)?;
    }
    if let Some(index) = action.candidate_index
        && cycle.candidates.get(index) != Some(&action.direction_enu)
    {
        return Err(AcquisitionError::InvalidInput(
            "continuation candidate index",
        ));
    }
    Ok(())
}

fn validate_timing(timing: Action, action: &Action, listen_s: f64) -> Result<()> {
    if action.cycle != timing.cycle
        || action.listen_start != timing.listen_start
        || action.listen_end != timing.listen_end
    {
        return Err(AcquisitionError::InvalidInput(
            "continuation timing mismatch",
        ));
    }
    if timing.listen_start == satkit::Instant::INVALID
        || timing.listen_end == satkit::Instant::INVALID
        || timing.listen_end <= timing.listen_start
        || ((timing.listen_end - timing.listen_start).as_seconds() - listen_s).abs() > 1e-6
    {
        return Err(AcquisitionError::InvalidInput("listen interval"));
    }
    Ok(())
}

fn validate_particle(particle: &PointingGeometry, mask: f64) -> Result<()> {
    positive(particle.range_m, "particle range")?;
    if !particle.elevation_rad.is_finite()
        || (particle.elevation_rad - particle.direction_enu.elevation_rad()).abs() > 1e-12
        || particle.visible != (particle.elevation_rad >= mask)
    {
        return Err(AcquisitionError::InvalidInput(
            "inconsistent particle geometry",
        ));
    }
    Ok(())
}

struct CycleScores {
    values: Vec<(EnuDirection, Vec<f64>)>,
}
impl CycleScores {
    fn logs(
        &mut self,
        command: EnuDirection,
        cycle: &PlanningCycle,
        model: &mut impl DetectionModel,
        listen_s: f64,
    ) -> Result<&[f64]> {
        let index = match self
            .values
            .iter()
            .position(|(direction, _)| *direction == command)
        {
            Some(index) => index,
            None => {
                self.values.push((
                    command,
                    detection_logs(model, command, &cycle.particles, listen_s)?,
                ));
                self.values.len() - 1
            }
        };
        Ok(&self.values[index].1)
    }
}
struct BeamNode {
    actions: Vec<Action>,
    direction: EnuDirection,
    log_survival: Vec<f64>,
    cost: f64,
}
struct PrefixSearch<'a, M> {
    cycles: &'a [PlanningCycle],
    retained: &'a [Action],
    weights: &'a [f64],
    settings: &'a PlannerSettings,
    tail: Vec<f64>,
    scores: Vec<CycleScores>,
    model: &'a mut M,
}
impl<M: DetectionModel> PrefixSearch<'_, M> {
    fn expand(
        &mut self,
        node: &BeamNode,
        depth: usize,
        boundary: Option<EnuDirection>,
    ) -> Result<Vec<BeamNode>> {
        let mut commands: Vec<_> = self.cycles[depth]
            .candidates
            .iter()
            .copied()
            .enumerate()
            .map(|(i, q)| (q, Some(i)))
            .collect();
        commands.push((self.retained[depth].direction_enu, None));
        commands.push((node.direction, None));
        let mut seen = Vec::new();
        let mut children = Vec::new();
        for (command, index) in commands {
            if seen.contains(&command) {
                continue;
            }
            seen.push(command);
            if !self.settings.reachable(node.direction, command)? {
                continue;
            }
            if let Some(next) = boundary
                && !self.settings.reachable(command, next)?
            {
                continue;
            }
            children.push(self.child(node, depth, command, index)?);
        }
        Ok(children)
    }
    fn child(
        &mut self,
        node: &BeamNode,
        depth: usize,
        command: EnuDirection,
        index: Option<usize>,
    ) -> Result<BeamNode> {
        let logs = self.scores[depth].logs(
            command,
            &self.cycles[depth],
            self.model,
            self.settings.listen_s,
        )?;
        let log_survival: Vec<_> = node
            .log_survival
            .iter()
            .zip(logs)
            .map(|(old, next)| old + next)
            .collect();
        let full: Vec<_> = log_survival
            .iter()
            .zip(&self.tail)
            .map(|(prefix, suffix)| prefix + suffix)
            .collect();
        let mut actions = node.actions.to_vec();
        actions.push(Action {
            direction_enu: command,
            candidate_index: index,
            ..self.cycles[depth].timing
        });
        Ok(BeamNode {
            actions,
            direction: command,
            cost: log_cost(self.weights, &full),
            log_survival,
        })
    }
    fn run(
        &mut self,
        initial: EnuDirection,
        horizon: usize,
        incumbent: f64,
    ) -> Result<Option<BeamNode>> {
        let mut beam = vec![BeamNode {
            actions: Vec::new(),
            direction: initial,
            log_survival: vec![0.0; self.weights.len()],
            cost: incumbent,
        }];
        for depth in 0..horizon {
            let boundary = if depth + 1 == horizon {
                self.retained.get(horizon).map(|a| a.direction_enu)
            } else {
                None
            };
            let mut next = Vec::new();
            for node in &beam {
                next.extend(self.expand(node, depth, boundary)?);
            }
            if next.is_empty() {
                return Ok(None);
            }
            // Stable ties follow candidate order, then retained, then hold.
            next.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            next.truncate(self.settings.beam_width);
            beam = next;
        }
        Ok(beam.into_iter().next())
    }
}

/// Optimize a prefix against the full retained continuation (Eqs. 9–10).
/// Beam search is approximate: incomplete prefixes are ranked with the fixed
/// suffix beyond the horizon. The incumbent remains a fallback, so returned
/// probability cannot decrease under the supplied belief and receiver model.
pub fn optimize_prefix(
    initial: EnuDirection,
    retained: &[Action],
    cycles: &[PlanningCycle],
    weights: &[f64],
    settings: &PlannerSettings,
    model: &mut impl DetectionModel,
) -> Result<ActionPlan> {
    optimize_with_pending(initial, retained, cycles, weights, settings, model).map(|(plan, _)| plan)
}

pub(crate) fn optimize_with_pending(
    initial: EnuDirection,
    retained: &[Action],
    cycles: &[PlanningCycle],
    weights: &[f64],
    settings: &PlannerSettings,
    model: &mut impl DetectionModel,
) -> Result<(ActionPlan, Vec<f64>)> {
    settings.validate()?;
    if initial.elevation_rad() < settings.elevation_mask_rad {
        return Err(AcquisitionError::InvalidInput(
            "initial direction must be above mask",
        ));
    }
    let weights = normalize_weights(weights)?;
    validate_cycles(cycles, retained, weights.len(), settings)?;
    let mut previous = initial;
    for action in retained {
        if !settings.reachable(previous, action.direction_enu)? {
            return Err(AcquisitionError::InvalidInput(
                "retained continuation is infeasible",
            ));
        }
        previous = action.direction_enu;
    }
    let horizon = settings.horizon.min(retained.len());
    let mut scores: Vec<_> = cycles
        .iter()
        .map(|_| CycleScores { values: Vec::new() })
        .collect();
    let mut survival = vec![0.0; weights.len()];
    let mut tail = vec![0.0; weights.len()];
    for (depth, (cycle, action)) in cycles.iter().zip(retained).enumerate() {
        let logs = scores[depth].logs(action.direction_enu, cycle, model, settings.listen_s)?;
        for (i, &log) in logs.iter().enumerate() {
            survival[i] += log;
            if depth >= horizon {
                tail[i] += log;
            }
        }
    }
    let incumbent = log_cost(&weights, &survival);
    let mut search = PrefixSearch {
        cycles,
        retained,
        weights: &weights,
        settings,
        tail,
        scores,
        model,
    };
    let best = search.run(initial, horizon, incumbent)?;
    let (actions, cost) = match best {
        Some(best) if best.cost < incumbent => {
            let mut actions = best.actions;
            actions.extend_from_slice(&retained[horizon..]);
            (actions, best.cost)
        }
        _ => (retained.to_vec(), incumbent),
    };
    let pending = search.scores[0]
        .logs(
            actions[0].direction_enu,
            &cycles[0],
            search.model,
            settings.listen_s,
        )?
        .to_vec();
    Ok((
        ActionPlan {
            actions,
            acquisition_probability: acquisition(cost),
            retained_acquisition_probability: acquisition(incumbent),
        },
        pending,
    ))
}
