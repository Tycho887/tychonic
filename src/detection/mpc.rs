use super::{
    AcquisitionError, Action, ActionPlan, DetectionModel, EnuDirection, Ephemeris, Observation,
    ObservationUpdate, ParticleBelief, PlannerSettings, PlanningCycle, PointingGeometry, Result,
    SearchSetup, SearchStatus, along_track_candidates, belief::posterior_weights,
    geometry::check_mask, planner::optimize_with_pending, pointing_geometry,
    systematic_resampling_indices,
};
use crate::kernels::{PreparedGeometry, propagate_sgp4_gcrf, r_of};
use satkit::{
    Instant,
    sgp4::{SGP4InitArgs, SGP4Source, SatRec},
};

/// Propagate an SGP4 source over an existing station schedule, preserving row
/// order. Reuses the checked GCRF propagation and prepared station rotations.
pub fn propagate_pointing_grid(
    source: &mut impl SGP4Source,
    geometry: &PreparedGeometry,
    elevation_mask_rad: f64,
) -> Result<Vec<PointingGeometry>> {
    check_mask(elevation_mask_rad)?;
    let states = propagate_sgp4_gcrf(source, geometry.timestamps())?;
    states
        .iter()
        .zip(geometry.iter_rows())
        .map(|(state, station)| pointing_geometry(&r_of(state), station, elevation_mask_rad))
        .collect()
}

struct PhaseSource {
    epoch: Instant,
    args: SGP4InitArgs,
    cache: Option<SatRec>,
}
impl PhaseSource {
    fn new(source: &Ephemeris, phase_s: f64) -> Result<Self> {
        let mut args = source
            .sgp4_init_args()
            .map_err(|e| crate::kernels::KernelError::Tle(e.to_string()))?;
        args.mo = (args.mo + args.no * phase_s / 60.0).rem_euclid(std::f64::consts::TAU);
        Ok(Self {
            epoch: source.epoch(),
            args,
            cache: None,
        })
    }
}
impl SGP4Source for PhaseSource {
    fn epoch(&self) -> Instant {
        self.epoch
    }
    fn satrec_mut(&mut self) -> &mut Option<SatRec> {
        &mut self.cache
    }
    fn sgp4_init_args(&self) -> satkit::sgp4::Result<SGP4InitArgs> {
        Ok(self.args)
    }
}

fn prepare_cycles(
    nominal: &mut Ephemeris,
    belief: &mut ParticleBelief,
    setup: &SearchSetup,
) -> Result<(EnuDirection, Vec<PlanningCycle>)> {
    let count = setup.config.cycle_count()?;
    let placeholder = EnuDirection::from_az_el(0.0, std::f64::consts::FRAC_PI_2)?;
    let timing: Vec<_> = (0..count)
        .map(|i| Action::at(setup.start, i, placeholder, &setup.config))
        .collect();
    let times: Vec<_> = timing.iter().map(Action::observation_time).collect();
    if times.windows(2).any(|w| w[0] >= w[1])
        || timing.iter().any(|a| a.listen_end <= a.listen_start)
    {
        return Err(AcquisitionError::InvalidInput(
            "cycle timing exceeds clock precision",
        ));
    }
    let station = PreparedGeometry::new(&[setup.start], &[setup.station])?;
    let initial = match setup.initial_direction {
        Some(direction) => direction,
        None => {
            propagate_pointing_grid(nominal, &station, setup.elevation_mask_rad)?[0].direction_enu
        }
    };
    if initial.elevation_rad() < setup.elevation_mask_rad {
        return Err(AcquisitionError::InvalidInput(
            "initial direction must be above mask",
        ));
    }
    let geometry = PreparedGeometry::new(&times, &vec![setup.station; count])?;
    let centers = propagate_pointing_grid(nominal, &geometry, setup.elevation_mask_rad)?;
    let lower = propagate_pointing_grid(
        &mut PhaseSource::new(nominal, -1.0)?,
        &geometry,
        setup.elevation_mask_rad,
    )?;
    let upper = propagate_pointing_grid(
        &mut PhaseSource::new(nominal, 1.0)?,
        &geometry,
        setup.elevation_mask_rad,
    )?;
    let mut cycles: Vec<_> = timing
        .into_iter()
        .map(|timing| PlanningCycle {
            timing,
            candidates: Vec::new(),
            particles: Vec::with_capacity(belief.len()),
        })
        .collect();
    for (index, orbit) in belief.originals_mut().iter_mut().enumerate() {
        let points = propagate_pointing_grid(orbit, &geometry, setup.elevation_mask_rad)
            .map_err(|e| particle_error(index, e))?;
        for (cycle, point) in cycles.iter_mut().zip(points) {
            cycle.particles.push(point);
        }
    }
    for (i, cycle) in cycles.iter_mut().enumerate() {
        let prior: Vec<_> = cycle.particles.iter().map(|p| p.direction_enu).collect();
        cycle.candidates = along_track_candidates(
            centers[i].direction_enu,
            &prior,
            upper[i].direction_enu.vector() - lower[i].direction_enu.vector(),
            setup.config.candidate_count,
            setup.config.candidate_extent_sigma,
        )?;
    }
    Ok((initial, cycles))
}
fn particle_error(index: usize, error: AcquisitionError) -> AcquisitionError {
    match error {
        AcquisitionError::Kernel(crate::kernels::KernelError::Tle(message)) => {
            AcquisitionError::Kernel(crate::kernels::KernelError::Tle(format!(
                "particle index {index}: {message}"
            )))
        }
        error => error,
    }
}

// Paper Eq. (8). Private: callers request BC-MPC, not a bootstrap policy.
fn sweep_order(count: usize) -> Vec<usize> {
    if count == 1 {
        return vec![0];
    }
    let middle = count / 2;
    (middle..count)
        .chain((0..count - 1).rev())
        .chain(1..middle)
        .collect()
}
fn bootstrap_pointing(
    previous: EnuDirection,
    candidates: &[EnuDirection],
    preferred: usize,
    setup: &SearchSetup,
) -> Result<(EnuDirection, Option<usize>)> {
    let target = candidates[preferred];
    let mut visible = Vec::new();
    let mut reachable = Vec::new();
    for (i, &direction) in candidates.iter().enumerate() {
        if direction.elevation_rad() < setup.elevation_mask_rad {
            continue;
        }
        visible.push(i);
        if setup
            .antenna
            .reachable(previous, direction, setup.config.move_s)?
        {
            reachable.push(i);
        }
    }
    let closest = |indices: &[usize]| {
        indices.iter().copied().min_by(|&i, &j| {
            target
                .separation(candidates[i])
                .total_cmp(&target.separation(candidates[j]))
        })
    };
    if let Some(index) = closest(&reachable) {
        return Ok((candidates[index], Some(index)));
    }
    if let Some(index) = closest(&visible) {
        let direction =
            setup
                .antenna
                .partial_slew(previous, candidates[index], setup.config.move_s)?;
        if direction.elevation_rad() >= setup.elevation_mask_rad {
            return Ok((direction, None));
        }
    }
    Ok((previous, None))
}
fn bootstrap(
    initial: EnuDirection,
    cycles: &[PlanningCycle],
    setup: &SearchSetup,
) -> Result<Vec<Action>> {
    let order = sweep_order(setup.config.candidate_count);
    let mut previous = initial;
    let mut actions = Vec::with_capacity(cycles.len());
    for (i, cycle) in cycles.iter().enumerate() {
        let (direction_enu, candidate_index) =
            bootstrap_pointing(previous, &cycle.candidates, order[i % order.len()], setup)?;
        actions.push(Action {
            direction_enu,
            candidate_index,
            ..cycle.timing
        });
        previous = direction_enu;
    }
    Ok(actions)
}

/// State for one finite contact. The caller owns actuator execution and its
/// clock; no background tasks, sleeps, receiver simulation, or policy switches.
#[derive(Debug)]
pub struct BcMpc {
    nominal: Ephemeris,
    belief: ParticleBelief,
    setup: SearchSetup,
    cycles: Vec<PlanningCycle>,
    direction: EnuDirection,
    cycle: usize,
    acquired: bool,
    plan: ActionPlan,
    pending_logs: Option<Vec<f64>>,
}
impl BcMpc {
    /// Prepare geometry and privately bootstrap a feasible continuation.
    /// Initial weights may be arbitrary finite nonnegative values with support.
    pub fn new(
        mut nominal: Ephemeris,
        mut belief: ParticleBelief,
        setup: SearchSetup,
    ) -> Result<Self> {
        setup.config.validate()?;
        setup.antenna.validate()?;
        check_mask(setup.elevation_mask_rad)?;
        if setup.start == Instant::INVALID {
            return Err(AcquisitionError::InvalidInput("search epoch"));
        }
        let (direction, cycles) = prepare_cycles(&mut nominal, &mut belief, &setup)?;
        let actions = bootstrap(direction, &cycles, &setup)?;
        Ok(Self {
            nominal,
            belief,
            setup,
            cycles,
            direction,
            cycle: 0,
            acquired: false,
            plan: ActionPlan {
                actions,
                acquisition_probability: 0.0,
                retained_acquisition_probability: 0.0,
            },
            pending_logs: None,
        })
    }
    pub fn belief(&self) -> &ParticleBelief {
        &self.belief
    }
    pub fn nominal(&self) -> &Ephemeris {
        &self.nominal
    }
    pub fn direction(&self) -> EnuDirection {
        self.direction
    }
    pub fn cycle(&self) -> usize {
        self.cycle
    }
    pub fn status(&self) -> SearchStatus {
        if self.acquired {
            SearchStatus::Acquired
        } else if self.cycle == self.cycles.len() {
            SearchStatus::Exhausted
        } else if self.pending_logs.is_some() {
            SearchStatus::AwaitingObservation
        } else {
            SearchStatus::Ready
        }
    }
    pub fn finished(&self) -> bool {
        matches!(
            self.status(),
            SearchStatus::Acquired | SearchStatus::Exhausted
        )
    }
    fn current_cycles(&self) -> Vec<PlanningCycle> {
        self.cycles[self.cycle..]
            .iter()
            .map(|cycle| PlanningCycle {
                timing: cycle.timing,
                candidates: cycle.candidates.to_vec(),
                particles: self
                    .belief
                    .indices()
                    .iter()
                    .map(|&i| cycle.particles[i])
                    .collect(),
            })
            .collect()
    }
    /// Return a borrowed complete plan; execute only its first action. All
    /// receiver evaluation must succeed before the plan becomes pending.
    /// Model evaluation may change between plans; geometry is cached, while
    /// likelihood caches are scoped to this call. Errors preserve controller state.
    pub fn plan(&mut self, model: &mut impl DetectionModel) -> Result<&ActionPlan> {
        if self.status() != SearchStatus::Ready {
            return Err(AcquisitionError::InvalidState(
                "plan requires a ready search",
            ));
        }
        let cycles = self.current_cycles();
        let config = self.setup.config;
        let settings = PlannerSettings {
            horizon: config.horizon,
            beam_width: config.beam_width,
            move_s: config.move_s,
            listen_s: config.listen_s,
            elevation_mask_rad: self.setup.elevation_mask_rad,
            antenna: self.setup.antenna,
        };
        let (plan, pending) = optimize_with_pending(
            self.direction,
            &self.plan.actions,
            &cycles,
            self.belief.weights(),
            &settings,
            model,
        )?;
        self.plan = plan;
        self.pending_logs = Some(pending);
        Ok(&self.plan)
    }
    /// Consume the completed first action. An execution/API failure is not a
    /// miss: leave this action pending and retry execution or abandon the search.
    /// `draw` is called once only if a miss requires systematic resampling.
    /// Unsupported misses or invalid draws leave all controller state intact.
    pub fn observe(
        &mut self,
        observation: Observation,
        mut draw: impl FnMut() -> f64,
    ) -> Result<ObservationUpdate> {
        let logs = self
            .pending_logs
            .as_ref()
            .ok_or(AcquisitionError::InvalidState(
                "observe requires a pending action",
            ))?;
        let resampled = if observation == Observation::Miss {
            let weights = posterior_weights(self.belief.weights(), logs)?;
            let (weights, indices, resampled) = self.updated_ensemble(weights, &mut draw)?;
            self.belief.commit(weights, indices);
            resampled
        } else {
            false
        };
        self.direction = self.plan.actions[0].direction_enu;
        self.plan.actions.remove(0);
        self.cycle += 1;
        self.acquired = observation == Observation::Acquired;
        self.pending_logs = None;
        Ok(ObservationUpdate {
            resampled,
            status: self.status(),
        })
    }
    fn updated_ensemble(
        &self,
        weights: Vec<f64>,
        draw: &mut impl FnMut() -> f64,
    ) -> Result<(Vec<f64>, Vec<usize>, bool)> {
        let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        if ess >= self.setup.config.ess_threshold * self.belief.len() as f64 {
            return Ok((weights, self.belief.indices().to_vec(), false));
        }
        let ancestors = systematic_resampling_indices(&weights, draw())?;
        let indices = ancestors
            .iter()
            .map(|&i| self.belief.indices()[i])
            .collect();
        Ok((
            vec![1.0 / weights.len() as f64; weights.len()],
            indices,
            true,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_atus_order_matches_paper() {
        assert_eq!(sweep_order(1), vec![0]);
        assert_eq!(sweep_order(5), vec![2, 3, 4, 3, 2, 1, 0, 1]);
        assert_eq!(sweep_order(17).len(), 32);
    }
}
