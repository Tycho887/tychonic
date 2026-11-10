use super::{AcquisitionError, Result, probability};
use satkit::{
    Instant, TLE,
    omm::OMM,
    sgp4::{SGP4InitArgs, SGP4Source, SatRec},
};

/// Owned Satkit element set; particles may mix TLEs and OMMs.
#[derive(Debug)]
// Small owned ensembles avoid per-particle allocations; resampling only copies indices.
#[allow(clippy::large_enum_variant)]
pub enum Ephemeris {
    Tle(TLE),
    Omm(OMM),
}
impl From<TLE> for Ephemeris {
    fn from(value: TLE) -> Self {
        Self::Tle(value)
    }
}
impl From<OMM> for Ephemeris {
    fn from(value: OMM) -> Self {
        Self::Omm(value)
    }
}
impl SGP4Source for Ephemeris {
    fn epoch(&self) -> Instant {
        match self {
            Self::Tle(x) => x.epoch(),
            Self::Omm(x) => x.epoch(),
        }
    }
    fn satrec_mut(&mut self) -> &mut Option<SatRec> {
        match self {
            Self::Tle(x) => x.satrec_mut(),
            Self::Omm(x) => x.satrec_mut(),
        }
    }
    fn sgp4_init_args(&self) -> satkit::sgp4::Result<SGP4InitArgs> {
        match self {
            Self::Tle(x) => x.sgp4_init_args(),
            Self::Omm(x) => x.sgp4_init_args(),
        }
    }
}

/// Explicit weighted orbit hypotheses. Resampling indexes the immutable
/// original pool; `indices()[i]` identifies the orbit for `weights()[i]`.
#[derive(Debug)]
pub struct ParticleBelief {
    originals: Vec<Ephemeris>,
    indices: Vec<usize>,
    weights: Vec<f64>,
}
impl ParticleBelief {
    pub fn new(particles: Vec<Ephemeris>, weights: Vec<f64>) -> Result<Self> {
        if particles.is_empty() || particles.len() != weights.len() {
            return Err(AcquisitionError::InvalidInput(
                "nonempty particles and matching weights required",
            ));
        }
        let weights = normalize_weights(&weights)?;
        Ok(Self {
            indices: (0..particles.len()).collect(),
            originals: particles,
            weights,
        })
    }
    pub fn len(&self) -> usize {
        self.weights.len()
    }
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
    pub fn particles(&self) -> &[Ephemeris] {
        &self.originals
    }
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn effective_sample_size(&self) -> f64 {
        1.0 / self.weights.iter().map(|w| w * w).sum::<f64>()
    }
    /// Apply Eq. (3) atomically. Values are miss probabilities, not detections.
    pub fn update_miss(&mut self, misses: &[f64]) -> Result<()> {
        let logs = misses
            .iter()
            .map(|&p| {
                probability(p)?;
                Ok(p.ln())
            })
            .collect::<Result<Vec<_>>>()?;
        self.weights = posterior_weights(&self.weights, &logs)?;
        Ok(())
    }
    /// Systematic resampling below threshold × N. Draw is called only when
    /// needed, and must return a finite uniform value in [0, 1).
    pub fn resample_if_needed(
        &mut self,
        threshold: f64,
        mut draw: impl FnMut() -> f64,
    ) -> Result<bool> {
        probability(threshold)?;
        if self.effective_sample_size() >= threshold * self.len() as f64 {
            return Ok(false);
        }
        let positions = systematic_resampling_indices(&self.weights, draw())?;
        self.indices = positions.iter().map(|&i| self.indices[i]).collect();
        let uniform_weight = 1.0 / self.len() as f64;
        self.weights.fill(uniform_weight);
        Ok(true)
    }
    pub(crate) fn originals_mut(&mut self) -> &mut [Ephemeris] {
        &mut self.originals
    }
    pub(crate) fn commit(&mut self, weights: Vec<f64>, indices: Vec<usize>) {
        self.weights = weights;
        self.indices = indices;
    }
}

/// Normalize arbitrary finite nonnegative weights without sum overflow.
pub fn normalize_weights(weights: &[f64]) -> Result<Vec<f64>> {
    if weights.is_empty() || weights.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(AcquisitionError::InvalidInput(
            "finite nonnegative weights required",
        ));
    }
    let maximum = weights.iter().copied().fold(0.0, f64::max);
    if maximum == 0.0 {
        return Err(AcquisitionError::InvalidInput(
            "positive weight support required",
        ));
    }
    let scaled: Vec<_> = weights.iter().map(|w| w / maximum).collect();
    let sum: f64 = scaled.iter().sum();
    Ok(scaled.into_iter().map(|w| w / sum).collect())
}

pub(crate) fn posterior_weights(weights: &[f64], log_misses: &[f64]) -> Result<Vec<f64>> {
    if weights.len() != log_misses.len() {
        return Err(AcquisitionError::InvalidInput("likelihood length"));
    }
    let logs: Vec<_> = weights
        .iter()
        .zip(log_misses)
        .map(|(w, p)| w.ln() + p)
        .collect();
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Err(AcquisitionError::ImpossibleObservation);
    }
    normalize_weights(&logs.iter().map(|x| (x - maximum).exp()).collect::<Vec<_>>())
}

/// Return systematic ancestor indices for arbitrary valid weights.
/// `uniform` is the single uniform draw in [0, 1), before division by N.
pub fn systematic_resampling_indices(weights: &[f64], uniform: f64) -> Result<Vec<usize>> {
    if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
        return Err(AcquisitionError::InvalidInput(
            "resampling draw must lie in [0, 1)",
        ));
    }
    let weights = normalize_weights(weights)?;
    let mut indices = Vec::with_capacity(weights.len());
    let mut index = 0;
    let mut cumulative = weights[0];
    for i in 0..weights.len() {
        let position = ((i as f64 + uniform) / weights.len() as f64).min(1.0 - f64::EPSILON);
        while index + 1 < weights.len() && cumulative <= position {
            index += 1;
            cumulative += weights[index];
        }
        indices.push(index);
    }
    Ok(indices)
}
