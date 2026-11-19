//! Deterministic station passes with known directions and constant instrument biases.
use tychonic::{
    kernels::*,
    numeris::DynVector,
    optim::{ContactId, ContactRangeRate},
    satkit::{Duration, ITRFCoord, Instant, frametransform},
};
use tychonic_constrained::*;

pub struct Fixture {
    pub truth: V7,
    pub seed: V7,
    pub rates: Vec<ContactRangeRate>,
    pub angles: Vec<ContactPointing>,
    pub biases: [f64; 3],
    pub sigmas: Vec<f64>,
    pub epoch: Instant,
    span: f64,
}

impl Fixture {
    pub fn new() -> Self {
        Self::with_centers([-4600., 0., 4800.])
    }

    pub fn with_centers(offsets: [f64; 3]) -> Self {
        let epoch = Instant::from_date(2024, 1, 1).unwrap();
        let truth = V7::from_slice(&[6.65e6, 2e4, -4e4, 0., 6800., 3660., 0.02]);
        let seed = truth + V7::from_slice(&[80., -90., 60., -0.06, 0.07, 0.05, -0.003]);
        let span = offsets.iter().copied().map(f64::abs).fold(0., f64::max) + 300.;
        let centers: Vec<_> = offsets
            .iter()
            .map(|&t| epoch + Duration::from_seconds(t))
            .collect();
        let params = DynVector::from_slice(truth.as_slice());
        let cfg = config_with_span(epoch, span);
        let center_states = cfg.sample(&params, &centers).unwrap();
        let mut times = Vec::new();
        let mut sites = Vec::new();
        for (&center, state) in centers.iter().zip(center_states.states()) {
            let fixed = frametransform::qgcrf2itrf(&center) * r_of(state);
            let subpoint = ITRFCoord::from_vector(&fixed);
            let site = GroundStation {
                lat_deg: subpoint.latitude_deg() + 0.8,
                lon_deg: subpoint.longitude_deg() - 0.4,
                alt_m: 200.,
            };
            for i in -8..=8 {
                times.push(center + Duration::from_seconds(i as f64 * 15.));
                sites.push(site);
            }
        }
        let samples = cfg.sample(&params, &times).unwrap();
        let geometry = PreparedGeometry::new(&times, &sites).unwrap();
        let mut rates = Vec::new();
        let mut angles = Vec::new();
        let mut sigmas = Vec::new();
        let biases = [0.24, -0.31, 0.08];
        for (i, site) in geometry.iter_rows().enumerate() {
            let state = &samples.states()[i];
            let (rr, _) =
                range_rate_partial(&r_of(state), &v_of(state), site.position(), site.velocity())
                    .unwrap();
            let local = site.rotation().to_enu(&(r_of(state) - site.position()));
            let contact = ContactId::new((i / 17) as u128);
            let time = times[i];
            let station = sites[i];
            rates.push(ContactRangeRate {
                time,
                station,
                contact,
                range_rate: rr + biases[i / 17],
            });
            sigmas.push(0.08 + 0.02 * (i % 3) as f64);
            angles.push(ContactPointing {
                time,
                station,
                contact,
                azimuth: local[0].atan2(local[1]),
                elevation: local[2].atan2(local[0].hypot(local[1])),
            });
        }
        Fixture {
            truth,
            seed,
            rates,
            angles,
            biases,
            span,
            sigmas,
            epoch,
        }
    }

    pub fn model(&self, selection: PointingSelection, fov: f64) -> ConstrainedRangeRateProblem {
        ConstrainedRangeRateProblem::new(
            config_with_span(self.epoch, self.span),
            self.seed,
            &self.rates,
            &self.sigmas,
            &self.angles,
            fov,
            selection,
        )
        .unwrap()
    }
}
pub fn config_with_span(epoch: Instant, seconds: f64) -> PropagationConfig {
    let mut settings = od_prop_settings();
    let span = Duration::from_seconds(seconds);
    settings
        .precompute_terms(&(epoch - span), &(epoch + span))
        .unwrap();
    PropagationConfig::new(
        epoch,
        settings,
        ParameterMode::StateAndDrag { delta_cd: 1e-4 },
    )
    .unwrap()
}
