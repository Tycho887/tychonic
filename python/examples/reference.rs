//! Regenerate Python parity fixtures using the Rust API, without Python bindings.
#![allow(clippy::disallowed_methods)] // Independent copies of regression inputs, as in core tests.
use serde_json::json;
use tychonic::{
    detection::{
        BcMpc, DetectionModel, EnuDirection, Ephemeris, ParticleBelief, PointingGeometry,
        SearchSetup,
    },
    estimate::{OrbitEkf, TcaUkf, UpdateOptions, tca_measurement_noise, white_acceleration_noise},
    kernels::*,
    measurements::{Observable, joint},
    numeris::{DynMatrix, DynVector, Matrix, Vector, optim::LmSettings},
    satkit::{Duration, Instant, TLE, omm::OMM},
};

fn rows<const N: usize>(p: &Matrix<f64, N, N>) -> Vec<Vec<f64>> {
    (0..N)
        .map(|i| (0..N).map(|j| p[(i, j)]).collect())
        .collect()
}
fn config(epoch: Instant, drag: bool) -> PropagationConfig {
    let mut settings = od_prop_settings();
    settings.use_spaceweather = false; // Keep fixtures independent of refreshed weather tables.
    PropagationConfig::new(
        epoch,
        settings,
        if drag {
            ParameterMode::StateAndDrag { delta_cd: 1e-4 }
        } else {
            ParameterMode::StateOnly { fixed_drag: 0.02 }
        },
    )
    .unwrap()
}
fn physical(state: &V6, g: &StationGeometry, time: Instant, baseline: Interferometer) -> Vec<f64> {
    let rr = range_rate_partial(&r_of(state), &v_of(state), g.position(), g.velocity())
        .unwrap()
        .0;
    let phase = baseline
        .evaluate(state, g, &v3(59.0, 3.0, 0.2), 0.0)
        .unwrap()
        .phase_rad;
    let differential = baseline
        .differential(state, g, &v3(59.0, 3.0, 0.2), time)
        .unwrap()
        .difference_hz;
    let angles = az_el_partial_prepared(&r_of(state), g.position(), g.rotation())
        .unwrap()
        .0;
    vec![
        -2.23e9 / SPEED_OF_LIGHT * rr,
        phase,
        differential,
        angles[0],
        angles[1],
    ]
}
fn orbit_reference(drag: bool) -> serde_json::Value {
    let epoch = Instant::from_rfc3339("2024-01-01T12:00:00.250000Z").unwrap();
    let station = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let times: Vec<_> = (0..7)
        .map(|i| epoch + Duration::from_seconds(i as f64 * 10.0))
        .collect();
    let geometry = PreparedGeometry::new(&times, &vec![station; times.len()]).unwrap();
    let baseline = Interferometer::new(2.23e9).unwrap();
    let mut truth = vec![7e6, 0.0, 1e5, 0.0, 7400.0, 100.0];
    if drag {
        truth.push(0.02);
    }
    let samples = config(epoch, drag)
        .sample(&DynVector::from_slice(&truth), &times)
        .unwrap();
    let values: Vec<_> = (0..times.len())
        .map(|i| {
            physical(
                &samples.states()[i],
                geometry.row(i).unwrap(),
                times[i],
                baseline,
            )
        })
        .collect();
    let sigmas = [0.5, 0.01, 0.7, 1e-4, 2e-4];
    let covariance = DynMatrix::from_fn(5, 5, |i, j| {
        sigmas[i] * sigmas[j] * if i == j { 1.0 } else { 0.15 }
    });
    let models = vec![
        Observable::Doppler {
            carrier_hz: 2.23e9,
            offset_hz: 0.0,
        },
        Observable::PhaseDifference {
            interferometer: baseline,
            baseline_enu_m: vec![v3(59.0, 3.0, 0.2); times.len()],
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        },
        Observable::DifferentialDoppler {
            interferometer: baseline,
            baseline_enu_m: vec![v3(59.0, 3.0, 0.2); times.len()],
        },
        Observable::AzimuthElevation,
    ];
    let batch = joint(
        geometry,
        models,
        values.clone(),
        vec![covariance.clone(); times.len()],
    )
    .unwrap();
    let mut x0 = truth.clone();
    for (x, dx) in x0
        .iter_mut()
        .zip([25.0, -30.0, 20.0, 0.02, -0.01, 0.03, 0.001])
    {
        *x += dx;
    }
    let fit = tychonic::fit_with_options(
        config(epoch, drag),
        std::slice::from_ref(&batch),
        &DynVector::from_slice(&x0),
        None,
        &LmSettings {
            max_iter: 300,
            ..LmSettings::default()
        },
    )
    .unwrap();
    let covariance_rows: Vec<_> = (0..5)
        .map(|i| (0..5).map(|j| covariance[(i, j)]).collect::<Vec<_>>())
        .collect();
    let filter = if drag {
        ekf::<7>(config(epoch, true), &x0, &batch)
    } else {
        ekf::<6>(config(epoch, false), &x0, &batch)
    };
    json!({"epoch":epoch.as_rfc3339(),"times":times.iter().map(Instant::as_rfc3339).collect::<Vec<_>>(),
        "truth":truth,"x0":x0,"values":values,"covariance":covariance_rows,
        "fit":{"x":fit.parameters.as_slice(),"rss":fit.rss,"nit":fit.iterations},"filter":filter})
}
fn ekf<const N: usize>(
    config: PropagationConfig,
    x0: &[f64],
    batch: &PreparedBatch,
) -> serde_json::Value {
    let prior = Matrix::from_fn(|i, j| {
        if i != j {
            0.0
        } else if i < 3 {
            1e4
        } else if i < 6 {
            0.01
        } else {
            1e-4
        }
    });
    let mut filter = OrbitEkf::<N>::new(config, Vector::from_slice(x0), prior).unwrap();
    let updates = filter
        .filter_batches(
            std::slice::from_ref(batch),
            |_, a, b| {
                white_acceleration_noise(
                    (b - a).as_seconds(),
                    &v3(1e-6, 1e-6, 1e-6),
                    if N == 7 { 1e-10 } else { 0.0 },
                )
            },
            UpdateOptions::default(),
        )
        .unwrap();
    let smooth = filter.smooth().unwrap();
    json!({"state":filter.state().as_slice(),"covariance":rows(filter.covariance()),
        "nis":updates.iter().map(|u|u.nis).collect::<Vec<_>>(),
        "states":filter.steps().iter().map(|s|s.x_updated.as_slice().to_vec()).collect::<Vec<_>>(),
        "smooth_states":smooth.iter().map(|(x,_)|x.as_slice().to_vec()).collect::<Vec<_>>(),
        "smooth_covariances":smooth.iter().map(|(_,p)|rows(p)).collect::<Vec<_>>()})
}
const TLE_TEXT: &str = "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992\n2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617";
fn tca_reference() -> serde_json::Value {
    let lines: Vec<_> = TLE_TEXT.lines().collect();
    let prior = TLE::load_2line(lines[0], lines[1]).unwrap();
    let start = prior.epoch + Duration::from_seconds(300.125);
    let station = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let mut truth = prior.clone();
    truth.epoch += Duration::from_seconds(5.0);
    let initial = Matrix::new([[22500.0, 0.0], [0.0, 40000.0]]);
    let mut filter = TcaUkf::new(prior, start, Vector::zeros(), initial).unwrap();
    let mut readings = Vec::new();
    for i in 0..8 {
        let time = start + Duration::from_seconds(i as f64 * 10.0);
        let state = initial_state_from_tle(&truth, &time).unwrap();
        let geometry = PreparedGeometry::new(&[time], &[station]).unwrap();
        let g = geometry.row(0).unwrap();
        let rr = range_rate_partial(&r_of(&state), &v_of(&state), g.position(), g.velocity())
            .unwrap()
            .0
            + 17.0;
        let angles = az_el_partial_prepared(&r_of(&state), g.position(), g.rotation())
            .unwrap()
            .0;
        let measurement = Measurement {
            time,
            station,
            range_rate: rr,
            azimuth: angles[0],
            elevation: angles[1],
        };
        let update = filter
            .observe(
                &measurement,
                &tca_measurement_noise(0.5).unwrap(),
                None,
                UpdateOptions::default(),
            )
            .unwrap();
        readings.push(
            json!({"time":time.as_rfc3339(),"values":[rr,angles[0],angles[1]],"nis":update.nis}),
        );
    }
    json!({"tle":TLE_TEXT,"start":start.as_rfc3339(),"readings":readings,"state":filter.state().as_slice(),
        "covariance":rows(filter.covariance()),"corrected_omm":OMM::from_tle(&filter.corrected_tle().unwrap())})
}
struct ZeroReceiver;
impl DetectionModel for ZeroReceiver {
    fn probabilities(
        &mut self,
        _: EnuDirection,
        _: &[PointingGeometry],
        _: f64,
        output: &mut [f64],
    ) -> tychonic::detection::Result<()> {
        output.fill(0.0);
        Ok(())
    }
}
fn acquisition_reference() -> serde_json::Value {
    let lines: Vec<_> = TLE_TEXT.lines().collect();
    let nominal = TLE::load_2line(lines[0], lines[1]).unwrap();
    let start = nominal.epoch + Duration::from_seconds(600.0);
    let state = initial_state_from_tle(&nominal, &start).unwrap();
    let local =
        tychonic::satkit::frametransform::qitrf2gcrf_approx(&start).conjugate() * r_of(&state);
    let (lat, lon, _) = tychonic::satkit::ITRFCoord::from_vector(&local).to_geodetic_rad();
    let station = GroundStation {
        lat_deg: lat.to_degrees() + 2.5,
        lon_deg: lon.to_degrees(),
        alt_m: 0.0,
    };
    let particles: Vec<_> = [-10.0, 0.0, 10.0]
        .into_iter()
        .map(|dt| {
            let mut t = nominal.clone();
            t.epoch += Duration::from_seconds(dt);
            OMM::from_tle(&t)
        })
        .collect();
    let mut setup = SearchSetup::new(station, start);
    setup.config.search_duration_s = 18.0;
    setup.config.horizon = 2;
    setup.config.beam_width = 5;
    setup.config.candidate_count = 5;
    let belief = ParticleBelief::new(
        particles
            .iter()
            .map(|p| Ephemeris::from(p.clone()))
            .collect(),
        vec![1.0; 3],
    )
    .unwrap();
    let mut controller = BcMpc::new(nominal.into(), belief, setup).unwrap();
    let plan = controller.plan(&mut ZeroReceiver).unwrap();
    json!({"tle":TLE_TEXT,"particles":particles,"start":start.as_rfc3339(),"station":[station.lat_deg,station.lon_deg,station.alt_m],
        "directions":plan.actions.iter().map(|a|a.direction_enu.vector().as_slice().to_vec()).collect::<Vec<_>>(),
        "listen_starts":plan.actions.iter().map(|a|a.listen_start.as_rfc3339()).collect::<Vec<_>>(),"probability":plan.acquisition_probability})
}
fn main() {
    println!("{}",serde_json::to_string_pretty(&json!({"orbit":orbit_reference(false),"drag":orbit_reference(true),"tca":tca_reference(),"acquisition":acquisition_reference()})).unwrap());
}
