//! Native reference and timings for the Python two-contact integration test.
//! Read the checked-in measurements and emit reference values on stdout.
use serde::Deserialize;
use serde_json::json;
use std::{error::Error, time::Instant as Clock};
use tychonic::{
    estimate::{OrbitEkf, UpdateOptions},
    kernels::*,
    measurements::{Observable, joint},
    numeris::{DynMatrix, DynVector, Matrix, optim::LmSettings},
    optim::{ODProblem, fit_lm_with_covariance},
    satkit::Instant,
};

#[derive(Deserialize)]
struct Input {
    reference_epoch_unix_s: f64,
    initial_state_gcrf: Vec<f64>,
    fixed_drag_m2_kg: f64,
    carrier_hz: f64,
    observations: Vec<Reading>,
    max_iterations: usize,
    gradient_tolerance: f64,
    step_tolerance: f64,
    cost_tolerance: f64,
}
#[derive(Deserialize)]
struct Site {
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
}
#[derive(Deserialize)]
struct Reading {
    epoch_unix_s: f64,
    station: Site,
    baseline_enu_m: [f64; 3],
    doppler_hz: f64,
    phase_difference_rad: f64,
    differential_doppler_hz: f64,
    azimuth_rad: f64,
    elevation_rad: f64,
    measurement_covariance: [[f64; 5]; 5],
}
fn batch(data: &[Reading], carrier: f64) -> Result<PreparedBatch<'static>> {
    let times: Vec<_> = data
        .iter()
        .map(|r| Instant::from_unixtime(r.epoch_unix_s))
        .collect();
    let sites: Vec<_> = data
        .iter()
        .map(|r| GroundStation {
            lat_deg: r.station.lat_deg,
            lon_deg: r.station.lon_deg,
            alt_m: r.station.alt_m,
        })
        .collect();
    let baselines = || {
        data.iter()
            .map(|r| V3::from_array(r.baseline_enu_m))
            .collect()
    };
    let sensor = Interferometer::new(carrier)?;
    let models = vec![
        Observable::Doppler {
            carrier_hz: carrier,
            offset_hz: 0.,
        },
        Observable::PhaseDifference {
            interferometer: sensor,
            baseline_enu_m: baselines(),
            frequency_offset_hz: 0.,
            phase_offset_rad: 0.,
        },
        Observable::DifferentialDoppler {
            interferometer: sensor,
            baseline_enu_m: baselines(),
        },
        Observable::AzimuthElevation,
    ];
    let values = data
        .iter()
        .map(|r| {
            vec![
                r.doppler_hz,
                r.phase_difference_rad,
                r.differential_doppler_hz,
                r.azimuth_rad,
                r.elevation_rad,
            ]
        })
        .collect();
    let noise = data
        .iter()
        .map(|r| DynMatrix::from_fn(5, 5, |i, j| r.measurement_covariance[i][j]))
        .collect();
    joint(
        PreparedGeometry::new(&times, &sites)?,
        models,
        values,
        noise,
    )
}
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let fixture: Input = serde_json::from_str(include_str!("../tests/data/two_contacts.json"))?;
    let c = fixture.carrier_hz;
    let reference = Instant::from_unixtime(fixture.reference_epoch_unix_s);
    let mode = ParameterMode::StateOnly {
        fixed_drag: fixture.fixed_drag_m2_kg,
    };
    let cfg = |t| {
        let mut settings = od_prop_settings();
        settings.use_spaceweather = false;
        PropagationConfig::new(t, settings, mode)
    };
    let rows = &fixture.observations;
    let start = Clock::now();
    let history = vec![batch(&rows[..21], c)?, batch(&rows[21..39], c)?];
    let live = batch(&rows[39..], c)?;
    let preprocessing = start.elapsed();
    let problem = ODProblem::new(cfg(reference)?, history)?;
    let settings = LmSettings {
        max_iter: fixture.max_iterations,
        grad_tol: fixture.gradient_tolerance,
        x_tol: fixture.step_tolerance,
        f_tol: fixture.cost_tolerance,
        ..Default::default()
    };
    let start = Clock::now();
    let fit = fit_lm_with_covariance(
        &problem,
        &DynVector::from_vec(fixture.initial_state_gcrf),
        None,
        &settings,
    )?;
    let publish = problem.timestamps()[problem.timestamps().len() - 1];
    let samples = problem.config().sample(&fit.parameters, &[publish])?;
    let phi = samples.stm()[0];
    let p0 = Matrix::<f64, 6, 6>::from_fn(|i, j| fit.covariance[(i, j)]);
    let covariance = phi * p0 * phi.transpose();
    let solve = start.elapsed();
    let mut x = V7::zeros();
    x.set_block(0, 0, &samples.states()[0]);
    x[6] = -0.5;
    let mut p = M7::zeros();
    p.set_block(0, 0, &((covariance + covariance.transpose()) * 0.5));
    p[(6, 6)] = 25.;
    let mut filter = OrbitEkf::with_doppler_bias(cfg(publish)?, x, p)?;
    let start = Clock::now();
    for row in 0..3 {
        filter.observe(
            &live.joint_observation::<5>(row)?,
            None,
            UpdateOptions::default(),
        )?;
    }
    let update = start.elapsed();
    eprintln!(
        "prepare {:.3} ms, solve and transport {:.3} ms, three updates {:.3} ms",
        preprocessing.as_secs_f64() * 1000.,
        solve.as_secs_f64() * 1000.,
        update.as_secs_f64() * 1000.
    );
    println!(
        "{}",
        json!({"rss":fit.rss,"published_state":samples.states()[0].as_slice(),"published_covariance":covariance.as_slice(),"final_state":filter.state().as_slice(),"final_covariance":filter.covariance().as_slice()})
    );
    Ok(())
}
