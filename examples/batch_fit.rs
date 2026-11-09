//! A noiseless synthetic LEO pass observed by two stations, fitted at the final epoch.
use std::error::Error;

use numeris::{DynVector, optim::LmSettings};
use satkit::{Duration, ITRFCoord, TLE, frametransform, orbitprop::SatPropertiesSimple};
use tychonic::{
    kernels::*,
    optim::{ODProblem, fit_lm},
};

// Satkit settings copies are explicit at this external API boundary.
#[allow(clippy::disallowed_methods)]
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let tle = TLE::load_2line(
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009",
        "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299350",
    )?;
    let settings = od_prop_settings();
    let cda_true = 0.05;
    let props = SatPropertiesSimple::new(cda_true, 0.0);
    let midpoint_state = initial_state_from_tle(&tle, &tle.epoch)?;
    let subsatellite = ITRFCoord::from_vector(
        &(frametransform::qgcrf2itrf_approx(&tle.epoch) * r_of(&midpoint_state)),
    );
    let (lat, lon, _) = subsatellite.to_geodetic_deg();
    let times: Vec<_> = (0..61)
        .map(|i| tle.epoch + Duration::from_seconds(-300.0 + 10.0 * i as f64))
        .collect();
    let reference_epoch = *times.last().unwrap();
    let truth = propagate_state(
        &midpoint_state,
        &tle.epoch,
        &reference_epoch,
        &settings,
        &props,
    )?;
    // Offset both sites to avoid exact zenith; alternate the observing station.
    let sites = [
        GroundStation {
            lat_deg: lat,
            lon_deg: lon + 0.2,
            alt_m: 100.0,
        },
        GroundStation {
            lat_deg: lat + 0.3,
            lon_deg: lon - 0.3,
            alt_m: 300.0,
        },
    ];
    let stations: Vec<_> = (0..times.len()).map(|i| sites[i % sites.len()]).collect();
    let (station, enu) = precompute_station_geometry(&times, &stations)?;
    let mut obs = Observations {
        range_rate: Vec::new(),
        azimuth: Vec::new(),
        elevation: Vec::new(),
        sigma_rr: 0.05,
        sigma_az: 1e-4,
        sigma_el: 1e-4,
    };
    // Independent endpoint propagations verify the dense batch implementation.
    for (i, t) in times.iter().enumerate() {
        let state = propagate_state(&truth, &reference_epoch, t, &settings, &props)?;
        let (rr, _) = range_rate_partial(
            &r_of(&state),
            &v_of(&state),
            &station.position[i],
            &station.velocity[i],
        )?;
        let ([az, el], _) =
            az_el_partial_with_rotation(&r_of(&state), &station.position[i], &enu.enu_to_gcrf[i])?;
        obs.range_rate.push(rr);
        obs.azimuth.push(az);
        obs.elevation.push(el);
    }
    let problem = ODProblem::new(
        PropagationConfig::new(
            reference_epoch,
            settings,
            ParameterMode::StateAndDrag { delta_cd: 1e-4 },
        )?,
        vec![PreparedBatch::combined(
            PreparedGeometry::from_trajectories(station, enu)?,
            obs,
        )?],
    )?;
    let mut guess = DynVector::zeros(7);
    for k in 0..6 {
        guess[k] = truth[k];
    }
    guess[0] += 30.0;
    guess[1] -= 25.0;
    guess[2] += 15.0;
    guess[3] += 0.02;
    guess[6] = cda_true + 0.001;
    let initial = problem.evaluate(&guess)?;
    println!(
        "Initial RSS = {:.6}, Jacobian = {} x {}",
        initial.rss(),
        initial.jacobian.nrows(),
        initial.jacobian.ncols()
    );
    let result = fit_lm(
        &problem,
        &guess,
        None,
        &LmSettings {
            grad_tol: 1e-6,
            x_tol: 1e-9,
            f_tol: 1e-10,
            ..LmSettings::default()
        },
    )?;
    let optimum = result.parameters;
    let pos_error = (0..3)
        .map(|k| (optimum[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    let vel_error = (3..6)
        .map(|k| (optimum[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    println!(
        "Final RSS = {:.6} after {} iterations",
        result.rss, result.iterations
    );
    println!("Position error = {pos_error:.6} m; velocity error = {vel_error:.8} m/s");
    println!("C_d A/m = {:.8} m²/kg (truth {cda_true:.8})", optimum[6]);
    Ok(())
}
