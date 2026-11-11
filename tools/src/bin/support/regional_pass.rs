//! Synthetic polar pass with simultaneous visibility at two specified sites.
use super::{AnyResult, Random, output};
use numeris::DynVector;
use satkit::{Duration, ITRFCoord, Instant, frametransform};
use serde_json::json;
use std::{fs, io::Write, path::Path};
use tychonic::kernels::*;
use tychonic::{numeris, satkit};

pub const TROMSO: GroundStation = GroundStation {
    lat_deg: 69.662,
    lon_deg: 18.940,
    alt_m: 100.0,
};
pub const SVALBARD: GroundStation = GroundStation {
    lat_deg: 78.230,
    lon_deg: 15.410,
    alt_m: 450.0,
};

fn elevation(x: &V6, g: &StationGeometry) -> f64 {
    let local = g.rotation().to_enu(&(r_of(x) - g.position()));
    local[2].atan2(local[0].hypot(local[1])).to_degrees()
}

/// Write phase_pass-compatible inputs, without fitting the preliminary data.
/// Stations are illustrative nominal coordinates, not surveyed phase centers.
pub fn generate(path: &Path, trials: usize) -> AnyResult<()> {
    let midpoint = Instant::from_datetime(2024, 1, 1, 12, 0, 0.0)?;
    let lat = (TROMSO.lat_deg + SVALBARD.lat_deg) / 2.0;
    let lon = (TROMSO.lon_deg + SVALBARD.lon_deg) / 2.0;
    let r = frametransform::qitrf2gcrf_approx(&midpoint)
        * ITRFCoord::from_geodetic_deg(lat, lon, 600e3).itrf;
    let up = r / r.norm();
    let east = v3(0.0, 0.0, 1.0).cross(&up).normalize();
    let north = up.cross(&east);
    let east_fraction = 97.6_f64.to_radians().cos() / (1.0 - up[2].powi(2)).sqrt();
    let v = (east * east_fraction + north * (1.0 - east_fraction.powi(2)).sqrt())
        * (satkit::consts::MU_EARTH / r.norm()).sqrt();
    let x = v6_from(&r, &v);
    let cfg = PropagationConfig::new(
        midpoint,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: 0.05 },
    )?;
    let all_times: Vec<_> = (0..=90)
        .map(|i| midpoint + Duration::from_seconds(-450.0 + i as f64 * 10.0))
        .collect();
    let states = cfg
        .sample(&DynVector::from_slice(x.as_slice()), &all_times)?
        .states()
        .to_vec();
    let first = PreparedGeometry::new(&all_times, &vec![TROMSO; all_times.len()])?;
    let second = PreparedGeometry::new(&all_times, &vec![SVALBARD; all_times.len()])?;
    let selected: Vec<_> = (0..all_times.len())
        .filter(|&i| {
            elevation(&states[i], first.row(i).unwrap()) >= 10.0
                && elevation(&states[i], second.row(i).unwrap()) >= 10.0
        })
        .collect();
    if selected.len() < 8 {
        return Err("fewer than eight simultaneously visible observations".into());
    }
    let times: Vec<_> = selected.iter().map(|&i| all_times[i]).collect();
    let truth: Vec<_> = selected.iter().map(|&i| states[i]).collect();
    let epoch = times[0];
    let geometry = PreparedGeometry::new(&times, &vec![TROMSO; times.len()])?;
    let baseline_enu_m = v3(59.0, 0.0, 0.0);
    let int = Interferometer::new(2.23e9)?;
    let ou = OuPhaseNoise::new(21600.0, 5_f64.to_radians())?;
    fs::create_dir_all(path)?;
    let source = json!({
        "trials":trials,"seed":20261103,"tau_hours":6,"stationary_drift_deg":5,
        "baseline_enu_m":[59,0,0],"carrier_hz":2.23e9,"sigma_rr_m_s":0.05,
        "cadence_seconds":10,"window_seconds":900,"min_elevation_deg":10,
        "retained_samples":times.len(),"retained_duration_seconds":(times.last().unwrap()-epoch).as_seconds(),
        "reference_epoch":epoch.to_string(),"station_lat_deg":TROMSO.lat_deg,
        "station_lon_deg":TROMSO.lon_deg,"station_alt_m":TROMSO.alt_m,
        "phase_noise_deg":[0.1,1,5,10],"fixed_drag_m2_kg":0.05,
        "prior_sigmas":[1000,1000,1000,1,1,1,10,std::f64::consts::PI],
        "scenario":"tromso-svalbard","orbit":"hypothetical initially circular 600 km polar LEO, inclination 97.6 degrees; midpoint above station-pair midpoint",
        "station_coordinates":"illustrative nominal coordinates, not surveyed antenna phase centers",
        "remote_station":{"name":"Svalbard (near Longyearbyen)","lat_deg":SVALBARD.lat_deg,"lon_deg":SVALBARD.lon_deg,"alt_m":SVALBARD.alt_m},
        "visibility":"both stations above 10 degrees; identical shared schedule in all comparison cases",
        "phase_convention":"continuous; local baseline antenna1 minus antenna2; frequency relative to carrier"
    });
    fs::write(
        path.join("config.json"),
        serde_json::to_string_pretty(&source)?,
    )?;
    let mut truth_file = output(
        path.join("truth.csv"),
        "seconds,x,y,z,vx,vy,vz,elevation_deg,remote_elevation_deg",
    )?;
    for (&i, state) in selected.iter().zip(&truth) {
        write!(truth_file, "{}", (all_times[i] - epoch).as_seconds())?;
        for value in state.as_slice() {
            write!(truth_file, ",{value:.16e}")?;
        }
        writeln!(
            truth_file,
            ",{:.9},{:.9}",
            elevation(state, first.row(i).unwrap()),
            elevation(state, second.row(i).unwrap())
        )?;
    }
    let mut priors = output(path.join("priors.csv"), "trial,seed,index,mean,sigma")?;
    let mut observations = output(
        path.join("observations.csv"),
        "trial,seed,phase_sigma_deg,seconds,doppler_hz,phase_rad,true_frequency_bias_hz,true_phase_mean_rad,true_drift_rad",
    )?;
    let n = times.len();
    let sigmas = [
        1000.0,
        1000.0,
        1000.0,
        1.0,
        1.0,
        1.0,
        10.0,
        std::f64::consts::PI,
    ];
    for trial in 0..trials {
        let seed = 20261103_u64.wrapping_add((trial as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let mut rng = Random(seed);
        for k in 0..8 {
            let mean = if k < 6 {
                truth[0][k] + sigmas[k] * rng.normal()
            } else {
                0.0
            };
            writeln!(priors, "{trial},{seed},{k},{mean:.16e},{:.16e}", sigmas[k])?;
        }
        let bias = 10.0 * rng.normal();
        let mean = std::f64::consts::PI * rng.normal();
        let mut drift = vec![ou.stationary_sigma_rad() * rng.normal(); n];
        for i in 1..n {
            let (a, q) = ou.transition((times[i] - times[i - 1]).as_seconds())?;
            drift[i] = a * drift[i - 1] + q.sqrt() * rng.normal();
        }
        let zf: Vec<_> = (0..n).map(|_| rng.normal()).collect();
        let zp: Vec<_> = (0..n).map(|_| rng.normal()).collect();
        for level in [0.1_f64, 1.0, 5.0, 10.0] {
            for i in 0..n {
                let pred =
                    int.evaluate(&truth[i], geometry.row(i).unwrap(), &baseline_enu_m, bias)?;
                writeln!(
                    observations,
                    "{trial},{seed},{level},{},{:.16e},{:.16e},{bias:.16e},{mean:.16e},{:.16e}",
                    (times[i] - epoch).as_seconds(),
                    pred.doppler_hz + 0.05 * 2.23e9 / SPEED_OF_LIGHT * zf[i],
                    pred.phase_rad + mean + drift[i] + level.to_radians() * zp[i],
                    drift[i]
                )?;
            }
        }
    }
    println!(
        "Tromsø–Svalbard: {} shared samples over {:.0} s; baseline {:.1} km",
        n,
        (times.last().unwrap() - epoch).as_seconds(),
        TROMSO.baseline_enu_to(&SVALBARD)?.norm() / 1000.0
    );
    Ok(())
}
