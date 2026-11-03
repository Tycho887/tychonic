//! Reproducible paired Monte Carlo study of Doppler and continuous phase OD.
use numeris::{DynMatrix, DynVector, Vector, optim::LmSettings};
use satkit::{Duration, ITRFCoord, TLE, frametransform};
use std::{
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};
use tychonic::{estimate::*, kernels::*, optim::*};
use tychonic::{numeris, satkit};

type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;

struct Options {
    trials: usize,
    seed: u64,
    output: PathBuf,
    tau_hours: f64,
    drift_deg: f64,
    baseline_m: f64,
    carrier_hz: f64,
    cadence_seconds: f64,
    window_seconds: f64,
    min_elevation_deg: f64,
    sigma_rr: f64,
    start_trial: usize,
}
impl Options {
    fn parse() -> AnyResult<Self> {
        let mut o = Self {
            trials: 100,
            seed: 20261103,
            output: "artifacts/phase_pass".into(),
            tau_hours: 6.0,
            drift_deg: 5.0,
            baseline_m: 59.0,
            carrier_hz: 2.23e9,
            cadence_seconds: 10.0,
            window_seconds: 900.0,
            min_elevation_deg: 10.0,
            sigma_rr: 0.05,
            start_trial: 0,
        };
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            if key == "--help" {
                println!(
                    "phase_pass [--trials 100] [--start-trial 0] [--seed 20261103] [--output artifacts/phase_pass]\n  [--tau-hours 6] [--drift-deg 5] [--baseline-m 59] [--carrier-hz 2230000000]\n  [--cadence-seconds 10] [--window-seconds 900] [--min-elevation-deg 10] [--sigma-rr 0.05]"
                );
                std::process::exit(0);
            }
            let value = args.next().ok_or("option needs a value")?;
            match key.as_str() {
                "--trials" => o.trials = value.parse()?,
                "--seed" => o.seed = value.parse()?,
                "--start-trial" => o.start_trial = value.parse()?,
                "--output" => o.output = value.into(),
                "--tau-hours" => o.tau_hours = value.parse()?,
                "--drift-deg" => o.drift_deg = value.parse()?,
                "--baseline-m" => o.baseline_m = value.parse()?,
                "--carrier-hz" => o.carrier_hz = value.parse()?,
                "--cadence-seconds" => o.cadence_seconds = value.parse()?,
                "--window-seconds" => o.window_seconds = value.parse()?,
                "--min-elevation-deg" => o.min_elevation_deg = value.parse()?,
                "--sigma-rr" => o.sigma_rr = value.parse()?,
                _ => return Err(format!("unknown option {key}").into()),
            }
        }
        if o.trials == 0
            || !o.cadence_seconds.is_finite()
            || o.cadence_seconds <= 0.0
            || !o.window_seconds.is_finite()
            || o.window_seconds <= 0.0
            || !o.min_elevation_deg.is_finite()
            || !(0.0..90.0).contains(&o.min_elevation_deg)
            || !o.sigma_rr.is_finite()
            || o.sigma_rr <= 0.0
        {
            return Err(
                "invalid trial count, window, cadence, elevation or range-rate sigma".into(),
            );
        }
        Ok(o)
    }
}
/// SplitMix64 with strictly positive uniforms and Box-Muller normals.
struct Random(u64);
impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}
fn csv(path: PathBuf, header: &str) -> AnyResult<BufWriter<File>> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "{header}")?;
    Ok(out)
}
fn config(epoch: satkit::Instant) -> Result<PropagationConfig> {
    PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: 0.05 },
    )
}
fn metrics(x: &V6, p: &M6, truth: &V6) -> AnyResult<(f64, f64, f64, f64, f64)> {
    let error = x - truth;
    let position = r_of(&error).norm();
    let velocity = v_of(&error).norm();
    let ps = (0..3).map(|i| p[(i, i)]).sum::<f64>().sqrt();
    let vs = (3..6).map(|i| p[(i, i)]).sum::<f64>().sqrt();
    let scales = Vector::<f64, 6>::from_array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0]);
    let mut scaled = *p;
    let mut e = error;
    for i in 0..6 {
        e[i] /= scales[i];
        for j in 0..6 {
            scaled[(i, j)] /= scales[i] * scales[j];
        }
    }
    let nees = e.dot(
        &scaled
            .cholesky()
            .map_err(|_| KernelError::Singular)?
            .solve(&e),
    );
    Ok((position, velocity, ps, vs, nees))
}
fn endpoint(
    out: &mut impl Write,
    trial: usize,
    level: f64,
    solver: &str,
    x: &[f64],
    p: &DynMatrix<f64>,
) -> AnyResult<()> {
    for i in 0..x.len() {
        for j in 0..x.len() {
            writeln!(
                out,
                "{trial},{level},{solver},{i},{j},{:.16e},{:.16e}",
                x[i],
                p[(i, j)]
            )?;
        }
    }
    Ok(())
}
fn history(
    out: &mut impl Write,
    series: (f64, &str, f64),
    x: &V6,
    p: &M6,
    truth: &V6,
    phase_offsets: (f64, f64),
) -> AnyResult<()> {
    let (level, solver, dt) = series;
    let (estimated_offset, true_offset) = phase_offsets;
    let (pe, ve, ps, vs, _) = metrics(x, p, truth)?;
    write!(
        out,
        "{level},{solver},{dt},{pe:.12e},{ve:.12e},{ps:.12e},{vs:.12e},{estimated_offset:.12e},{true_offset:.12e}"
    )?;
    for k in 0..6 {
        write!(out, ",{:.16e}", x[k])?;
    }
    writeln!(out)?;
    Ok(())
}
fn main() -> AnyResult<()> {
    let o = Options::parse()?;
    let lm_settings = LmSettings {
        grad_tol: 1e-2,
        x_tol: 1e-5,
        f_tol: 1e-5,
        max_iter: 100,
        ..LmSettings::default()
    };
    let ou = OuPhaseNoise::new(o.tau_hours * 3600.0, o.drift_deg.to_radians())?;
    let interferometer = Interferometer::new(v3(o.baseline_m, 0.0, 0.0), o.carrier_hz)?;
    let tle = TLE::load_2line(
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009",
        "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299350",
    )?;
    let midpoint = initial_state_from_tle(&tle, &tle.epoch)?;
    let subsatellite =
        ITRFCoord::from_vector(&(frametransform::qgcrf2itrf_approx(&tle.epoch) * r_of(&midpoint)));
    let (lat, lon, _) = subsatellite.to_geodetic_deg();
    let station = GroundStation {
        lat_deg: lat,
        lon_deg: lon,
        alt_m: 100.0,
    };
    let all_times: Vec<_> = (0..=(o.window_seconds / o.cadence_seconds).floor() as usize)
        .map(|i| {
            tle.epoch
                + Duration::from_seconds(-o.window_seconds / 2.0 + i as f64 * o.cadence_seconds)
        })
        .collect();
    let all_geometry = PreparedGeometry::new(&all_times, &vec![station; all_times.len()])?;
    let all_truth =
        config(tle.epoch)?.sample(&DynVector::from_slice(midpoint.as_slice()), &all_times)?;
    let selected: Vec<_> = (0..all_times.len())
        .filter(|&i| {
            let g = all_geometry.row(i).unwrap();
            let local = g
                .rotation()
                .to_enu(&(r_of(&all_truth.states[i]) - g.position()));
            local[2].atan2(local[0].hypot(local[1])) >= o.min_elevation_deg.to_radians()
        })
        .collect();
    if selected.len() < 8 {
        return Err("fewer than eight visible observations".into());
    }
    let times: Vec<_> = selected.iter().map(|&i| all_times[i]).collect();
    let truth: Vec<_> = selected.iter().map(|&i| all_truth.states[i]).collect();
    let geometry = Arc::new(PreparedGeometry::new(&times, &vec![station; times.len()])?);
    let epoch = times[0];
    let final_truth = *truth.last().unwrap();
    let n = times.len();
    let sigma_f = o.carrier_hz / SPEED_OF_LIGHT * o.sigma_rr;
    fs::create_dir_all(&o.output)?;
    fs::write(
        o.output.join("solver_settings.json"),
        format!(
            "{{\"start_trial\": {}, \"grad_tol\": {}, \"x_tol\": {}, \"f_tol\": {}, \"max_iter\": {}, \"numeris\": \"0.6.0\", \"satkit\": \"0.24.1\", \"iteration_limit_check\": \"undamped QR step or predicted-cost tolerance\"}}\n",
            o.start_trial,
            lm_settings.grad_tol,
            lm_settings.x_tol,
            lm_settings.f_tol,
            lm_settings.max_iter
        ),
    )?;
    fs::write(
        o.output.join("config.json"),
        format!(
            "{{\n  \"trials\": {}, \"seed\": {}, \"tau_hours\": {}, \"stationary_drift_deg\": {},\n  \"baseline_enu_m\": [{}, 0, 0], \"carrier_hz\": {}, \"sigma_rr_m_s\": {},\n  \"cadence_seconds\": {}, \"window_seconds\": {}, \"min_elevation_deg\": {},\n  \"retained_samples\": {}, \"retained_duration_seconds\": {}, \"reference_epoch\": \"{}\",\n  \"station_lat_deg\": {}, \"station_lon_deg\": {}, \"station_alt_m\": 100,\n  \"phase_noise_deg\": [0.1, 1, 5, 10], \"fixed_drag_m2_kg\": 0.05,\n  \"prior_sigmas\": [1000,1000,1000,1,1,1,10,3.141592653589793],\n  \"phase_convention\": \"continuous; baseline antenna1 minus antenna2; frequency relative to carrier\"\n}}\n",
            o.trials,
            o.seed,
            o.tau_hours,
            o.drift_deg,
            o.baseline_m,
            o.carrier_hz,
            o.sigma_rr,
            o.cadence_seconds,
            o.window_seconds,
            o.min_elevation_deg,
            n,
            (times[n - 1] - epoch).as_seconds(),
            epoch,
            lat,
            lon
        ),
    )?;
    let mut truth_file = csv(
        o.output.join("truth.csv"),
        "seconds,x,y,z,vx,vy,vz,elevation_deg",
    )?;
    for i in 0..n {
        let local = geometry
            .row(i)
            .unwrap()
            .rotation()
            .to_enu(&(r_of(&truth[i]) - geometry.row(i).unwrap().position()));
        write!(truth_file, "{}", (times[i] - epoch).as_seconds())?;
        for v in truth[i].as_slice() {
            write!(truth_file, ",{v:.16e}")?;
        }
        writeln!(
            truth_file,
            ",{}",
            local[2].atan2(local[0].hypot(local[1])).to_degrees()
        )?;
    }
    let mut obs = csv(
        o.output.join("observations.csv"),
        "trial,seed,phase_sigma_deg,seconds,doppler_hz,phase_rad,true_frequency_bias_hz,true_phase_mean_rad,true_drift_rad",
    )?;
    let mut priors = csv(o.output.join("priors.csv"), "trial,seed,index,mean,sigma")?;
    let mut results = csv(
        o.output.join("results.csv"),
        "trial,phase_sigma_deg,solver,status,position_m,velocity_m_s,position_sigma_m,velocity_sigma_m_s,nees,mean_nis,rss,iterations,error,convergence",
    )?;
    let mut endpoints = csv(
        o.output.join("endpoints.csv"),
        "trial,phase_sigma_deg,solver,i,j,state_i,covariance_ij",
    )?;
    let mut histories = csv(
        o.output.join("representative.csv"),
        "phase_sigma_deg,solver,seconds,position_m,velocity_m_s,position_sigma_m,velocity_sigma_m_s,estimated_phase_offset_rad,true_phase_offset_rad,x,y,z,vx,vy,vz",
    )?;
    let mut failure_count = 0;
    println!(
        "{} visible samples over {:.0} s; {} paired trials; OU {} h / {} deg",
        n,
        (times[n - 1] - epoch).as_seconds(),
        o.trials,
        o.tau_hours,
        o.drift_deg
    );
    for trial in o.start_trial..o.start_trial + o.trials {
        let seed = o
            .seed
            .wrapping_add((trial as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let mut rng = Random(seed);
        let sigmas = DynVector::from_slice(&[
            1000.0,
            1000.0,
            1000.0,
            1.0,
            1.0,
            1.0,
            10.0,
            std::f64::consts::PI,
        ]);
        let mut mean = DynVector::zeros(8);
        for i in 0..6 {
            mean[i] = truth[0][i] + sigmas[i] * rng.normal();
        }
        let prior = RfPrior {
            mean,
            sigma: sigmas,
        };
        for i in 0..8 {
            writeln!(
                priors,
                "{trial},{seed},{i},{:.16e},{:.16e}",
                prior.mean[i], prior.sigma[i]
            )?;
        }
        let true_frequency_bias = prior.sigma[6] * rng.normal();
        let true_phase_mean = prior.sigma[7] * rng.normal();
        let mut drift = vec![ou.stationary_sigma_rad() * rng.normal(); n];
        for i in 1..n {
            let (a, q) = ou.transition((times[i] - times[i - 1]).as_seconds())?;
            drift[i] = a * drift[i - 1] + q.sqrt() * rng.normal();
        }
        let zf: Vec<_> = (0..n).map(|_| rng.normal()).collect();
        let zp: Vec<_> = (0..n).map(|_| rng.normal()).collect();
        let predictions: Vec<_> = (0..n)
            .map(|i| {
                interferometer.evaluate(&truth[i], geometry.row(i).unwrap(), true_frequency_bias)
            })
            .collect::<Result<_>>()?;
        for level in [0.0_f64, 0.1, 1.0, 5.0, 10.0] {
            let sigma_phase = if level == 0.0 {
                1.0
            } else {
                level.to_radians()
            };
            let data = PassiveRfData {
                geometry: geometry.clone(),
                interferometer,
                doppler_hz: (0..n)
                    .map(|i| predictions[i].doppler_hz + sigma_f * zf[i])
                    .collect(),
                phase_rad: (level > 0.0).then(|| {
                    (0..n)
                        .map(|i| {
                            predictions[i].phase_rad
                                + true_phase_mean
                                + drift[i]
                                + sigma_phase * zp[i]
                        })
                        .collect()
                }),
                sigma_doppler_hz: sigma_f,
                sigma_phase_rad: sigma_phase,
            };
            for i in 0..n {
                writeln!(
                    obs,
                    "{trial},{seed},{level},{},{:.16e},{},{true_frequency_bias:.16e},{true_phase_mean:.16e},{:.16e}",
                    (times[i] - epoch).as_seconds(),
                    data.doppler_hz[i],
                    data.phase_rad
                        .as_ref()
                        .map(|v| format!("{:.16e}", v[i]))
                        .unwrap_or_default(),
                    drift[i]
                )?;
            }
            let batch_run = (|| -> AnyResult<()> {
                let problem =
                    PassiveRfProblem::new(config(epoch)?, data.clone(), prior.clone(), ou)?;
                let fit = fit_passive_rf(&problem, &prior.mean, &lm_settings)?;
                let sens = config(epoch)?.sample(
                    &DynVector::from_slice(&fit.parameters.as_slice()[..6]),
                    &times,
                )?;
                let mut p0 = M6::zeros();
                for i in 0..6 {
                    for j in 0..6 {
                        p0[(i, j)] = fit.covariance[(i, j)];
                    }
                }
                let mut f = DynMatrix::zeros(8, 8);
                for i in 0..8 {
                    f[(i, i)] = 1.0;
                }
                for i in 0..6 {
                    for j in 0..6 {
                        f[(i, j)] = sens.stm[n - 1][(i, j)];
                    }
                }
                let p_end = &f * &fit.covariance * f.transpose();
                let x_end = sens.states[n - 1];
                let p = sens.stm[n - 1] * p0 * sens.stm[n - 1].transpose();
                let (pe, ve, ps, vs, nees) = metrics(&x_end, &p, &final_truth)?;
                writeln!(
                    results,
                    "{trial},{level},batch,ok,{pe:.12e},{ve:.12e},{ps:.12e},{vs:.12e},{nees:.12e},,{:.12e},{},,{}",
                    fit.rss,
                    fit.iterations,
                    if fit.linearized_tolerance {
                        "verified_linearized_tolerance"
                    } else {
                        "numeris"
                    }
                )?;
                let mut x = fit.parameters.as_slice().to_vec();
                x[..6].copy_from_slice(x_end.as_slice());
                endpoint(&mut endpoints, trial, level, "batch", &x, &p_end)?;
                if trial == o.start_trial {
                    for i in 0..n {
                        history(
                            &mut histories,
                            (level, "batch", (times[i] - epoch).as_seconds()),
                            &sens.states[i],
                            &(sens.stm[i] * p0 * sens.stm[i].transpose()),
                            &truth[i],
                            (
                                fit.parameters[7] + fit.drift_rad.get(i).copied().unwrap_or(0.0),
                                true_phase_mean + drift[i],
                            ),
                        )?;
                    }
                }
                Ok(())
            })();
            if let Err(e) = batch_run {
                failure_count += 1;
                let error = e.to_string().replace([',', '\n', '\r'], " ");
                writeln!(results, "{trial},{level},batch,failed,,,,,,,,,{error},")?;
                eprintln!("trial {trial}, phase {level}: batch {error}");
            }
            let filter_run = (|| -> AnyResult<()> {
                let mut filter = PassiveRfEkf::<9>::new(config(epoch)?, ou, &prior)?;
                let updates = filter.filter_data(&data, UpdateOptions::default())?;
                let mean_nis = updates.iter().map(|r| r.0).sum::<f64>() / n as f64;
                let x = V6::from_slice(&filter.state().as_slice()[..6]);
                let p = filter.covariance().block::<6, 6>(0, 0);
                let (pe, ve, ps, vs, nees) = metrics(&x, &p, &final_truth)?;
                writeln!(
                    results,
                    "{trial},{level},ekf,ok,{pe:.12e},{ve:.12e},{ps:.12e},{vs:.12e},{nees:.12e},{mean_nis:.12e},,,,direct"
                )?;
                let mut pdyn = DynMatrix::zeros(9, 9);
                for i in 0..9 {
                    for j in 0..9 {
                        pdyn[(i, j)] = filter.covariance()[(i, j)];
                    }
                }
                endpoint(
                    &mut endpoints,
                    trial,
                    level,
                    "ekf",
                    filter.state().as_slice(),
                    &pdyn,
                )?;
                let smooth = match filter.smooth() {
                    Ok(s) => s,
                    Err(e) => {
                        failure_count += 1;
                        writeln!(
                            results,
                            "{trial},{level},rts,failed,,,,,,,,,{},",
                            e.to_string().replace([',', '\n', '\r'], " ")
                        )?;
                        return Ok(());
                    }
                };
                writeln!(
                    results,
                    "{trial},{level},rts,ok,{pe:.12e},{ve:.12e},{ps:.12e},{vs:.12e},{nees:.12e},,,,,direct"
                )?;
                endpoint(
                    &mut endpoints,
                    trial,
                    level,
                    "rts",
                    smooth[n - 1].0.as_slice(),
                    &pdyn,
                )?;
                if trial == o.start_trial {
                    for i in 0..n {
                        for (solver, state, cov) in [
                            (
                                "ekf",
                                filter.steps()[i].x_updated,
                                filter.steps()[i].p_updated,
                            ),
                            ("rts", smooth[i].0, smooth[i].1),
                        ] {
                            history(
                                &mut histories,
                                (level, solver, (times[i] - epoch).as_seconds()),
                                &V6::from_slice(&state.as_slice()[..6]),
                                &cov.block::<6, 6>(0, 0),
                                &truth[i],
                                (state[7] + state[8], true_phase_mean + drift[i]),
                            )?;
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = filter_run {
                failure_count += 1;
                let error = e.to_string().replace([',', '\n', '\r'], " ");
                writeln!(results, "{trial},{level},ekf,failed,,,,,,,,,{error},")?;
                writeln!(results, "{trial},{level},rts,failed,,,,,,,,,EKF failed,")?;
                eprintln!("trial {trial}, phase {level}: EKF {error}");
            }
        }
        if (trial + 1 - o.start_trial) % 10 == 0 || trial + 1 - o.start_trial == o.trials {
            results.flush()?;
            println!(
                "Completed {}/{} trials ({} solver failures)",
                trial + 1 - o.start_trial,
                o.trials,
                failure_count
            );
        }
    }
    println!(
        "Outputs: {}. Plot and summarize with python3 scripts/plot_phase_pass.py {}",
        o.output.display(),
        o.output.display()
    );
    if failure_count > 0 {
        return Err(
            format!("study recorded {failure_count} solver failures; inspect results.csv").into(),
        );
    }
    Ok(())
}
