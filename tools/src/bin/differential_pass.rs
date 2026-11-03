//! Paired extension of an existing phase_pass simulation. Three cases distinguish
//! the spatial differential signal from ordinary independent-noise averaging.
use numeris::{DynMatrix, DynVector, optim::LmSettings};
use satkit::{Duration, Instant};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use tychonic::{estimate::*, kernels::*, optim::*};
use tychonic::{numeris, satkit};
type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;
#[path = "support/regional_pass.rs"]
mod regional_pass;
struct Random(u64);
impl Random {
    fn normal(&mut self) -> f64 {
        fn uniform(state: &mut u64) -> f64 {
            *state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            ((z >> 12) as f64 + 0.5) / ((1u64 << 52) as f64)
        }
        (-2.0 * uniform(&mut self.0).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * uniform(&mut self.0)).cos()
    }
}
fn rows(path: &Path) -> AnyResult<Vec<Vec<String>>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .skip(1)
        .map(|line| line.split(',').map(str::to_owned).collect())
        .collect())
}
fn number(row: &[String], i: usize) -> AnyResult<f64> {
    Ok(row.get(i).ok_or("missing CSV column")?.parse()?)
}
fn config_number(config: &Value, key: &str) -> AnyResult<f64> {
    config[key]
        .as_f64()
        .ok_or_else(|| format!("missing {key}").into())
}
fn output(path: PathBuf, header: &str) -> AnyResult<BufWriter<File>> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "{header}")?;
    Ok(f)
}
fn errors(x: &V6, p: &M6, truth: &V6) -> Result<(f64, f64, f64)> {
    let e = x - truth;
    let pe = r_of(&e).norm();
    let ve = v_of(&e).norm();
    let scales = [1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0];
    let mut scaled = *p;
    let mut se = e;
    for i in 0..6 {
        se[i] /= scales[i];
        for j in 0..6 {
            scaled[(i, j)] /= scales[i] * scales[j];
        }
    }
    let nees = se.dot(
        &scaled
            .cholesky()
            .map_err(|_| KernelError::Singular)?
            .solve(&se),
    );
    Ok((pe, ve, nees))
}
fn endpoint(x: &[f64], p: &DynMatrix<f64>) -> Value {
    json!({"state":x,"covariance":(0..p.nrows()).map(|i| (0..p.ncols()).map(|j| p[(i,j)]).collect::<Vec<_>>()).collect::<Vec<_>>()})
}
struct FilterOutput {
    x: DynVector<f64>,
    covariance: DynMatrix<f64>,
    mean_nis: f64,
    rts_ok: bool,
}
fn filter<const N: usize>(
    config: PropagationConfig,
    ou: OuPhaseNoise,
    prior: &RfPrior,
    data: &PassiveRfData,
    extensions: &RfExtensions,
) -> AnyResult<FilterOutput> {
    let mut f = PassiveRfEkf::<N>::new_with_calibration(
        config,
        ou,
        prior,
        extensions.baseline_sigma_m,
        extensions.differential_baseline_sigma_m,
    )?;
    let updates = f.filter_data_with_extensions(data, extensions, UpdateOptions::default())?;
    let mut covariance = DynMatrix::zeros(N, N);
    for i in 0..N {
        for j in 0..N {
            covariance[(i, j)] = f.covariance()[(i, j)];
        }
    }
    Ok(FilterOutput {
        x: DynVector::from_slice(f.state().as_slice()),
        covariance,
        mean_nis: updates.iter().map(|u| u.0).sum::<f64>() / updates.len() as f64,
        rts_ok: f.smooth().is_ok(),
    })
}
fn main() -> AnyResult<()> {
    let mut input = PathBuf::from("artifacts/phase_pass");
    let mut out = PathBuf::from("artifacts/differential_pass");
    let mut trials = 100_usize;
    let mut baseline_mm = 5.0_f64;
    let mut noise_scale = 1.0_f64;
    let mut second_sigma_rr = None;
    let mut scenario = String::from("short-baseline");
    let mut explicit_input = false;
    let mut explicit_output = false;
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!(
                "differential_pass [--input artifacts/phase_pass] [--output artifacts/differential_pass]\n  [--trials 100] [--baseline-sigma-mm 5] [--noise-scale 1] [--antenna2-sigma-rr 0.05]\n  [--scenario short-baseline|tromso-svalbard]"
            );
            return Ok(());
        }
        let v = args.next().ok_or("option requires value")?;
        match key.as_str() {
            "--input" => {
                input = v.into();
                explicit_input = true;
            }
            "--output" => {
                out = v.into();
                explicit_output = true;
            }
            "--trials" => trials = v.parse()?,
            "--baseline-sigma-mm" => baseline_mm = v.parse()?,
            "--noise-scale" => noise_scale = v.parse()?,
            "--antenna2-sigma-rr" => second_sigma_rr = Some(v.parse::<f64>()?),
            "--scenario" => scenario = v,
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if !["short-baseline", "tromso-svalbard"].contains(&scenario.as_str()) {
        return Err("unknown scenario; use short-baseline or tromso-svalbard".into());
    }
    if trials == 0
        || !baseline_mm.is_finite()
        || baseline_mm < 0.0
        || !noise_scale.is_finite()
        || noise_scale <= 0.0
    {
        return Err("invalid trial count, baseline sigma or noise scale".into());
    }
    if scenario == "tromso-svalbard" {
        if explicit_input {
            return Err(
                "--scenario tromso-svalbard generates its own polar pass; omit --input".into(),
            );
        }
        if !explicit_output {
            out = PathBuf::from("artifacts/differential_tromso_svalbard");
        }
        input = out.join("source_inputs");
        regional_pass::generate(&input, trials)?;
    }
    let source: Value = serde_json::from_str(&fs::read_to_string(input.join("config.json"))?)?;
    let epoch = Instant::from_string(source["reference_epoch"].as_str().ok_or("missing epoch")?)?;
    let carrier = config_number(&source, "carrier_hz")?;
    let baseline = source["baseline_enu_m"]
        .as_array()
        .ok_or("missing baseline")?;
    let source_int = Interferometer::new(
        v3(
            baseline[0].as_f64().ok_or("baseline E")?,
            baseline[1].as_f64().ok_or("baseline N")?,
            baseline[2].as_f64().ok_or("baseline U")?,
        ),
        carrier,
    )?;
    let int = source_int.with_spherical_phase();
    let ou = OuPhaseNoise::new(
        config_number(&source, "tau_hours")? * 3600.0,
        config_number(&source, "stationary_drift_deg")?.to_radians(),
    )?;
    let drag = config_number(&source, "fixed_drag_m2_kg")?;
    let cfg = PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: drag },
    )?;
    let sigma1_original = config_number(&source, "sigma_rr_m_s")? * carrier / SPEED_OF_LIGHT;
    let sigma1 = sigma1_original * noise_scale;
    let sigma2 = second_sigma_rr.unwrap_or(config_number(&source, "sigma_rr_m_s")?) * carrier
        / SPEED_OF_LIGHT
        * noise_scale;
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err("antenna 2 noise must be positive".into());
    }
    let sigma = (baseline_mm > 0.0).then_some(v3(
        baseline_mm / 1000.0,
        baseline_mm / 1000.0,
        baseline_mm / 1000.0,
    ));
    let truth_rows = rows(&input.join("truth.csv"))?;
    let times: Vec<_> = truth_rows
        .iter()
        .map(|r| Ok(epoch + Duration::from_seconds(number(r, 0)?)))
        .collect::<AnyResult<_>>()?;
    let truth: Vec<_> = truth_rows
        .iter()
        .map(|r| {
            Ok(V6::from_slice(
                &(1..7)
                    .map(|i| number(r, i))
                    .collect::<AnyResult<Vec<_>>>()?,
            ))
        })
        .collect::<AnyResult<_>>()?;
    let site = GroundStation {
        lat_deg: config_number(&source, "station_lat_deg")?,
        lon_deg: config_number(&source, "station_lon_deg")?,
        alt_m: config_number(&source, "station_alt_m")?,
    };
    let differential_int = if scenario == "tromso-svalbard" {
        Some(Interferometer::new(
            site.baseline_enu_to(&regional_pass::SVALBARD)?,
            carrier,
        )?)
    } else {
        None
    };
    let differential_sigma = differential_int.and(sigma);
    let nominal_differential_int = differential_int.unwrap_or(int);
    let geometry = Arc::new(PreparedGeometry::new(&times, &vec![site; times.len()])?);
    let mut priors = BTreeMap::<usize, RfPrior>::new();
    for r in rows(&input.join("priors.csv"))? {
        let trial = r[0].parse::<usize>()?;
        let k = r[2].parse::<usize>()?;
        if k >= 8 {
            return Err("source prior must contain eight RF parameters".into());
        }
        let prior = priors.entry(trial).or_insert_with(|| RfPrior {
            mean: DynVector::zeros(8),
            sigma: DynVector::zeros(8),
        });
        prior.mean[k] = number(&r, 3)?;
        prior.sigma[k] = number(&r, 4)?;
    }
    let mut groups = BTreeMap::<(usize, i32), Vec<Vec<String>>>::new();
    for r in rows(&input.join("observations.csv"))? {
        let level = number(&r, 2)?;
        if level == 0.0 {
            continue;
        }
        groups
            .entry((r[0].parse()?, (level * 10.0).round() as i32))
            .or_default()
            .push(r);
    }
    if priors.len() < trials {
        return Err("input has fewer trials than requested".into());
    }
    fs::create_dir_all(&out)?;
    fs::copy(input.join("truth.csv"), out.join("truth.csv"))?;
    fs::copy(input.join("priors.csv"), out.join("priors.csv"))?;
    fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&json!({"source":source,"input":input,
        "trials":trials,"baseline_sigma_mm_per_component":baseline_mm,"antenna1_sigma_hz":sigma1,
        "antenna2_sigma_hz":sigma2,"noise_scale":noise_scale,
        "differential_sign":"f1-f2","receiver_frequency_offsets":"common and cancelling; differential offset calibrated",
        "phase_and_frequency_noise":"independent estimates; phase OU not differentiated",
            "phase_path_model":"spherical, rationalized exact range difference",
        "scenario":scenario,"phase_baseline_enu_m":int.baseline_enu().as_slice(),
        "differential_baseline_enu_m":nominal_differential_int.baseline_enu().as_slice(),
        "differential_baseline_length_m":nominal_differential_int.baseline_enu().norm(),
        "differential_baseline_sigma_mm_per_component":if differential_int.is_some() { baseline_mm } else { 0.0 },
        "baseline_calibration":"constant ENU vector errors; independent local phase and remote Doppler corrections when separate",
        "differential_station":if differential_int.is_some() {source["remote_station"].clone()} else {Value::Null},
        "signal_time_model":"instantaneous simultaneous reception; light time, emission-time difference and clock errors omitted",
        "cases":["phase_doppler","averaging_control","phase_doppler_differential"]}))?,
    )?;
    let mut result = output(
        out.join("results.csv"),
        "trial,phase_sigma_deg,case,solver,status,position_m,velocity_m_s,nees,mean_nis,iterations,convergence,rts_ok,error",
    )?;
    let mut samples = output(
        out.join("observations.csv"),
        "trial,phase_sigma_deg,seconds,f1_hz,f2_hz,phase_rad,difference_hz,true_difference_hz,true_baseline_error_e_m,true_baseline_error_n_m,true_baseline_error_u_m,true_differential_baseline_error_e_m,true_differential_baseline_error_n_m,true_differential_baseline_error_u_m",
    )?;
    let mut endpoints = BufWriter::new(File::create(out.join("endpoints.jsonl"))?);
    let settings = LmSettings {
        grad_tol: 1e-2,
        x_tol: 1e-5,
        f_tol: 1e-5,
        max_iter: 100,
        ..LmSettings::default()
    };
    let mut failures = 0;
    let mut count = 0;
    for (&trial, prior) in priors.iter().take(trials) {
        let first = groups
            .get(&(trial, 1))
            .ok_or("source lacks 0.1 degree phase data")?;
        let seed = first[0][1].parse::<u64>()?;
        let mut rng = Random(seed ^ 0xd2f4_0b73_12ce_9901);
        let unit_baseline_error = v3(rng.normal(), rng.normal(), rng.normal());
        let btrue = if let Some(sigma) = sigma {
            v3(
                sigma[0] * unit_baseline_error[0],
                sigma[1] * unit_baseline_error[1],
                sigma[2] * unit_baseline_error[2],
            )
        } else {
            V3::zeros()
        };
        let truth_int = int.corrected(&btrue)?;
        let second_noise: Vec<_> = (0..times.len()).map(|_| sigma2 * rng.normal()).collect();
        let mut remote_rng = Random(seed ^ 0x96aa_108f_741a_33e9);
        let remote_error = differential_sigma.map_or(V3::zeros(), |s| {
            v3(
                s[0] * remote_rng.normal(),
                s[1] * remote_rng.normal(),
                s[2] * remote_rng.normal(),
            )
        });
        let truth_differential_int = if let Some(remote) = differential_int {
            remote.corrected(&remote_error)?
        } else {
            truth_int
        };
        for level in [0.1_f64, 1.0, 5.0, 10.0] {
            let source_rows = groups
                .get(&(trial, (level * 10.0).round() as i32))
                .ok_or("source lacks phase level")?;
            if source_rows.len() != times.len() {
                return Err("source schedule mismatch".into());
            }
            let mut f1 = Vec::new();
            let mut phases = Vec::new();
            let mut differences = Vec::new();
            let mut averaging = Vec::new();
            for i in 0..times.len() {
                if (number(&source_rows[i], 3)? - (times[i] - epoch).as_seconds()).abs() > 1e-6 {
                    return Err("source timestamps differ".into());
                }
                let bias = number(&source_rows[i], 6)?;
                let row = geometry.row(i).unwrap();
                let nominal = source_int.evaluate(&truth[i], row, bias)?;
                let actual = truth_int.evaluate(&truth[i], row, bias)?;
                let first_noise = (number(&source_rows[i], 4)? - nominal.doppler_hz) * noise_scale;
                let dtrue = truth_differential_int
                    .differential(&truth[i], row, times[i])?
                    .difference_hz;
                let observed = actual.doppler_hz + first_noise;
                let d = dtrue + first_noise - second_noise[i];
                let phase = number(&source_rows[i], 5)? + actual.phase_rad - nominal.phase_rad;
                f1.push(observed);
                differences.push(d);
                phases.push(phase);
                // Counterfactual co-located receivers isolate the benefit of
                // averaging independent frequency errors without spatial data.
                let weight1 = sigma2.powi(2) / (sigma1.powi(2) + sigma2.powi(2));
                averaging.push(
                    actual.doppler_hz + weight1 * first_noise + (1.0 - weight1) * second_noise[i],
                );
                writeln!(
                    samples,
                    "{trial},{level},{},{observed:.16e},{:.16e},{phase:.16e},{d:.16e},{dtrue:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
                    (times[i] - epoch).as_seconds(),
                    observed - d,
                    btrue[0],
                    btrue[1],
                    btrue[2],
                    remote_error[0],
                    remote_error[1],
                    remote_error[2]
                )?;
            }
            for case in [
                "phase_doppler",
                "averaging_control",
                "phase_doppler_differential",
            ] {
                let data = PassiveRfData {
                    geometry: geometry.clone(),
                    interferometer: int,
                    doppler_hz: if case == "averaging_control" {
                        averaging.clone()
                    } else {
                        f1.clone()
                    },
                    phase_rad: Some(phases.clone()),
                    sigma_doppler_hz: if case == "averaging_control" {
                        sigma1 * sigma2 / (sigma1.powi(2) + sigma2.powi(2)).sqrt()
                    } else {
                        sigma1
                    },
                    sigma_phase_rad: level.to_radians(),
                };
                let dd = (case == "phase_doppler_differential").then(|| DifferentialDopplerData {
                    difference_hz: differences.clone(),
                    sigma_antenna2_hz: sigma2,
                });
                let extensions = RfExtensions {
                    baseline_sigma_m: sigma,
                    differential_doppler: dd,
                    differential_interferometer: differential_int,
                    differential_baseline_sigma_m: differential_sigma,
                };
                let problem = PassiveRfProblem::new_extended(
                    cfg.clone(),
                    data.clone(),
                    prior.clone(),
                    ou,
                    extensions.clone(),
                )?;
                let batch = (|| -> AnyResult<()> {
                    let fit = fit_passive_rf(&problem, &problem.initial_parameters(), &settings)?;
                    let sens = cfg.sample(
                        &DynVector::from_slice(&fit.parameters.as_slice()[..6]),
                        &[times[times.len() - 1]],
                    )?;
                    let mut p0 = M6::zeros();
                    for i in 0..6 {
                        for j in 0..6 {
                            p0[(i, j)] = fit.covariance[(i, j)];
                        }
                    }
                    let p = sens.stm[0] * p0 * sens.stm[0].transpose();
                    let x = sens.states[0];
                    let (pe, ve, nees) = errors(&x, &p, truth.last().unwrap())?;
                    writeln!(
                        result,
                        "{trial},{level},{case},batch,ok,{pe:.12e},{ve:.12e},{nees:.12e},,{}, {},,",
                        fit.iterations,
                        if fit.linearized_tolerance {
                            "verified"
                        } else {
                            "numeris"
                        }
                    )?;
                    writeln!(
                        endpoints,
                        "{}",
                        json!({"trial":trial,"phase_sigma_deg":level,"case":case,"solver":"batch",
                        "epoch":"reference","estimate":endpoint(fit.parameters.as_slice(),&fit.covariance),"true_baseline_error_enu_m":btrue.as_slice(),"true_differential_baseline_error_enu_m":remote_error.as_slice()})
                    )?;
                    Ok(())
                })();
                if let Err(e) = batch {
                    failures += 1;
                    writeln!(
                        result,
                        "{trial},{level},{case},batch,failed,,,,,,,,{}",
                        e.to_string().replace([',', '\n'], " ")
                    )?;
                }
                let ekf = (|| -> AnyResult<()> {
                    let f = if sigma.is_some() && differential_sigma.is_some() {
                        filter::<15>(cfg.clone(), ou, prior, &data, &extensions)?
                    } else if sigma.is_some() || differential_sigma.is_some() {
                        filter::<12>(cfg.clone(), ou, prior, &data, &extensions)?
                    } else {
                        filter::<9>(cfg.clone(), ou, prior, &data, &extensions)?
                    };
                    let x = V6::from_slice(&f.x.as_slice()[..6]);
                    let mut p = M6::zeros();
                    for i in 0..6 {
                        for j in 0..6 {
                            p[(i, j)] = f.covariance[(i, j)];
                        }
                    }
                    let (pe, ve, nees) = errors(&x, &p, truth.last().unwrap())?;
                    writeln!(
                        result,
                        "{trial},{level},{case},ekf,ok,{pe:.12e},{ve:.12e},{nees:.12e},{:.12e},,direct,{},",
                        f.mean_nis, f.rts_ok
                    )?;
                    writeln!(
                        endpoints,
                        "{}",
                        json!({"trial":trial,"phase_sigma_deg":level,"case":case,"solver":"ekf","epoch":"final",
                        "estimate":endpoint(f.x.as_slice(),&f.covariance),"true_baseline_error_enu_m":btrue.as_slice(),"true_differential_baseline_error_enu_m":remote_error.as_slice()})
                    )?;
                    if !f.rts_ok {
                        failures += 1;
                    }
                    Ok(())
                })();
                if let Err(e) = ekf {
                    failures += 1;
                    writeln!(
                        result,
                        "{trial},{level},{case},ekf,failed,,,,,,,,{}",
                        e.to_string().replace([',', '\n'], " ")
                    )?;
                }
            }
        }
        count += 1;
        if count % 10 == 0 || count == trials {
            result.flush()?;
            println!("Completed {count}/{trials} trials ({failures} failures)");
        }
    }
    if count != trials {
        return Err("input has fewer trials than requested".into());
    }
    println!("Output: {}", out.display());
    if failures > 0 {
        return Err(format!("{failures} recorded solver failures").into());
    }
    Ok(())
}
