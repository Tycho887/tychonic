//! Profile-specific adapters for the constrained, contact-bias and RF estimators.
use crate::{
    common::*,
    config::NativePropagation,
    solve::{Rows, rows},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use tychonic::{
    kernels::*,
    measurements::{GaussianPrior, RfBuilder, RfPriors},
    numeris::{DynVector, optim::LmSettings},
    optim::{constrained::*, *},
    satkit::Instant,
};

type Rate = (String, String, [f64; 3], f64, f64);
type Angle = (String, String, [f64; 3], f64, f64);
type BiasFit = (
    Vec<f64>,
    Option<Rows>,
    Vec<(String, f64)>,
    f64,
    usize,
    String,
    Option<f64>,
);
fn station(s: [f64; 3]) -> GroundStation {
    GroundStation {
        lat_deg: s[0],
        lon_deg: s[1],
        alt_m: s[2],
    }
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(config,guess,rates,angles,angle_sigmas,iterations=None,fov=None,near=None))]
pub fn contacts(
    py: Python<'_>,
    config: &NativePropagation,
    guess: Vec<f64>,
    rates: Vec<Rate>,
    angles: Vec<Angle>,
    angle_sigmas: [f64; 2],
    iterations: Option<usize>,
    fov: Option<f64>,
    near: Option<usize>,
) -> PyResult<BiasFit> {
    py.detach(move || {
        let mut rr = Vec::new();
        let mut sigma = Vec::new();
        for (id, time, site, value, sd) in rates {
            rr.push(ContactRangeRate {
                contact: ContactId::parse(&id).map_err(kernel_error)?,
                station: station(site),
                time: instant(&time)?,
                range_rate: value,
            });
            sigma.push(sd);
        }
        let point = angles
            .into_iter()
            .map(|(id, time, site, azimuth, elevation)| {
                Ok(ContactPointing {
                    contact: ContactId::parse(&id).map_err(kernel_error)?,
                    time: instant(&time)?,
                    station: station(site),
                    azimuth,
                    elevation,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let epoch = config.inner.reference_epoch();
        let (first, end) = rr
            .iter()
            .map(|r| r.time)
            .chain(point.iter().map(|r| r.time))
            .fold((epoch, epoch), |(a, b), t| (a.min(t), b.max(t)));
        let config = config.for_span(first, end)?;
        if let Some(fov) = fov {
            if guess.len() != 7 {
                return Err(PyValueError::new_err("constrained fit requires drag"));
            }
            let model = ConstrainedRangeRateProblem::new(
                config,
                V7::from_slice(&guess),
                &rr,
                &sigma,
                &point,
                fov,
                near.map_or(PointingSelection::All, PointingSelection::NearTca),
            )
            .map_err(kernel_error)?;
            let mut opts = FitOptions::default();
            if let Some(n) = iterations {
                opts.max_iterations = n;
            }
            let fit = fit_constrained_range_rate(&model, &opts)
                .map_err(|e| SolverError::new_err(e.to_string()))?;
            Ok((
                fit.parameters.as_slice().to_vec(),
                None,
                fit.biases
                    .into_iter()
                    .map(|(id, b)| (id.to_string(), b))
                    .collect(),
                fit.rss,
                fit.iterations,
                last(model.epochs())?,
                Some(fit.worst_violation_rad),
            ))
        } else {
            let first = *sigma
                .first()
                .ok_or_else(|| PyValueError::new_err("contact fit requires Doppler"))?;
            if sigma.iter().any(|v| (v / first - 1.).abs() > 1e-12) {
                return Err(PyValueError::new_err(
                    "contact-bias solver requires common range-rate uncertainty",
                ));
            }
            let soft: Vec<_> = point
                .into_iter()
                .map(|p| PointingConstraint {
                    time: p.time,
                    station: p.station,
                    azimuth: p.azimuth,
                    elevation: p.elevation,
                })
                .collect();
            let problem = ContactBiasProblem::new(
                config,
                &rr,
                &soft,
                MeasurementNoise {
                    sigma_rr: first,
                    sigma_az: angle_sigmas[0],
                    sigma_el: angle_sigmas[1],
                },
            )
            .map_err(kernel_error)?;
            let settings = lm(iterations);
            let fit = fit_varpro(&problem, &DynVector::from_vec(guess), None, &settings)
                .map_err(fit_error)?;
            Ok((
                fit.parameters.as_slice().to_vec(),
                Some(rows(&fit.covariance)),
                problem
                    .contacts()
                    .iter()
                    .zip(fit.biases)
                    .map(|(id, b)| (id.to_string(), b))
                    .collect(),
                fit.rss,
                fit.iterations,
                last(problem.inner().timestamps())?,
                None,
            ))
        }
    })
}
fn last(epochs: &[Instant]) -> PyResult<String> {
    epochs
        .iter()
        .max()
        .map(Instant::as_rfc3339)
        .ok_or_else(|| PyValueError::new_err("empty observations"))
}
fn lm(iterations: Option<usize>) -> LmSettings<f64> {
    let mut s = LmSettings::default();
    if let Some(n) = iterations {
        s.max_iter = n;
    }
    s
}

#[derive(serde::Deserialize)]
struct RfOptions {
    carrier: f64,
    sigmas: [f64; 3],
    offset: [f64; 2],
    phase: [f64; 2],
    drift: [f64; 2],
    baseline_sigma: Option<[f64; 3]>,
    differential_sigma: Option<[f64; 3]>,
    iterations: Option<usize>,
}

type RfResult = (Vec<f64>, Rows, f64, usize, Vec<f64>, String);
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn passive(
    py: Python<'_>,
    config: &NativePropagation,
    guess: Vec<f64>,
    prior: (Vec<f64>, Vec<f64>),
    times: Vec<String>,
    sites: Vec<[f64; 3]>,
    values: Rows,
    baselines: Vec<[f64; 3]>,
    differential_baselines: Vec<[f64; 3]>,
    options: &str,
) -> PyResult<RfResult> {
    let cfg = config.build()?;
    let o: RfOptions = json(options)?;
    let times = times
        .iter()
        .map(|t| instant(t))
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(move || {
        let g = PreparedGeometry::new(&times, &sites.into_iter().map(station).collect::<Vec<_>>())
            .map_err(kernel_error)?;
        let sensor = Interferometer::new(o.carrier).map_err(kernel_error)?;
        let mut columns = values.into_iter();
        let doppler = columns
            .next()
            .ok_or_else(|| PyValueError::new_err("Doppler column required"))?;
        let data = PassiveRfData::new(
            g,
            sensor,
            baselines.into_iter().map(V3::from_array).collect(),
            doppler,
            columns.next(),
            o.sigmas[0],
            o.sigmas[1],
        )
        .map_err(kernel_error)?;
        let scalar = |v: [f64; 2]| GaussianPrior {
            mean: v[0],
            sigma: v[1],
        };
        let priors = RfPriors {
            orbit_mean: DynVector::from_vec(prior.0),
            orbit_sigma: DynVector::from_vec(prior.1),
            frequency_offset_hz: scalar(o.offset),
            phase_offset_rad: scalar(o.phase),
        };
        let drift = OuPhaseNoise::new(o.drift[0], o.drift[1]).map_err(kernel_error)?;
        let mut builder = RfBuilder::new(cfg, data, priors, drift);
        if let Some(s) = o.baseline_sigma {
            builder = builder.baseline_calibration(V3::from_array(s));
        }
        if let Some(difference_hz) = columns.next() {
            builder = builder.differential_doppler(
                DifferentialDopplerData {
                    difference_hz,
                    sigma_antenna2_hz: o.sigmas[2],
                },
                Some((
                    sensor,
                    differential_baselines
                        .into_iter()
                        .map(V3::from_array)
                        .collect(),
                )),
            );
        }
        if let Some(s) = o.differential_sigma {
            builder = builder.differential_baseline_calibration(V3::from_array(s));
        }
        let problem = builder.build().map_err(kernel_error)?;
        let mut initial = problem.initial_parameters();
        if guess.len() != config.inner.mode().columns() {
            return Err(PyValueError::new_err("invalid RF guess dimension"));
        }
        initial.as_mut_slice()[..guess.len()].copy_from_slice(&guess);
        let fit = fit_passive_rf(&problem, &initial, &lm(o.iterations)).map_err(fit_error)?;
        Ok((
            fit.parameters.as_slice().to_vec(),
            rows(&fit.covariance),
            fit.rss,
            fit.iterations,
            fit.drift_rad,
            last(&times)?,
        ))
    })
}
