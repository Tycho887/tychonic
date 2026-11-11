#![allow(clippy::disallowed_methods)]
#[path = "support/mod.rs"]
mod copies;

use copies::copy_config;
use tychonic::{
    estimate::{MeasurementModel, OrbitEkf, PassiveRfEkf, UpdateOptions},
    kernels::*,
    measurements::{self as m, Observable},
    numeris::{DynMatrix, DynVector, Matrix, Vector},
    optim::{
        DifferentialDopplerData, ODProblem, PassiveRfData, PassiveRfProblem, RfExtensions, RfPrior,
    },
    satkit::{Duration, Instant},
};

struct Fixture {
    config: PropagationConfig,
    geometry: PreparedGeometry,
    state: V6,
    phase: Vec<V3>,
    differential: Vec<V3>,
    interferometer: Interferometer,
}
impl Fixture {
    fn new(spherical: bool) -> Self {
        let epoch = Instant::from_unixtime(1704110400.25);
        let times: Vec<_> = [20.0, 0.0, 20.0, 10.0, 40.0]
            .into_iter()
            .map(|dt| epoch + Duration::from_seconds(dt))
            .collect();
        let geometry = PreparedGeometry::new(
            &times,
            &[GroundStation {
                lat_deg: 40.0,
                lon_deg: -75.0,
                alt_m: 100.0,
            }; 5],
        )
        .unwrap();
        let g = geometry.row(1).unwrap();
        let state = v6_from(
            &(g.position() + g.rotation().to_gcrf(&v3(80e3, 40e3, 400e3))),
            &(g.velocity() + g.rotation().to_gcrf(&v3(7100.0, 3000.0, 0.0))),
        );
        let interferometer = Interferometer::new(2.23e9).unwrap();
        Self {
            config: PropagationConfig::new(
                epoch,
                od_prop_settings(),
                ParameterMode::StateOnly { fixed_drag: 0.02 },
            )
            .unwrap(),
            geometry,
            state,
            phase: vec![
                v3(59.0, 3.0, 0.2),
                v3(4.0, -30.0, 20.0),
                v3(-59.0, -3.0, -0.2),
                V3::zeros(),
                v3(10.0, 20.0, 30.0),
            ],
            differential: (0..5)
                .map(|i| v3(100e3 + i as f64 * 2000.0, -30e3, 5e3))
                .collect(),
            interferometer: if spherical {
                interferometer.with_spherical_phase()
            } else {
                interferometer
            },
        }
    }
    fn samples(&self) -> Sensitivities {
        self.config
            .sample(
                &DynVector::from_slice(self.state.as_slice()),
                self.geometry.timestamps(),
            )
            .unwrap()
    }
    fn values(&self, separate: bool, bias: f64, phase_offset: f64) -> Vec<Vec<f64>> {
        let samples = self.samples();
        let baselines = if separate {
            &self.differential
        } else {
            &self.phase
        };
        (0..self.geometry.len())
            .map(|i| {
                let g = self.geometry.row(i).unwrap();
                let p = self
                    .interferometer
                    .evaluate(&samples.states()[i], g, &self.phase[i], bias)
                    .unwrap();
                let d = self
                    .interferometer
                    .differential(
                        &samples.states()[i],
                        g,
                        &baselines[i],
                        self.geometry.timestamps()[i],
                    )
                    .unwrap();
                vec![p.doppler_hz, p.phase_rad + phase_offset, d.difference_hz]
            })
            .collect()
    }
    fn joint(&self) -> PreparedBatch<'_> {
        m::joint(
            &self.geometry,
            vec![
                Observable::DifferentialDoppler {
                    interferometer: self.interferometer,
                    baseline_enu_m: self.differential.to_vec(),
                },
                Observable::PhaseDifference {
                    interferometer: self.interferometer,
                    baseline_enu_m: self.phase.to_vec(),
                    frequency_offset_hz: 0.0,
                    phase_offset_rad: 0.0,
                },
                Observable::Doppler {
                    carrier_hz: 2.23e9,
                    offset_hz: 0.0,
                },
            ],
            self.values(true, 0.0, 0.0),
            (0..self.geometry.len())
                .map(|_| {
                    DynMatrix::from_fn(3, 3, |i, j| {
                        Matrix::new([[0.09, 0.0, 0.09], [0.0, 0.0001, 0.0], [0.09, 0.0, 0.25]])
                            [(i, j)]
                    })
                })
                .collect(),
        )
        .unwrap()
    }
    fn rf(&self, separate: bool) -> PassiveRfProblem<'_> {
        let values = self.values(separate, 2.0, 0.7);
        let data = PassiveRfData::new(
            &self.geometry,
            self.interferometer,
            self.phase.to_vec(),
            values.iter().map(|v| v[0]).collect(),
            Some(values.iter().map(|v| v[1]).collect()),
            0.3,
            0.01,
        )
        .unwrap();
        let mut mean = self.state.as_slice().to_vec();
        mean.extend([2.0, 0.7]);
        let prior = RfPrior::new(
            DynVector::from_slice(&mean),
            DynVector::from_slice(&[1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 10.0, 3.0]),
        )
        .unwrap();
        PassiveRfProblem::new_extended(
            copy_config(&self.config),
            data,
            prior,
            OuPhaseNoise::new(21600.0, 0.087).unwrap(),
            RfExtensions {
                baseline_sigma_m: Some(v3(0.005, 0.005, 0.005)),
                differential_doppler: Some(DifferentialDopplerData {
                    difference_hz: values.iter().map(|v| v[2]).collect(),
                    sigma_antenna2_hz: 0.4,
                }),
                differential_interferometer: separate.then_some(self.interferometer),
                differential_baselines_enu_m: separate.then(|| self.differential.to_vec()),
                differential_baseline_sigma_m: separate.then_some(v3(0.005, 0.005, 0.005)),
            },
        )
        .unwrap()
    }
}

fn check_jacobian(x: &DynVector<f64>, evaluate: impl Fn(&DynVector<f64>) -> Evaluation) {
    let e = evaluate(x);
    for k in 0..x.len() {
        let step = if k < 3 {
            0.5
        } else if k < 6 {
            0.01
        } else {
            0.001
        };
        let mut plus = DynVector::from_slice(x.as_slice());
        let mut minus = DynVector::from_slice(x.as_slice());
        plus[k] += step;
        minus[k] -= step;
        let a = evaluate(&plus);
        let b = evaluate(&minus);
        for row in 0..e.residual.len() {
            let fd = (a.residual[row] - b.residual[row]) / (2.0 * step);
            let h = e.jacobian[(row, k)];
            assert!(
                (fd - h).abs() < 2e-5 + 2e-4 * h.abs(),
                "[{row},{k}] {fd} vs {h}"
            );
        }
    }
}

#[test]
fn variable_joint_and_named_batches_keep_baselines_aligned_with_epochs() {
    for spherical in [false, true] {
        let f = Fixture::new(spherical);
        let values = f.values(true, 0.0, 0.0);
        assert!(values[0][1] * values[2][1] < 0.0);
        if !spherical {
            assert!((values[0][1] + values[2][1]).abs() < 1e-10);
        }
        let joint = f.joint();
        let phase = m::phase_difference(
            &f.geometry,
            f.interferometer,
            f.phase.to_vec(),
            values.iter().map(|v| v[1]).collect(),
            0.01,
        )
        .unwrap();
        let legacy = PreparedBatch::phase_difference(
            &f.geometry,
            f.interferometer,
            f.phase.to_vec(),
            values.iter().map(|v| v[1]).collect(),
            0.01,
        )
        .unwrap();
        let differential = m::differential_doppler(
            &f.geometry,
            f.interferometer,
            f.differential.to_vec(),
            values.iter().map(|v| v[2]).collect(),
            0.5,
        )
        .unwrap();
        let samples = f.samples();
        for batch in [&joint, &phase, &legacy, &differential] {
            assert!(batch.evaluate_from_sensitivities(&samples).unwrap().rss() < 1e-16);
        }
        let x = DynVector::from_slice(f.state.as_slice());
        check_jacobian(&x, |x| joint.evaluate(x, &f.config).unwrap());
        let batches = vec![joint, phase, legacy, differential];
        let problem = ODProblem::new(copy_config(&f.config), &batches).unwrap();
        assert!(problem.evaluate(&x).unwrap().rss() < 1e-16);
        check_jacobian(&x, |x| problem.evaluate(x).unwrap());
        check_calibrated_filter(&f, &batches[0]);
    }
}

fn check_calibrated_filter(f: &Fixture, batch: &PreparedBatch) {
    let mut automatic = OrbitEkf::<6>::new(copy_config(&f.config), f.state, M6::eye()).unwrap();
    let mut manual = OrbitEkf::<6>::new(copy_config(&f.config), f.state, M6::eye()).unwrap();
    let results = automatic
        .filter_batches(
            std::slice::from_ref(batch),
            |_, _, _| Ok(M6::zeros()),
            UpdateOptions::default(),
        )
        .unwrap();
    assert_eq!(
        results.iter().map(|r| r.row).collect::<Vec<_>>(),
        [1, 3, 0, 2, 4]
    );
    for i in [1, 3, 0, 2, 4] {
        let mut observation = batch.joint_observation::<3>(i).unwrap();
        observation.model = m::joint_sensor(
            f.geometry.row(i).unwrap(),
            f.geometry.timestamps()[i],
            vec![
                Observable::Doppler {
                    carrier_hz: 2.23e9,
                    offset_hz: 0.0,
                },
                Observable::PhaseDifference {
                    interferometer: f.interferometer,
                    baseline_enu_m: f.phase[i],
                    frequency_offset_hz: 0.0,
                    phase_offset_rad: 0.0,
                },
                Observable::DifferentialDoppler {
                    interferometer: f.interferometer,
                    baseline_enu_m: f.differential[i],
                },
            ],
        )
        .unwrap();
        manual
            .observe(&observation, None, UpdateOptions::default())
            .unwrap();
    }
    assert_eq!(automatic.state(), manual.state());
    assert_eq!(automatic.covariance(), manual.covariance());
}

#[test]
fn supplied_vectors_match_raw_gcrf_kernels_and_scalar_sensors() {
    let f = Fixture::new(false);
    let samples = f.samples();
    for i in 0..f.geometry.len() {
        let g = f.geometry.row(i).unwrap();
        let state = &samples.states()[i];
        let time = f.geometry.timestamps()[i];
        let p = phase_difference_partial(
            state,
            g.position(),
            g.velocity(),
            &g.rotation().to_gcrf(&f.phase[i]),
            2.23e9,
            0.0,
        )
        .unwrap();
        let omega = tychonic::satkit::frametransform::qitrf2gcrf_approx(&time)
            * v3(0.0, 0.0, tychonic::satkit::consts::OMEGA_EARTH);
        let d = differential_doppler_partial(
            state,
            g.position(),
            g.velocity(),
            &g.rotation().to_gcrf(&f.differential[i]),
            &omega,
            2.23e9,
        )
        .unwrap();
        let phase = m::phase_difference_sensor(g, time, f.interferometer, f.phase[i])
            .unwrap()
            .evaluate(state)
            .unwrap();
        let differential =
            m::differential_doppler_sensor(g, time, f.interferometer, f.differential[i])
                .unwrap()
                .evaluate(state)
                .unwrap();
        assert_eq!(phase.0[0], p.phase_rad);
        assert_eq!(phase.1, p.partial.block::<1, 6>(1, 0));
        assert_eq!(differential.0[0], d.difference_hz);
        assert_eq!(differential.1, d.partial);
    }
}

#[test]
fn variable_rf_calibration_columns_and_conditional_drift_match_numerical_derivatives() {
    for separate in [false, true] {
        let f = Fixture::new(true);
        let problem = f.rf(separate);
        let mut x = problem.initial_parameters();
        assert!(problem.evaluate(&x).unwrap().rss() < 1e-16);
        assert!(
            problem
                .conditional_drift(&x)
                .unwrap()
                .iter()
                .all(|v| v.abs() < 1e-10)
        );
        for k in 8..x.len() {
            x[k] = 0.002 * (k as f64 - 9.0);
        }
        check_jacobian(&x, |x| problem.evaluate(x).unwrap());
        assert!(
            problem
                .conditional_drift(&x)
                .unwrap()
                .iter()
                .any(|v| v.abs() > 1e-5)
        );
    }
}

fn check_rf_filter<const N: usize>(separate: bool) {
    let f = Fixture::new(true);
    let problem = f.rf(separate);
    let ext = problem.extensions();
    let make_filter = || {
        PassiveRfEkf::<N>::new_with_calibration(
            copy_config(&f.config),
            OuPhaseNoise::new(21600.0, 0.087).unwrap(),
            problem.prior(),
            ext.baseline_sigma_m,
            ext.differential_baseline_sigma_m,
        )
        .unwrap()
    };
    let mut automatic = make_filter();
    let mut manual = make_filter();
    automatic
        .filter_data_with_extensions(problem.data(), ext, UpdateOptions::default())
        .unwrap();
    let values = f.values(separate, 2.0, 0.7);
    let covariance = ext
        .differential_doppler
        .as_ref()
        .unwrap()
        .covariance(problem.data());
    for i in [1, 3, 0, 2, 4] {
        manual.predict_to(f.geometry.timestamps()[i], None).unwrap();
        manual
            .update_with_differential_baseline(
                &Vector::from_slice(&values[i]),
                f.geometry.row(i).unwrap(),
                f.interferometer,
                &f.phase[i],
                separate.then_some((f.interferometer, f.differential[i])),
                &covariance,
                UpdateOptions::default(),
            )
            .unwrap();
    }
    assert_eq!(automatic.state(), manual.state());
    assert_eq!(automatic.covariance(), manual.covariance());
    let before = (
        *automatic.state(),
        *automatic.covariance(),
        automatic.epoch(),
    );
    assert!(
        automatic
            .update(
                &Vector::from_array([0.0, 0.0]),
                f.geometry.row(0).unwrap(),
                f.interferometer,
                &v3(f64::NAN, 0.0, 0.0),
                &Matrix::eye(),
                UpdateOptions::default()
            )
            .is_err()
    );
    assert_eq!(
        before,
        (
            *automatic.state(),
            *automatic.covariance(),
            automatic.epoch()
        )
    );
}

#[test]
fn variable_rf_filter_uses_original_rows_and_preserves_state_on_invalid_baseline() {
    check_rf_filter::<12>(false);
    check_rf_filter::<15>(true);
}

fn check_invalid_baselines(f: &Fixture, baselines: Vec<V3>) {
    let n = f.geometry.len();
    assert!(
        m::phase_difference(
            &f.geometry,
            f.interferometer,
            baselines.to_vec(),
            vec![0.0; n],
            0.01
        )
        .is_err()
    );
    assert!(
        m::differential_doppler(
            &f.geometry,
            f.interferometer,
            baselines.to_vec(),
            vec![0.0; n],
            0.5
        )
        .is_err()
    );
    assert!(
        PreparedBatch::phase_difference(
            &f.geometry,
            f.interferometer,
            baselines.to_vec(),
            vec![0.0; n],
            0.01
        )
        .is_err()
    );
    assert!(
        PassiveRfData::new(
            &f.geometry,
            f.interferometer,
            baselines.to_vec(),
            vec![0.0; n],
            Some(vec![0.0; n]),
            0.3,
            0.01
        )
        .is_err()
    );
    assert!(
        m::joint(
            &f.geometry,
            vec![Observable::DifferentialDoppler {
                interferometer: f.interferometer,
                baseline_enu_m: baselines
            }],
            vec![vec![0.0]; n],
            vec![DynMatrix::from_fn(1, 1, |_, _| 1.0); n]
        )
        .is_err()
    );
}

#[test]
fn baseline_lengths_and_numeric_range_are_checked_at_measurement_boundaries() {
    let f = Fixture::new(false);
    for n in [0, 4, 6] {
        check_invalid_baselines(&f, vec![V3::zeros(); n]);
    }
    for invalid in [f64::NAN, f64::INFINITY, 1e308] {
        let mut baselines = f.phase.to_vec();
        baselines[2][1] = invalid;
        check_invalid_baselines(&f, baselines);
        assert!(
            m::phase_difference_sensor(
                f.geometry.row(0).unwrap(),
                f.geometry.timestamps()[0],
                f.interferometer,
                v3(invalid, 0.0, 0.0)
            )
            .is_err()
        );
    }
    assert!(
        m::phase_difference(
            &f.geometry,
            f.interferometer,
            vec![V3::zeros(); 5],
            vec![0.0; 5],
            0.01
        )
        .is_ok()
    );
}

fn check_invalid_differential_setup(f: &Fixture, ext: RfExtensions) {
    let problem = f.rf(true);
    assert!(
        PassiveRfProblem::new_extended(
            copy_config(&f.config),
            problem.data(),
            problem.prior(),
            OuPhaseNoise::new(21600.0, 0.087).unwrap(),
            &ext
        )
        .is_err()
    );
    let mut filter = PassiveRfEkf::<15>::new_with_calibration(
        copy_config(&f.config),
        OuPhaseNoise::new(21600.0, 0.087).unwrap(),
        problem.prior(),
        ext.baseline_sigma_m,
        ext.differential_baseline_sigma_m,
    )
    .unwrap();
    let before = (
        *filter.state(),
        *filter.covariance(),
        filter.epoch(),
        filter.timestamps().len(),
    );
    assert!(
        filter
            .filter_data_with_extensions(problem.data(), &ext, UpdateOptions::default())
            .is_err()
    );
    assert_eq!(
        before,
        (
            *filter.state(),
            *filter.covariance(),
            filter.epoch(),
            filter.timestamps().len()
        )
    );
}

#[test]
fn separate_differential_inputs_require_a_complete_validated_baseline_sequence() {
    let f = Fixture::new(true);
    let problem = f.rf(true);
    let mut missing = copies::copy_extensions(problem.extensions());
    missing.differential_baselines_enu_m = None;
    check_invalid_differential_setup(&f, missing);
    let mut extra = copies::copy_extensions(problem.extensions());
    extra.differential_interferometer = None;
    check_invalid_differential_setup(&f, extra);
    let mut short = copies::copy_extensions(problem.extensions());
    short.differential_baselines_enu_m.as_mut().unwrap().pop();
    check_invalid_differential_setup(&f, short);
    let mut invalid = copies::copy_extensions(problem.extensions());
    invalid.differential_baselines_enu_m.as_mut().unwrap()[2][0] = f64::NAN;
    check_invalid_differential_setup(&f, invalid);
}

#[test]
fn rf_builder_carries_separate_measurement_baselines_and_calibration() {
    use m::{GaussianPrior, RfBuilder, RfPriors};
    let f = Fixture::new(true);
    let problem = f.rf(true);
    let builder = RfBuilder::new(
        copy_config(&f.config),
        problem.data(),
        RfPriors {
            orbit_mean: DynVector::from_slice(f.state.as_slice()),
            orbit_sigma: DynVector::from_slice(&[1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0]),
            frequency_offset_hz: GaussianPrior {
                mean: 2.0,
                sigma: 10.0,
            },
            phase_offset_rad: GaussianPrior {
                mean: 0.7,
                sigma: 3.0,
            },
        },
        OuPhaseNoise::new(21600.0, 0.087).unwrap(),
    )
    .baseline_calibration(v3(0.005, 0.005, 0.005))
    .differential_doppler(
        DifferentialDopplerData {
            difference_hz: problem
                .extensions()
                .differential_doppler
                .as_ref()
                .unwrap()
                .difference_hz
                .to_vec(),
            sigma_antenna2_hz: 0.4,
        },
        Some((f.interferometer, f.differential.to_vec())),
    )
    .differential_baseline_calibration(v3(0.005, 0.005, 0.005));
    let built = builder.build().unwrap();
    let x = built.initial_parameters();
    assert_eq!(
        built.evaluate(&x).unwrap().jacobian,
        problem.evaluate(&x).unwrap().jacobian
    );
    let mut filter = builder.filter::<15>().unwrap();
    builder
        .filter_data(&mut filter, UpdateOptions::default())
        .unwrap();
}

#[test]
fn varying_baselines_support_covariance_fitting_across_distinct_schedules() {
    use tychonic::numeris::optim::LmSettings;
    let f = Fixture::new(true);
    let values = f.values(true, 0.0, 0.0);
    let rows = [1, 2, 4];
    let geometry = PreparedGeometry::new(
        &rows.map(|i| f.geometry.timestamps()[i]),
        &rows.map(|i| f.geometry.stations()[i]),
    )
    .unwrap();
    let second = m::differential_doppler(
        geometry,
        f.interferometer,
        rows.map(|i| f.differential[i]).to_vec(),
        rows.map(|i| values[i][2]).to_vec(),
        0.5,
    )
    .unwrap();
    let batches = vec![f.joint(), second];
    let mut guess = DynVector::from_slice(f.state.as_slice());
    guess[0] += 10.0;
    guess[4] += 0.005;
    let fit = m::fit_with_covariance(
        copy_config(&f.config),
        &batches,
        &guess,
        None,
        &LmSettings {
            grad_tol: 1e-3,
            x_tol: 1e-6,
            f_tol: 1e-6,
            ..LmSettings::default()
        },
    )
    .unwrap();
    assert!(fit.rss < 1e-8);
    assert!(fit.covariance.cholesky().is_ok());
}
