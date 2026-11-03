use numeris::{
    DynMatrix, DynVector,
    optim::{LmSettings, least_squares_lm_dyn},
};
use satkit::{Duration, Instant};
use tychonic::{
    kernels::*,
    optim::{
        ContactBiasMap, ContactBiasProblem, ContactId, ContactRangeRate, PointingConstraint,
        fit_bootstrap, fit_varpro,
    },
};

fn epoch() -> Instant {
    Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap()
}

fn site() -> GroundStation {
    GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    }
}

fn second_site() -> GroundStation {
    GroundStation {
        lat_deg: 40.3,
        lon_deg: -74.2,
        alt_m: 400.0,
    }
}

fn close(actual: f64, expected: f64, atol: f64, rtol: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "{context}: got {actual:.12e}, expected {expected:.12e}"
    );
}

fn leo_state(epoch: Instant) -> DynVector<f64> {
    let (centre, basis) = precompute_station_geometry(&[epoch], &[site()]).unwrap();
    let state = v6_from(
        &(centre.position[0] + basis.up(0) * 300e3 + basis.east(0) * 80e3 + basis.north(0) * 40e3),
        &(centre.velocity[0] + basis.east(0) * 7100.0 + basis.north(0) * 3000.0),
    );
    DynVector::from_slice(state.as_slice())
}

fn state_config(epoch: Instant) -> PropagationConfig {
    PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: 0.02 },
    )
    .unwrap()
}

fn state_and_drag_config(epoch: Instant) -> PropagationConfig {
    PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateAndDrag { delta_cd: 1e-4 },
    )
    .unwrap()
}

fn at_times(epoch: Instant, offsets: &[f64]) -> Vec<Instant> {
    offsets
        .iter()
        .map(|dt| epoch + Duration::from_seconds(*dt))
        .collect()
}

fn contact_indices(contacts: &[ContactId]) -> Vec<usize> {
    let mut seen: Vec<ContactId> = Vec::new();
    contacts
        .iter()
        .map(
            |contact| match seen.iter().position(|value| value == contact) {
                Some(index) => index,
                None => {
                    seen.push(*contact);
                    seen.len() - 1
                }
            },
        )
        .collect()
}

fn range_rates_from_state(
    config: &PropagationConfig,
    parameters: &DynVector<f64>,
    times: &[Instant],
    stations: &[GroundStation],
    contacts: &[ContactId],
    biases: &[f64],
) -> Vec<ContactRangeRate> {
    let geometry = PreparedGeometry::new(times, stations).unwrap();
    let sensitivities = config.sample(parameters, times).unwrap();
    let indices = contact_indices(contacts);
    times
        .iter()
        .enumerate()
        .map(|(i, time)| {
            let row = geometry.row(i).unwrap();
            let (range_rate, _) = range_rate_partial(
                &r_of(&sensitivities.states[i]),
                &v_of(&sensitivities.states[i]),
                row.position(),
                row.velocity(),
            )
            .unwrap();
            ContactRangeRate {
                time: *time,
                station: stations[i],
                contact: contacts[i],
                range_rate: range_rate + biases[indices[i]],
            }
        })
        .collect()
}

fn pointing_from_state(
    config: &PropagationConfig,
    parameters: &DynVector<f64>,
    times: &[Instant],
    stations: &[GroundStation],
) -> Vec<PointingConstraint> {
    let geometry = PreparedGeometry::new(times, stations).unwrap();
    let sensitivities = config.sample(parameters, times).unwrap();
    times
        .iter()
        .enumerate()
        .map(|(i, time)| {
            let row = geometry.row(i).unwrap();
            let ([azimuth, elevation], _) = az_el_partial_prepared(
                &r_of(&sensitivities.states[i]),
                row.position(),
                row.rotation(),
            )
            .unwrap();
            PointingConstraint {
                time: *time,
                station: stations[i],
                azimuth,
                elevation,
            }
        })
        .collect()
}

fn one_hot_bias_matrix(rows: usize, contacts: usize, indices: &[usize]) -> DynMatrix<f64> {
    assert_eq!(rows, indices.len());
    let mut matrix = DynMatrix::<f64>::zeros(rows, contacts);
    for (row, &contact) in indices.iter().enumerate() {
        matrix[(row, contact)] = 1.0;
    }
    matrix
}

// (1) Projection algebra through ContactBiasMap on a synthetic evaluation.
#[test]
fn contact_bias_projection_is_idempotent_and_ignores_pointing_rows() {
    let row_contact = [
        Some(0),
        Some(1),
        None,
        Some(2),
        Some(0),
        Some(1),
        Some(2),
        None,
        Some(0),
    ];
    let row_weight = [2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.5];
    let contacts = 3;
    let rows = row_contact.len();
    let columns = 3;
    let map = ContactBiasMap::new(&row_contact, &row_weight, contacts).unwrap();
    assert_eq!(map.contacts(), contacts);
    assert_eq!(map.bias_rows(), 7);

    let original = Evaluation {
        residual: DynVector::from_vec(
            (0..rows)
                .map(|row| 0.37 * (row as f64 - 2.0) + 0.11)
                .collect(),
        ),
        jacobian: DynMatrix::from_fn(rows, columns, |row, column| {
            ((row + 1) as f64 * (column as f64 + 0.5)).sin()
        }),
    };
    let mut projected = original.clone();
    map.project(&mut projected).unwrap();

    // After projection, every contact's residual and Jacobian column have zero
    // weighted sum.
    for contact in 0..contacts {
        let residual_sum: f64 = (0..rows)
            .filter(|&row| row_contact[row] == Some(contact))
            .map(|row| row_weight[row] * projected.residual[row])
            .sum();
        assert!(residual_sum.abs() < 1e-13, "contact {contact} residual sum");
        for column in 0..columns {
            let jacobian_sum: f64 = (0..rows)
                .filter(|&row| row_contact[row] == Some(contact))
                .map(|row| row_weight[row] * projected.jacobian[(row, column)])
                .sum();
            assert!(
                jacobian_sum.abs() < 1e-13,
                "contact {contact}, column {column} Jacobian sum"
            );
        }
    }

    // P is idempotent: applying the projector again changes nothing.
    let mut twice = projected.clone();
    map.project(&mut twice).unwrap();
    for row in 0..rows {
        close(
            twice.residual[row],
            projected.residual[row],
            1e-14,
            1e-13,
            "second projection residual",
        );
        for column in 0..columns {
            close(
                twice.jacobian[(row, column)],
                projected.jacobian[(row, column)],
                1e-14,
                1e-13,
                "second projection Jacobian",
            );
        }
    }

    // Rows without a bias column are bit-identical.
    for (row, contact) in row_contact.iter().enumerate() {
        if contact.is_some() {
            continue;
        }
        assert_eq!(
            projected.residual[row].to_bits(),
            original.residual[row].to_bits()
        );
        for column in 0..columns {
            assert_eq!(
                projected.jacobian[(row, column)].to_bits(),
                original.jacobian[(row, column)].to_bits()
            );
        }
    }

    // solve() is the direct weighted contact mean of the raw residual.
    let solved = map.solve(&original.residual).unwrap();
    assert_eq!(solved.len(), contacts);
    for (contact, value) in solved.iter().enumerate() {
        let mut weighted_sum = 0.0;
        let mut norm = 0.0;
        for row in 0..rows {
            if row_contact[row] != Some(contact) {
                continue;
            }
            weighted_sum += row_weight[row] * original.residual[row];
            norm += row_weight[row] * row_weight[row];
        }
        close(
            *value,
            -weighted_sum / norm,
            1e-14,
            1e-13,
            "direct weighted mean",
        );
    }
}

fn short_arc_problem() -> (ContactBiasProblem, DynVector<f64>, Vec<f64>) {
    let epoch = epoch();
    let truth = leo_state(epoch);
    let config = state_config(epoch);
    let offsets: Vec<f64> = (0..12).map(|i| -300.0 + 50.0 * i as f64).collect();
    let times = at_times(epoch, &offsets);
    let stations = vec![site(); times.len()];
    let contact_ids = [
        ContactId::new(0x1111),
        ContactId::new(0x2222),
        ContactId::new(0x3333),
    ];
    let contacts: Vec<_> = (0..times.len()).map(|i| contact_ids[i / 4]).collect();
    let biases = [0.3, -0.5, 0.8];
    let range_rates =
        range_rates_from_state(&config, &truth, &times, &stations, &contacts, &biases);
    let pointing_times = at_times(epoch, &[-180.0, 160.0]);
    let pointing_stations = [site(), second_site()];
    let pointing = pointing_from_state(&config, &truth, &pointing_times, &pointing_stations);
    let noise = MeasurementNoise {
        sigma_rr: 0.04,
        sigma_az: 1e-3,
        sigma_el: 1e-3,
    };
    let problem = ContactBiasProblem::new(config, &range_rates, &pointing, noise).unwrap();
    (problem, truth, biases.to_vec())
}

// (2) The projected Jacobian matches central differences of the reduced residual.
#[test]
fn reduced_jacobian_matches_central_differences_on_a_short_arc() {
    let (problem, truth, _) = short_arc_problem();
    let mut guess = truth.clone();
    guess[0] += 4.0;
    guess[2] -= 3.0;
    guess[3] += 0.002;
    guess[5] -= 0.001;
    let reduced = problem.evaluate_reduced(&guess).unwrap();
    let raw = problem.evaluate(&guess).unwrap();
    let range_rate_rows = 12;
    for row in range_rate_rows..raw.residual.len() {
        assert_eq!(
            reduced.residual[row].to_bits(),
            raw.residual[row].to_bits(),
            "pointing residual row {row} changed"
        );
        for column in 0..raw.jacobian.ncols() {
            assert_eq!(
                reduced.jacobian[(row, column)].to_bits(),
                raw.jacobian[(row, column)].to_bits(),
                "pointing Jacobian row {row}, column {column} changed"
            );
        }
    }
    let steps = [1.0, 1.0, 1.0, 0.01, 0.01, 0.01];
    for column in 0..guess.len() {
        let mut plus = guess.clone();
        let mut minus = guess.clone();
        plus[column] += steps[column];
        minus[column] -= steps[column];
        let positive = problem.evaluate_reduced(&plus).unwrap();
        let negative = problem.evaluate_reduced(&minus).unwrap();
        for row in 0..reduced.residual.len() {
            let difference =
                (positive.residual[row] - negative.residual[row]) / (2.0 * steps[column]);
            close(
                reduced.jacobian[(row, column)],
                difference,
                2e-5,
                3e-4,
                &format!("reduced Jacobian row {row}, column {column}"),
            );
        }
    }
}

fn multi_contact_problem() -> (
    ContactBiasProblem,
    DynVector<f64>,
    Vec<f64>,
    Vec<ContactId>,
    MeasurementNoise,
) {
    let epoch = epoch();
    let truth = leo_state(epoch);
    let config = state_config(epoch);
    let offsets: Vec<f64> = (0..24).map(|i| -600.0 + 50.0 * i as f64).collect();
    let times = at_times(epoch, &offsets);
    let stations: Vec<_> = (0..times.len())
        .map(|i| if i % 2 == 0 { site() } else { second_site() })
        .collect();
    let contact_ids = [
        ContactId::new(0xAAAA),
        ContactId::new(0xBBBB),
        ContactId::new(0xCCCC),
        ContactId::new(0xDDDD),
    ];
    let contacts: Vec<_> = (0..times.len()).map(|i| contact_ids[i / 6]).collect();
    let biases = [0.4, -0.7, 0.25, 1.0];
    let range_rates =
        range_rates_from_state(&config, &truth, &times, &stations, &contacts, &biases);
    let noise = MeasurementNoise {
        sigma_rr: 0.02,
        sigma_az: 1e-3,
        sigma_el: 1e-3,
    };
    let problem = ContactBiasProblem::new(config, &range_rates, &[], noise).unwrap();
    (problem, truth, biases.to_vec(), contacts, noise)
}

// (3) Varpro recovers injected per-contact biases and the state, and agrees with
// an independent augmented LM on [J | A].
#[test]
fn varpro_multi_contact_fit_matches_the_augmented_joint_lm() {
    let (problem, truth, biases, contacts, noise) = multi_contact_problem();
    let mut guess = truth.clone();
    guess[0] += 20.0;
    guess[1] -= 15.0;
    guess[2] += 12.0;
    guess[3] += 0.01;
    guess[4] -= 0.005;
    guess[5] += 0.004;
    let settings = LmSettings {
        grad_tol: 1e-8,
        x_tol: 1e-12,
        f_tol: 1e-14,
        max_iter: 100,
        ..LmSettings::default()
    };
    let fit = fit_varpro(problem.clone(), &guess, None, &settings).unwrap();
    assert!(fit.rss < 1e-8, "reduced RSS {}", fit.rss);
    assert!(fit.iterations > 0);
    let position_error = (0..3)
        .map(|k| (fit.parameters[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    let velocity_error = (3..6)
        .map(|k| (fit.parameters[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(position_error < 0.01, "position error {position_error} m");
    assert!(velocity_error < 1e-4, "velocity error {velocity_error} m/s");
    for (actual, expected) in fit.biases.iter().zip(&biases) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "bias {actual} vs {expected}"
        );
    }

    // Rebuild the joint problem [x; b] with one-hot bias columns. The bias
    // variable is whitened, so the physical bias is sigma_rr * b.
    let columns = 6;
    let indices = contact_indices(&contacts);
    let rows = indices.len();
    let contact_count = biases.len();
    let matrix = one_hot_bias_matrix(rows, contact_count, &indices);
    let scales = [100.0, 100.0, 100.0, 0.1, 0.1, 0.1];
    let residual = |augmented: &DynVector<f64>| {
        let mut parameters = guess.clone();
        for k in 0..columns {
            parameters[k] += scales[k] * augmented[k];
        }
        let raw = problem.inner().evaluate(&parameters).unwrap();
        let mut values = raw.residual;
        for row in 0..rows {
            for contact in 0..contact_count {
                values[row] += matrix[(row, contact)] * augmented[columns + contact];
            }
        }
        values
    };
    let jacobian = |augmented: &DynVector<f64>| {
        let mut parameters = guess.clone();
        for k in 0..columns {
            parameters[k] += scales[k] * augmented[k];
        }
        let raw = problem.inner().evaluate(&parameters).unwrap();
        let mut joint = DynMatrix::<f64>::zeros(rows, columns + contact_count);
        for row in 0..rows {
            for k in 0..columns {
                joint[(row, k)] = raw.jacobian[(row, k)] * scales[k];
            }
            for contact in 0..contact_count {
                joint[(row, columns + contact)] = matrix[(row, contact)];
            }
        }
        joint
    };
    let augmented = least_squares_lm_dyn(
        residual,
        jacobian,
        &DynVector::zeros(columns + contact_count),
        &settings,
    )
    .unwrap();
    let mut joint_parameters = guess.clone();
    for k in 0..columns {
        joint_parameters[k] += scales[k] * augmented.x[k];
    }
    for k in 0..columns {
        close(
            joint_parameters[k],
            fit.parameters[k],
            1e-3,
            1e-9,
            "augmented state component",
        );
    }
    for (contact, &injected) in biases.iter().enumerate() {
        let physical_bias = augmented.x[columns + contact] * noise.sigma_rr;
        assert!(
            (physical_bias - fit.biases[contact]).abs() < 1e-4,
            "augmented bias {physical_bias} vs {}",
            fit.biases[contact]
        );
        assert!(
            (physical_bias - injected).abs() < 1e-4,
            "augmented bias {physical_bias} vs injected {injected}"
        );
    }
}

// (4) Covariance equals the Schur complement and the state block of the joint
// inverse information matrix.
#[test]
fn covariance_matches_the_schur_complement_and_joint_state_block() {
    let (problem, truth, _, contacts, _) = multi_contact_problem();
    let mut guess = truth.clone();
    guess[0] += 10.0;
    guess[1] -= 8.0;
    guess[3] += 0.01;
    let raw = problem.evaluate(&guess).unwrap();
    let reduced = problem.evaluate_reduced(&guess).unwrap();
    let columns = 6;
    let contact_count = problem.bias_map().contacts();
    let rows = raw.residual.len();
    let dof = rows - columns - contact_count;
    assert!(dof > 0);

    let information = &reduced.jacobian.transpose() * &reduced.jacobian;
    let schur = information.inverse().unwrap() * (reduced.rss() / dof as f64);
    let covariance = problem.covariance(&guess).unwrap();
    for row in 0..columns {
        for column in 0..columns {
            close(
                covariance[(row, column)],
                schur[(row, column)],
                1e-12,
                1e-6,
                "Schur covariance",
            );
        }
    }

    let indices = contact_indices(&contacts);
    let matrix = one_hot_bias_matrix(rows, contact_count, &indices);
    let jtj = &raw.jacobian.transpose() * &raw.jacobian;
    let jta = &raw.jacobian.transpose() * &matrix;
    let atj = &matrix.transpose() * &raw.jacobian;
    let ata = &matrix.transpose() * &matrix;
    let mut joint_information =
        DynMatrix::<f64>::zeros(columns + contact_count, columns + contact_count);
    joint_information.set_block(0, 0, &jtj);
    joint_information.set_block(0, columns, &jta);
    joint_information.set_block(columns, 0, &atj);
    joint_information.set_block(columns, columns, &ata);
    let joint_inverse = joint_information.inverse().unwrap();
    let state_block = joint_inverse.block(0, 0, columns, columns);
    for row in 0..columns {
        for column in 0..columns {
            close(
                covariance[(row, column)],
                state_block[(row, column)] * (reduced.rss() / dof as f64),
                1e-12,
                1e-6,
                "joint state block",
            );
        }
    }
}

// (5) Interleaved repeated contact ids share one bias; pointing rows do not.
#[test]
fn interleaved_contact_ids_share_a_bias_and_pointing_rows_are_excluded() {
    let epoch = epoch();
    let truth = leo_state(epoch);
    let config = state_config(epoch);
    let offsets = [-120.0, -80.0, -40.0, 0.0, 40.0];
    let times = at_times(epoch, &offsets);
    let stations = vec![site(); times.len()];
    let first = ContactId::new(0x0101);
    let second = ContactId::new(0x0202);
    let contacts = [first, second, first, second, first];
    let biases = [0.6, -0.35];
    let range_rates =
        range_rates_from_state(&config, &truth, &times, &stations, &contacts, &biases);
    let pointing = pointing_from_state(&config, &truth, &[times[0]], &[site()]);
    let noise = MeasurementNoise {
        sigma_rr: 0.03,
        sigma_az: 1e-3,
        sigma_el: 1e-3,
    };
    let problem = ContactBiasProblem::new(config, &range_rates, &pointing, noise).unwrap();
    assert_eq!(problem.contacts(), &[first, second]);
    assert_eq!(problem.bias_map().contacts(), 2);
    assert_eq!(problem.bias_map().bias_rows(), 5);
    assert_eq!(problem.inner().rows(), 5 + 2);
    let solved = problem.solve_biases(&truth).unwrap();
    assert_eq!(solved.len(), 2);
    for contact in 0..2 {
        close(
            solved[contact],
            biases[contact],
            1e-12,
            1e-11,
            "grouped contact bias",
        );
    }
}

// (6) Constructor and map error cases.
#[test]
fn invalid_inputs_and_insufficient_degrees_of_freedom_are_rejected() {
    let epoch = epoch();
    let truth = leo_state(epoch);
    let config = state_config(epoch);
    let noise = MeasurementNoise {
        sigma_rr: 0.05,
        sigma_az: 1e-3,
        sigma_el: 1e-3,
    };
    assert!(matches!(
        ContactBiasProblem::new(config.clone(), &[], &[], noise),
        Err(KernelError::InvalidInput(_))
    ));

    assert!(matches!(
        ContactBiasMap::new(&[Some(0)], &[1.0], 2),
        Err(KernelError::InvalidInput(_))
    ));
    assert!(matches!(
        ContactBiasMap::new(&[Some(0), None], &[1.0], 1),
        Err(KernelError::InvalidInput(_))
    ));
    for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ContactBiasMap::new(&[Some(0)], &[weight], 1),
            Err(KernelError::InvalidInput(_))
        ));
    }
    assert!(matches!(
        ContactBiasMap::new(&[Some(1)], &[1.0], 1),
        Err(KernelError::InvalidInput(_))
    ));
    let map = ContactBiasMap::new(&[Some(0), None], &[1.0, 0.0], 1).unwrap();
    let mut short = Evaluation {
        residual: DynVector::zeros(1),
        jacobian: DynMatrix::<f64>::zeros(1, 2),
    };
    assert!(matches!(
        map.project(&mut short),
        Err(KernelError::InvalidInput(_))
    ));
    assert!(matches!(
        map.solve(&DynVector::zeros(1)),
        Err(KernelError::InvalidInput(_))
    ));

    let times = [epoch];
    let range_rates = range_rates_from_state(
        &config,
        &truth,
        &times,
        &[site()],
        &[ContactId::new(1)],
        &[0.1],
    );
    let problem = ContactBiasProblem::new(config, &range_rates, &[], noise).unwrap();
    assert!(matches!(
        problem.covariance(&truth),
        Err(KernelError::InvalidInput(_))
    ));
}

// (7) 24 h bootstrap: multi-contact range-rate with injected biases plus weak
// pointing from a perturbed old trajectory. This is slow; run it explicitly:
//   SATKIT_OFFLINE=1 cargo test --locked --test bootstrap -- --ignored
#[test]
#[ignore = "24 h state-and-drag bootstrap; run with --ignored"]
fn bootstrap_24_hour_fit_recovers_state_and_drag_from_weak_pointing() {
    let epoch = epoch();
    let truth_drag = 0.02;
    let mut truth_parameters = leo_state(epoch).as_slice().to_vec();
    truth_parameters.push(truth_drag);
    let truth = DynVector::from_vec(truth_parameters);
    let config = state_and_drag_config(epoch);

    let offsets: Vec<f64> = (0..=144).map(|i| 600.0 * i as f64).collect();
    let times = at_times(epoch, &offsets);
    let stations: Vec<_> = (0..times.len())
        .map(|i| if i % 2 == 0 { site() } else { second_site() })
        .collect();
    let contacts: Vec<_> = (0..times.len())
        .map(|i| ContactId::new(0xB007_0000 + (i / 18) as u128))
        .collect();
    let biases: Vec<f64> = (0..=8)
        .map(|contact| 0.15 * (1.7 * contact as f64).sin())
        .collect();
    let range_rates =
        range_rates_from_state(&config, &truth, &times, &stations, &contacts, &biases);

    // "Old" trajectory: perturbed epoch state and drag. Pointing pseudo-
    // measurements come from this trajectory, so they pull only weakly.
    let mut old = truth.clone();
    old[0] += 800.0;
    old[1] -= 600.0;
    old[2] += 400.0;
    old[3] += 0.8;
    old[4] -= 0.6;
    old[5] += 0.4;
    old[6] = 0.018;
    let pointing_offsets: Vec<f64> = (0..=48).map(|i| 1800.0 * i as f64).collect();
    let pointing_times = at_times(epoch, &pointing_offsets);
    let pointing_stations = vec![site(); pointing_times.len()];
    let pointing = pointing_from_state(&config, &old, &pointing_times, &pointing_stations);
    let noise = MeasurementNoise {
        sigma_rr: 0.05,
        sigma_az: 3e-3,
        sigma_el: 3e-3,
    };
    let problem = ContactBiasProblem::new(config.clone(), &range_rates, &pointing, noise).unwrap();
    let initial_rss = problem.evaluate_reduced(&old).unwrap().rss();
    let settings = LmSettings {
        grad_tol: 1e-4,
        x_tol: 1e-7,
        f_tol: 1e-8,
        max_iter: 40,
        ..LmSettings::default()
    };
    let fit = fit_bootstrap(
        config,
        &range_rates,
        &pointing,
        noise,
        &old,
        None,
        &settings,
    )
    .unwrap();
    assert!(fit.rss < initial_rss, "fit did not improve the residuals");
    let position_error = (0..3)
        .map(|k| (fit.parameters[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    let velocity_error = (3..6)
        .map(|k| (fit.parameters[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(position_error < 5000.0, "position error {position_error} m");
    assert!(velocity_error < 0.5, "velocity error {velocity_error} m/s");
    assert!(
        (fit.parameters[6] - truth_drag).abs() < 0.01,
        "drag error {}",
        fit.parameters[6] - truth_drag
    );
    for (actual, expected) in fit.biases.iter().zip(&biases) {
        assert!(
            (actual - expected).abs() < 0.1,
            "bias {actual} vs {expected}"
        );
    }
}
