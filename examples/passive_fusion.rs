//! Fit and stream all four calibrated observables. Requires Satkit data.
use tychonic::{
    OrbitFilter,
    estimate::{MeasurementModel, UpdateOptions},
    kernels::*,
    measurements::{self as m, Observable},
    numeris::{DynMatrix, DynVector, optim::LmSettings},
    satkit::{Duration, Instant},
};

// Satkit settings copies are explicit at this external API boundary.
#[allow(clippy::disallowed_methods)]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let epoch = Instant::from_unixtime(1704110400.25);
    let times: Vec<_> = (0..13)
        .map(|i| epoch + Duration::from_seconds(i as f64 * 10.0))
        .collect();
    let station = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let geometry = PreparedGeometry::new(&times, &vec![station; times.len()])?;
    let g = geometry.row(0).unwrap();
    let initial = v6_from(
        &(g.position() + g.rotation().to_gcrf(&v3(80e3, 40e3, 400e3))),
        &(g.velocity() + g.rotation().to_gcrf(&v3(7100.0, 3000.0, 0.0))),
    );
    let config = PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: 0.02 },
    )?;
    let baseline_enu_m = v3(59.0, 3.0, 0.2);
    let interferometer = Interferometer::new(2.23e9)?;
    let observables = vec![
        Observable::Doppler {
            carrier_hz: 2.23e9,
            offset_hz: 0.0,
        },
        Observable::PhaseDifference {
            interferometer,
            baseline_enu_m,
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        },
        Observable::DifferentialDoppler {
            interferometer,
            baseline_enu_m,
        },
        Observable::AzimuthElevation,
    ];

    // Replace synthetic rows with extracted receiver measurements in this order:
    // Doppler Hz, phase rad, differential Hz, azimuth rad, elevation rad.
    let truth = config.sample(&DynVector::from_slice(initial.as_slice()), &times)?;
    let values = (0..times.len())
        .map(|i| {
            let sensor =
                m::joint_sensor::<5>(geometry.row(i).unwrap(), times[i], observables.clone())?;
            Ok(sensor.evaluate(&truth.states()[i])?.0.as_slice().to_vec())
        })
        .collect::<Result<Vec<_>>>()?;

    // f1 and f1-f2 share antenna-1 noise. Phase and angles are independent
    // here; supply measured cross terms when they are correlated.
    let frequency = m::doppler_differential_covariance(0.5, 0.5)?;
    let mut covariance = DynMatrix::zeros(5, 5);
    for (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
        covariance[(i, j)] = frequency[(usize::from(i == 2), usize::from(j == 2))];
    }
    covariance[(1, 1)] = 0.01_f64.powi(2);
    covariance[(3, 3)] = 1e-4_f64.powi(2);
    covariance[(4, 4)] = 2e-4_f64.powi(2);
    let batch = m::joint(
        geometry,
        observables
            .into_iter()
            .map(|o| o.map_baseline(|b| vec![b; times.len()]))
            .collect(),
        values,
        vec![covariance; times.len()],
    )?;
    let mut guess = initial;
    guess[0] += 30.0;
    guess[4] += 0.02;
    let settings = LmSettings {
        grad_tol: 1e-2,
        x_tol: 1e-5,
        f_tol: 1e-5,
        ..LmSettings::default()
    };
    let batches = vec![batch];
    let fit = m::fit_with_options(
        PropagationConfig::new(
            config.reference_epoch(),
            config.settings().clone(),
            config.mode(),
        )?,
        &batches,
        &DynVector::from_slice(guess.as_slice()),
        None,
        &settings,
    )?;
    println!("Batch fit: RSS={}, iterations={}", fit.rss, fit.iterations);

    let covariance = M6::from_fn(|i, j| {
        if i == j {
            if i < 3 { 1e4 } else { 0.01 }
        } else {
            0.0
        }
    });
    let mut filter = OrbitFilter::new(config, guess, covariance)?;
    let updates = filter.filter_batches(
        &batches,
        |_, _, _| Ok(M6::zeros()),
        UpdateOptions::default(),
    )?;
    let smoothed = filter.smooth()?;
    println!(
        "Streaming: {} updates, {} smoothed epochs",
        updates.len(),
        smoothed.len()
    );
    Ok(())
}
