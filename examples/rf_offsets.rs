//! Named priors for existing RF offset/drift batch and streaming models.
use tychonic::{
    estimate::UpdateOptions,
    kernels::*,
    measurements::{GaussianPrior, RfBuilder, RfPriors},
    numeris::DynVector,
    optim::PassiveRfData,
    satkit::{Duration, Instant},
};

// Satkit settings copies are explicit at this external API boundary.
#[allow(clippy::disallowed_methods)]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let epoch = Instant::from_unixtime(1704110400.25);
    let times: Vec<_> = (0..13)
        .map(|i| epoch + Duration::from_seconds(i as f64 * 10.0))
        .collect();
    let geometry = PreparedGeometry::new(
        &times,
        &vec![
            GroundStation {
                lat_deg: 40.0,
                lon_deg: -75.0,
                alt_m: 100.0,
            };
            times.len()
        ],
    )?;
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
    let interferometer = Interferometer::new(v3(59.0, 3.0, 0.2), 2.23e9)?;
    let orbit_mean = DynVector::from_slice(initial.as_slice());
    let states = config.sample(&orbit_mean, &times)?;
    let predictions = (0..times.len())
        .map(|i| interferometer.evaluate(&states.states()[i], geometry.row(i).unwrap(), 3.0))
        .collect::<Result<Vec<_>>>()?;
    // Replace these synthetic values with extracted receiver observations.
    let data = PassiveRfData::new(
        geometry,
        interferometer,
        predictions.iter().map(|p| p.doppler_hz).collect(),
        Some(predictions.iter().map(|p| p.phase_rad + 0.2).collect()),
        0.5,
        0.01,
    )?;
    let priors = RfPriors {
        orbit_mean,
        orbit_sigma: DynVector::from_slice(&[100.0, 100.0, 100.0, 0.1, 0.1, 0.1]),
        frequency_offset_hz: GaussianPrior {
            mean: 3.0,
            sigma: 1.0,
        },
        phase_offset_rad: GaussianPrior {
            mean: 0.2,
            sigma: 0.5,
        },
    };
    let builder = RfBuilder::new(config, data, priors, OuPhaseNoise::new(21600.0, 0.1)?);
    let fit = builder.fit()?;
    let mut filter = builder.filter::<9>()?;
    let updates = builder.filter_data(&mut filter, UpdateOptions::default())?;
    println!(
        "RF fit RSS={}; {} streaming updates",
        fit.rss,
        updates.len()
    );
    Ok(())
}
