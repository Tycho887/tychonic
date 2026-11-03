//! A standalone geometric observation: no external ephemeris data is needed.
use tychonic::kernels::{differential_doppler_partial, v3, v6_from};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let station_position = v3(6_378_137.0, 0.0, 0.0);
    let earth_rotation = v3(0.0, 0.0, tychonic::satkit::consts::OMEGA_EARTH);
    let station_velocity = earth_rotation.cross(&station_position);
    let baseline = v3(0.0, 59.0, 0.0); // antenna 1 minus antenna 2, in GCRF
    let state = v6_from(
        &v3(6_950_000.0, 100_000.0, 200_000.0),
        &v3(0.0, 7_500.0, 0.0),
    );
    let observation = differential_doppler_partial(
        &state,
        &station_position,
        &station_velocity,
        &baseline,
        &earth_rotation,
        2.23e9,
    )?;
    println!("f1 - f2: {:.6} Hz", observation.difference_hz);
    println!("state Jacobian: {:?}", observation.partial);
    Ok(())
}
