//! Seeded BC-MPC discovery with an independent simulated satellite and receiver.
#[path = "support/acquisition.rs"]
mod simulation;

use simulation::{GaussianReceiver, RECEIVER_SEED, RESAMPLING_SEED, run_simulation};
use tychonic::detection::SearchStatus;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = run_simulation(GaussianReceiver)?;
    println!(
        "BC-MPC: receiver seed {RECEIVER_SEED}, resampling seed {RESAMPLING_SEED}, initial error {:.2} deg",
        result.initial_pointing_error_rad.to_degrees(),
    );
    println!("cycle elapsed_s error_deg receiver_p draw slew_s remaining_p resampled outcome");
    for step in &result.records {
        println!(
            "{:>5} {:>9.1} {:>9.2} {:>10.4} {:>4.2} {:>6.2} {:>11.4} {:>9} {:?}",
            step.cycle,
            step.elapsed_s,
            step.pointing_error_rad.to_degrees(),
            step.detection_probability,
            step.receiver_draw,
            step.slew_time_s,
            step.predicted_acquisition_probability,
            step.resampled,
            step.observation,
        );
    }
    println!(
        "{:?} after {} cycles ({:.1} simulated seconds)",
        result.status,
        result.controller_cycles,
        (result.plant_time - result.setup.start).as_seconds(),
    );
    if result.status != SearchStatus::Acquired {
        return Err("seeded example exhausted its budget without acquisition".into());
    }
    Ok(())
}
