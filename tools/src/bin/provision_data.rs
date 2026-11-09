//! Explicit data provisioning, separate from numerical kernel execution.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tychonic::satkit::utils::update_datafiles(None, false)?;
    Ok(())
}
