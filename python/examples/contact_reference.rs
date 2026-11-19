//! Export the three-contact Rust qualification data for the installed-wheel tests.
#[path = "../../constrained/tests/support/mod.rs"]
#[allow(dead_code)] // Other constructor variants are used by the constrained solver tests.
mod support;

fn main() {
    let f = support::Fixture::with_space_weather([-4600., 0., 4800.], false);
    let problem = f.model(tychonic_constrained::PointingSelection::default(), 0.0003);
    let references = problem.evaluate(&f.truth).unwrap();
    let rows:Vec<_>=f.rates.iter().zip(&f.angles).zip(&f.sigmas).map(|((r,a),sigma)|serde_json::json!({
        "id":r.contact.to_string(),"epoch":r.time.as_rfc3339(),"station":[r.station.lat_deg,r.station.lon_deg,r.station.alt_m],
        "rate":r.range_rate,"az":a.azimuth,"el":a.elevation,"sigma":sigma
    })).collect();
    println!(
        "{}",
        serde_json::json!({"seed":f.seed.as_slice(),"truth":f.truth.as_slice(),"epoch":f.epoch.as_rfc3339(),"biases":f.biases,"rss":references.rss,"rows":rows})
    );
}
