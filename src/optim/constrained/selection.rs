use super::*;
use satkit::Duration;

pub(super) fn select(
    config: &PropagationConfig,
    seed: &V7,
    geometry: &PreparedGeometry,
    offset: usize,
    readings: &[ContactPointing],
    selection: PointingSelection,
) -> Result<Vec<Cone>> {
    let mut passes = BTreeMap::<ContactId, BTreeMap<Instant, Cone>>::new();
    for (i, &pointing) in readings.iter().enumerate() {
        let station_row = offset + i;
        let site = station_at(geometry, station_row)?;
        let direction =
            convert_az_el_to_gcrf_prepared(pointing.azimuth, pointing.elevation, site.rotation())?;
        passes
            .entry(pointing.contact)
            .or_default()
            .entry(pointing.time)
            .or_insert(Cone {
                pointing,
                station_row,
                direction,
                epoch: 0,
            });
    }
    let PointingSelection::NearTca(k) = selection else {
        return Ok(passes
            .into_values()
            .flat_map(|pass| pass.into_values())
            .collect());
    };
    let times: BTreeSet<_> = passes
        .values()
        .filter(|pass| pass.len() > k)
        .flat_map(|p| p.keys().copied())
        .collect();
    let epochs: Vec<_> = times.into_iter().collect();
    let positions = sample_seed_states(config, seed, &epochs)?;
    let positions: BTreeMap<_, _> = epochs.into_iter().zip(positions.iter().map(r_of)).collect();
    let mut selected = Vec::new();
    for pass in passes.into_values() {
        let pass: Vec<_> = pass.into_values().collect();
        if pass.len() <= k {
            selected.extend(pass);
        } else {
            let tca = closest(&pass, geometry, &positions)?;
            let times: Vec<_> = pass.iter().map(|c| c.pointing.time).collect();
            let keep = near_tca(&times, tca, k);
            selected.extend(
                pass.into_iter()
                    .enumerate()
                    .filter_map(|(i, c)| keep.contains(&i).then_some(c)),
            );
        }
    }
    Ok(selected)
}

fn closest(
    pass: &[Cone],
    geometry: &PreparedGeometry,
    positions: &BTreeMap<Instant, V3>,
) -> Result<usize> {
    let mut best = (f64::INFINITY, 0);
    for (i, c) in pass.iter().enumerate() {
        let slant =
            (positions[&c.pointing.time] - station_at(geometry, c.station_row)?.position()).norm();
        if !slant.is_finite() {
            return Err(KernelError::Numerical("seed slant range overflow"));
        }
        if slant < best.0 {
            best = (slant, i);
        }
    }
    Ok(best.1)
}

fn near_tca(times: &[Instant], center: usize, count: usize) -> BTreeSet<usize> {
    let last = times.len() - 1;
    let right = last - center;
    // Split the budget evenly. Give an odd extra slot to the earlier side,
    // and transfer any slots that a partially observed side cannot fill.
    let before = (count / 2)
        .max((count - 1).saturating_sub(right))
        .min(center);
    let mut chosen = BTreeSet::from([center]);
    select_side(times, center, 0, before, &mut chosen);
    select_side(times, center, last, count - 1 - before, &mut chosen);
    chosen
}

fn select_side(
    times: &[Instant],
    center: usize,
    edge: usize,
    count: usize,
    chosen: &mut BTreeSet<usize>,
) {
    let half_span = (times[edge] - times[center]).as_seconds() * 0.5;
    for j in 1..=count {
        let target = times[center] + Duration::from_seconds(half_span * j as f64 / count as f64);
        let candidates = center.min(edge)..=center.max(edge);
        if let Some(i) = candidates
            .filter(|i| !chosen.contains(i))
            .min_by_key(|&i| (times[i] - target).usec.unsigned_abs())
        {
            chosen.insert(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_sides_and_partial_passes() {
        let epoch = Instant::from_date(2024, 1, 1).unwrap();
        let times: Vec<_> = (0..21)
            .map(|i| epoch + Duration::from_seconds(i as f64))
            .collect();
        assert_eq!(near_tca(&times, 10, 3), BTreeSet::from([5, 10, 15]));
        assert_eq!(near_tca(&times, 0, 3), BTreeSet::from([0, 5, 10]));
        assert_eq!(near_tca(&times, 20, 3), BTreeSet::from([10, 15, 20]));
        assert_eq!(near_tca(&times, 7, 1), BTreeSet::from([7]));
        assert_eq!(near_tca(&times, 1, 10).len(), 10);
    }
}
