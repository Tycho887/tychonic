use super::{AcquisitionError, EnuDirection, PointingGeometry, Result, positive};
use crate::kernels::{StationGeometry, V3};
use std::f64::consts::{FRAC_PI_2, PI};

/// Ideal great-circle, rest-to-rest actuator limits in radians/s and radians/s².
#[derive(Clone, Copy, Debug)]
pub struct AntennaLimits {
    pub rate_rad_s: f64,
    pub acceleration_rad_s2: f64,
}
impl Default for AntennaLimits {
    fn default() -> Self {
        Self {
            rate_rad_s: 1.10_f64.to_radians(),
            acceleration_rad_s2: 1.0_f64.to_radians(),
        }
    }
}
impl AntennaLimits {
    pub fn validate(&self) -> Result<()> {
        positive(self.rate_rad_s, "slew rate")?;
        positive(self.acceleration_rad_s2, "slew acceleration")
    }
    pub fn slew_time_s(&self, start: EnuDirection, target: EnuDirection) -> Result<f64> {
        self.validate()?;
        let angle = start.separation(target);
        let time = if angle <= self.rate_rad_s * (self.rate_rad_s / self.acceleration_rad_s2) {
            2.0 * (angle / self.acceleration_rad_s2).sqrt()
        } else {
            angle / self.rate_rad_s + self.rate_rad_s / self.acceleration_rad_s2
        };
        if !time.is_finite() {
            return Err(AcquisitionError::InvalidInput(
                "unrepresentable slew duration",
            ));
        }
        Ok(time)
    }
    /// Whether the slew fits into the move portion of a cycle.
    pub fn reachable(
        &self,
        start: EnuDirection,
        target: EnuDirection,
        move_s: f64,
    ) -> Result<bool> {
        if !move_s.is_finite() || move_s < 0.0 {
            return Err(AcquisitionError::InvalidInput("move duration"));
        }
        Ok(self.slew_time_s(start, target)? <= move_s + 1e-12)
    }
    pub fn partial_slew(
        &self,
        start: EnuDirection,
        target: EnuDirection,
        move_s: f64,
    ) -> Result<EnuDirection> {
        self.validate()?;
        if !move_s.is_finite() || move_s < 0.0 {
            return Err(AcquisitionError::InvalidInput("move duration"));
        }
        let angle = if move_s <= 2.0 * (self.rate_rad_s / self.acceleration_rad_s2) {
            self.acceleration_rad_s2 * (move_s / 2.0).powi(2)
        } else {
            self.rate_rad_s * (move_s - self.rate_rad_s / self.acceleration_rad_s2)
        };
        let distance = start.separation(target);
        if angle >= distance {
            return Ok(target);
        }
        if angle == 0.0 {
            return Ok(start);
        }
        let tangent = target.vector() - start.vector() * start.vector().dot(target.vector());
        let tangent = if tangent.norm() < 1e-14 {
            perpendicular(start)
        } else {
            tangent / tangent.norm()
        };
        EnuDirection::new(start.vector() * angle.cos() + tangent * angle.sin())
    }
}

pub(crate) fn check_mask(mask: f64) -> Result<()> {
    if !mask.is_finite() || !(0.0..=FRAC_PI_2).contains(&mask) {
        return Err(AcquisitionError::InvalidInput(
            "elevation mask must lie in [0, π/2]",
        ));
    }
    Ok(())
}

/// Topocentric acquisition geometry from a GCRF spacecraft position and an
/// existing prepared station row. Zenith has a valid direction and elevation.
pub fn pointing_geometry(
    position_gcrf: &V3,
    station: &StationGeometry,
    elevation_mask_rad: f64,
) -> Result<PointingGeometry> {
    check_mask(elevation_mask_rad)?;
    let local = station
        .rotation()
        .to_enu(&(position_gcrf - station.position()));
    let range_m = local.norm();
    if range_m == 0.0 {
        return Err(crate::kernels::KernelError::ZeroRange.into());
    }
    positive(range_m, "line-of-sight range")?;
    let direction_enu = EnuDirection::new(local)?;
    let elevation_rad = direction_enu.elevation_rad();
    Ok(PointingGeometry {
        direction_enu,
        range_m,
        elevation_rad,
        visible: elevation_rad >= elevation_mask_rad,
    })
}

fn perpendicular(center: EnuDirection) -> V3 {
    let index = (0..3)
        .min_by(|&i, &j| {
            center.vector()[i]
                .abs()
                .total_cmp(&center.vector()[j].abs())
        })
        .unwrap_or(0);
    let mut axis = V3::zeros();
    axis[index] = 1.0;
    let tangent = axis - center.vector() * axis.dot(center.vector());
    tangent / tangent.norm()
}

fn spherical_offset(center: EnuDirection, point: EnuDirection) -> Result<V3> {
    let angle = center.separation(point);
    if PI - angle < 1e-12 {
        return Err(AcquisitionError::InvalidInput("antipodal prior direction"));
    }
    if angle < 1e-14 {
        return Ok(V3::zeros());
    }
    let tangent = point.vector() - center.vector() * center.vector().dot(point.vector());
    Ok(tangent * (angle / tangent.norm()))
}

// Principal axis of the prior offsets, solved in a deterministic 2D tangent
// basis. Avoids iterative eigensolvers and also handles coincident particles.
fn dominant_tangent(center: EnuDirection, offsets: &[V3]) -> V3 {
    let first = perpendicular(center);
    let second = center.vector().cross(&first);
    let mut covariance = [0.0; 3];
    for offset in offsets {
        let x = first.dot(offset);
        let y = second.dot(offset);
        covariance[0] += x * x;
        covariance[1] += x * y;
        covariance[2] += y * y;
    }
    let angle = 0.5 * (2.0 * covariance[1]).atan2(covariance[0] - covariance[2]);
    first * angle.cos() + second * angle.sin()
}

/// Ordered moving candidates on the nominal tangent plane. `phase_difference`
/// is the ENU direction difference from nominal ±1-second mean-anomaly shifts,
/// evaluated at the same observation epoch. Spread is the population standard
/// deviation of the initial *unweighted* prior, independent of current belief.
pub fn along_track_candidates(
    center: EnuDirection,
    prior: &[EnuDirection],
    phase_difference: V3,
    count: usize,
    extent_sigma: f64,
) -> Result<Vec<EnuDirection>> {
    check_candidate_inputs(prior, &phase_difference, count, extent_sigma)?;
    let offsets = prior
        .iter()
        .map(|&p| spherical_offset(center, p))
        .collect::<Result<Vec<_>>>()?;
    let tangent = phase_difference - center.vector() * center.vector().dot(&phase_difference);
    let tangent = if tangent.norm() < 1e-12 {
        dominant_tangent(center, &offsets)
    } else {
        tangent / tangent.norm()
    };
    let projections: Vec<_> = offsets.iter().map(|x| x.dot(&tangent)).collect();
    let mean = projections.iter().sum::<f64>() / projections.len() as f64;
    let variance =
        projections.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / projections.len() as f64;
    let extent = extent_sigma * variance.sqrt();
    if !extent.is_finite() || extent >= PI {
        return Err(AcquisitionError::InvalidInput(
            "candidate extent must be less than π",
        ));
    }
    if count == 1 || extent < 1e-14 {
        return Ok(vec![center; count]);
    }
    (0..count)
        .map(|i| {
            if i == count / 2 {
                return Ok(center);
            }
            let angle = extent * (2.0 * i as f64 / (count - 1) as f64 - 1.0);
            EnuDirection::new(center.vector() * angle.cos() + tangent * angle.sin())
        })
        .collect()
}

fn check_candidate_inputs(
    prior: &[EnuDirection],
    phase_difference: &V3,
    count: usize,
    extent_sigma: f64,
) -> Result<()> {
    if prior.is_empty()
        || count == 0
        || count.is_multiple_of(2)
        || !extent_sigma.is_finite()
        || extent_sigma < 0.0
        || !phase_difference.iter().all(|x| x.is_finite())
    {
        return Err(AcquisitionError::InvalidInput(
            "candidate construction inputs",
        ));
    }
    Ok(())
}
