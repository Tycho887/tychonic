#![allow(clippy::disallowed_methods)]
//! Explicit copies of mutable regression inputs, separate from validated APIs.
#![allow(dead_code)]
use tychonic::satkit::Instant;
use tychonic::{
    kernels::*,
    numeris::{DynMatrix, DynVector},
};

pub fn copy_config(value: &PropagationConfig) -> PropagationConfig {
    PropagationConfig::new(
        value.reference_epoch(),
        value.settings().clone(),
        value.mode(),
    )
    .unwrap()
}
pub fn copy_evaluation(value: &Evaluation) -> Evaluation {
    Evaluation {
        residual: DynVector::from_slice(value.residual.as_slice()),
        jacobian: DynMatrix::from_slice(
            value.jacobian.nrows(),
            value.jacobian.ncols(),
            value.jacobian.as_slice(),
        ),
    }
}
pub fn copy_station(value: &GCRFTrajectory) -> GCRFTrajectory {
    GCRFTrajectory {
        timestamps: value.timestamps.to_vec(),
        ground_stations: value.ground_stations.to_vec(),
        position: value.position.to_vec(),
        velocity: value.velocity.to_vec(),
    }
}
pub fn copy_enu(value: &EnuBasisTrajectory) -> EnuBasisTrajectory {
    EnuBasisTrajectory {
        timestamps: value.timestamps.to_vec(),
        enu_to_gcrf: value.enu_to_gcrf.to_vec(),
    }
}
#[derive(Clone)]
pub struct SampleInput {
    pub timestamps: Vec<Instant>,
    pub states: Vec<V6>,
    pub stm: Vec<M6>,
    pub state_dcd: Vec<V6>,
}
impl SampleInput {
    pub fn copy_from(value: &Sensitivities) -> Self {
        Self {
            timestamps: value.timestamps().to_vec(),
            states: value.states().to_vec(),
            stm: value.stm().to_vec(),
            state_dcd: value.state_dcd().to_vec(),
        }
    }
    pub fn prepare(&self) -> Result<Sensitivities> {
        Sensitivities::new(
            self.timestamps.to_vec(),
            self.states.to_vec(),
            self.stm.to_vec(),
            self.state_dcd.to_vec(),
        )
    }
}

pub fn copy_data_owned(
    value: &tychonic::optim::PassiveRfData<'_>,
) -> tychonic::optim::PassiveRfData<'static> {
    tychonic::optim::PassiveRfData::new(
        PreparedGeometry::new(value.geometry().timestamps(), value.geometry().stations()).unwrap(),
        value.interferometer(),
        value.baselines_enu_m().to_vec(),
        value.doppler_hz().to_vec(),
        value.phase_rad().map(<[f64]>::to_vec),
        value.sigma_doppler_hz(),
        value.sigma_phase_rad(),
    )
    .unwrap()
}
pub fn copy_prior(value: &tychonic::optim::RfPrior) -> tychonic::optim::RfPrior {
    tychonic::optim::RfPrior::new(
        DynVector::from_slice(value.mean().as_slice()),
        DynVector::from_slice(value.sigma().as_slice()),
    )
    .unwrap()
}
pub fn copy_extensions(value: &tychonic::optim::RfExtensions) -> tychonic::optim::RfExtensions {
    tychonic::optim::RfExtensions {
        baseline_sigma_m: value.baseline_sigma_m,
        differential_doppler: value.differential_doppler.as_ref().map(|d| {
            tychonic::optim::DifferentialDopplerData {
                difference_hz: d.difference_hz.to_vec(),
                sigma_antenna2_hz: d.sigma_antenna2_hz,
            }
        }),
        differential_interferometer: value.differential_interferometer,
        differential_baselines_enu_m: value
            .differential_baselines_enu_m
            .as_ref()
            .map(|b| b.to_vec()),
        differential_baseline_sigma_m: value.differential_baseline_sigma_m,
    }
}

pub fn copy_observations(value: &Observations) -> Observations {
    Observations {
        range_rate: value.range_rate.to_vec(),
        azimuth: value.azimuth.to_vec(),
        elevation: value.elevation.to_vec(),
        sigma_rr: value.sigma_rr,
        sigma_az: value.sigma_az,
        sigma_el: value.sigma_el,
    }
}
