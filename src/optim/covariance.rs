//! Absolute covariance from calibrated, whitened observation information.
use super::FitError;
use numeris::{DynMatrix, dynmatrix::DynSvd};
use std::fmt;

/// Statistical interpretation of a returned parameter covariance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovarianceMethod {
    /// Inverse whitened information, using absolute measurement uncertainties.
    /// No residual scale estimate or LM damping is included.
    AbsoluteWhitenedInformation,
}

impl fmt::Display for CovarianceMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbsoluteWhitenedInformation => f.write_str("absolute_whitened_information"),
        }
    }
}

/// Failure to identify finite parameter uncertainty.
#[derive(Debug, PartialEq)]
pub enum CovarianceError {
    /// Empty dimensions or non-finite Jacobian entries.
    InvalidInput(&'static str),
    /// Insufficient independent observation information. No matrix is returned.
    RankDeficient {
        /// Numerical rank of the column-normalized whitened Jacobian.
        rank: usize,
        /// Number of fitted parameters.
        parameters: usize,
    },
    /// Factorization failed or physical covariance exceeds floating-point range.
    Numerical(&'static str),
}

impl fmt::Display for CovarianceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid covariance input: {message}"),
            Self::RankDeficient { rank, parameters } => {
                write!(
                    f,
                    "covariance rank deficient: rank {rank} for {parameters} parameters"
                )
            }
            Self::Numerical(message) => write!(f, "covariance numerical failure: {message}"),
        }
    }
}
impl std::error::Error for CovarianceError {}

/// Separate error type preserves the existing `FitError` variants and fit APIs.
#[derive(Debug)]
pub enum CovarianceFitError {
    /// Propagation, observation evaluation, or LM convergence failure.
    Fit(FitError),
    /// The converged fit has no usable absolute parameter covariance.
    Covariance(CovarianceError),
}

impl fmt::Display for CovarianceFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fit(error) => error.fmt(f),
            Self::Covariance(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for CovarianceFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fit(error) => Some(error),
            Self::Covariance(error) => Some(error),
        }
    }
}
impl From<FitError> for CovarianceFitError {
    fn from(error: FitError) -> Self {
        Self::Fit(error)
    }
}
impl From<crate::kernels::KernelError> for CovarianceFitError {
    fn from(error: crate::kernels::KernelError) -> Self {
        Self::Fit(FitError::Kernel(error))
    }
}
impl From<CovarianceError> for CovarianceFitError {
    fn from(error: CovarianceError) -> Self {
        Self::Covariance(error)
    }
}

/// Physical parameter uncertainty and its statistical interpretation.
#[derive(Debug)]
pub struct ParameterCovariance {
    /// Symmetric covariance in the corresponding physical parameter units.
    pub matrix: DynMatrix<f64>,
    /// Numerical rank, equal to the number of columns for successful results.
    pub rank: usize,
    /// Supplied calibrated measurement covariance defines absolute uncertainty.
    pub method: CovarianceMethod,
}

/// Invert the information in a calibrated, whitened physical Jacobian.
///
/// For `J = L^-1 H`, where `R = L L^T` is absolute measurement covariance,
/// this returns `(H^T R^-1 H)^-1`. RSS, optimizer scales, and damping do not
/// enter the calculation. Equal observation/parameter counts are supported.
///
/// Normalize each nonzero column by its Euclidean norm, factor the normalized
/// matrix with column-pivoted QR, and apply numeris SVD only to the compact R.
/// This avoids the quadratic row storage in numeris 0.6.0's tall SVD.
/// Numerical rank counts singular values greater than
/// `f64::EPSILON * max(rows, columns) * largest_singular_value` in these
/// normalized coordinates. Undo the column permutation and normalization to
/// return physical covariance. Zero columns and deficient rank return errors;
/// no pseudo-inverse, regularization, or residual scaling is used.
///
/// Memory is O(rows * columns + columns²). The input must have nonempty
/// dimensions and finite values. Factorization and arithmetic errors are
/// returned without a partial covariance.
///
/// ```
/// use tychonic::{numeris::DynMatrix, optim::parameter_covariance};
/// let j = DynMatrix::from_rows(2, 2, &[2.0_f64, 0.0, 0.0, 4.0]);
/// let uncertainty = parameter_covariance(&j)?;
/// assert!((uncertainty.matrix[(0, 0)] - 0.25).abs() < 1e-14);
/// assert!((uncertainty.matrix[(1, 1)] - 0.0625).abs() < 1e-14);
/// # Ok::<(), tychonic::optim::CovarianceError>(())
/// ```
pub fn parameter_covariance(
    jacobian: &DynMatrix<f64>,
) -> Result<ParameterCovariance, CovarianceError> {
    validate(jacobian)?;
    let columns = jacobian.ncols();
    let (normalized, inverse_norms) = normalize_columns(jacobian)?;
    let qr = normalized.qr_col_pivot();
    let r = qr.r();
    require_finite(r.as_slice())?;
    let svd = r
        .svd()
        .map_err(|_| CovarianceError::Numerical("SVD failed"))?;
    let singular = svd.singular_values();
    require_finite(singular)?;
    let tolerance = f64::EPSILON * jacobian.nrows().max(columns) as f64 * singular[0];
    let rank = singular.iter().filter(|&&value| value > tolerance).count();
    if rank != columns {
        return Err(CovarianceError::RankDeficient {
            rank,
            parameters: columns,
        });
    }
    let matrix = physical_covariance(&svd, qr.permutation(), &inverse_norms)?;
    Ok(ParameterCovariance {
        matrix,
        rank,
        method: CovarianceMethod::AbsoluteWhitenedInformation,
    })
}

fn validate(j: &DynMatrix<f64>) -> Result<(), CovarianceError> {
    if j.nrows() == 0 || j.ncols() == 0 {
        return Err(CovarianceError::InvalidInput("empty Jacobian dimensions"));
    }
    if !j.as_slice().iter().all(|v| v.is_finite()) {
        return Err(CovarianceError::InvalidInput("Jacobian must be finite"));
    }
    Ok(())
}

fn normalize_columns(j: &DynMatrix<f64>) -> Result<(DynMatrix<f64>, Vec<f64>), CovarianceError> {
    // Zero padding permits QR to diagnose wide, underdetermined Jacobians.
    let mut normalized = DynMatrix::zeros(j.nrows().max(j.ncols()), j.ncols());
    let mut inverse_norms = Vec::with_capacity(j.ncols());
    for column in 0..j.ncols() {
        let max = (0..j.nrows())
            .map(|row| j[(row, column)].abs())
            .fold(0.0_f64, f64::max);
        if max == 0.0 {
            inverse_norms.push(1.0);
            continue;
        }
        let root = (0..j.nrows())
            .map(|row| (j[(row, column)] / max).powi(2))
            .sum::<f64>()
            .sqrt();
        let inverse = (1.0 / max) / root;
        require_finite(&[inverse])?;
        if inverse == 0.0 {
            return Err(CovarianceError::Numerical("column normalization underflow"));
        }
        inverse_norms.push(inverse);
        for row in 0..j.nrows() {
            normalized[(row, column)] = (j[(row, column)] / max) / root;
        }
    }
    Ok((normalized, inverse_norms))
}

fn physical_covariance(
    svd: &DynSvd<f64>,
    permutation: &[usize],
    inverse_norms: &[f64],
) -> Result<DynMatrix<f64>, CovarianceError> {
    let columns = permutation.len();
    // Rows of this factor are original physical parameter coordinates.
    let mut factor = DynMatrix::zeros(columns, columns);
    for (position, &original) in permutation.iter().enumerate() {
        for component in 0..columns {
            factor[(original, component)] = (svd.vt()[(component, position)]
                / svd.singular_values()[component])
                * inverse_norms[original];
        }
    }
    require_finite(factor.as_slice())?;
    let mut covariance = DynMatrix::zeros(columns, columns);
    for i in 0..columns {
        for j in 0..=i {
            let value = (0..columns)
                .map(|k| factor[(i, k)] * factor[(j, k)])
                .sum::<f64>();
            covariance[(i, j)] = value;
            covariance[(j, i)] = value;
        }
    }
    require_finite(covariance.as_slice())?;
    if (0..columns).any(|i| covariance[(i, i)] <= 0.0) {
        return Err(CovarianceError::Numerical(
            "physical covariance diagonal underflow",
        ));
    }
    Ok(covariance)
}

fn require_finite(values: &[f64]) -> Result<(), CovarianceError> {
    if !values.iter().all(|v| v.is_finite()) {
        return Err(CovarianceError::Numerical(
            "arithmetic exceeds numeric range",
        ));
    }
    Ok(())
}
