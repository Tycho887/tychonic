# Absolute calibrated batch covariance

Version 0.9.4 adds covariance-enabled fitting without changing existing fits.
The caller supplies calibrated extracted observables, their absolute covariance,
an initial physical orbit guess, and existing propagation and LM settings.

## Interfaces

`measurements::fit_with_covariance(config, batches, guess, scales, settings)`
builds a prepared problem and delegates to
`optim::fit_lm_with_covariance(problem, guess, scales, settings)`.

`CovarianceFit` returns physical parameters, LM iterations, RSS, covariance,
numerical rank, and `CovarianceMethod::AbsoluteWhitenedInformation`.
`CovarianceFitError` distinguishes existing fitting errors from covariance errors.
`CovarianceError::RankDeficient { rank, parameters }` returns no matrix.
Existing `FitError` variants, fit functions, and Varpro covariance are unchanged.

`optim::parameter_covariance(jacobian)` supports consumers of an already
calibrated, whitened physical Jacobian. It returns `ParameterCovariance`.

## Statistical model and units

For physical derivatives H and absolute observation covariance R = L Lᵀ,
prepared kernels provide J = L⁻¹ H. The covariance is (Jᵀ J)⁻¹ at the converged
fit. It does not include LM damping, RSS scaling, or reduced chi-square.
Equal observation and parameter counts are valid when J has full rank.
A noiseless exact fit still has positive uncertainty when R is positive.

The state ordering is GCRF position metres, then velocity metres per second.
A seventh parameter, when configured, is C_d A/m. Covariance entries use the
products of the corresponding physical units. Joint observation order is
Doppler Hz, continuous phase rad, differential Doppler Hz, azimuth rad,
elevation rad, omitting absent components. Correlations are retained.

## Numerical method

Normalize each nonzero physical Jacobian column by its Euclidean norm. Apply
numeris column-pivoted QR to the normalized Jacobian, then numeris SVD to its
small R factor. This avoids numeris 0.6.0's internal full row-sized U allocation
when directly taking a tall SVD. Temporary storage is O(rows × parameters),
plus O(parameters²). Only the final retained LM evaluation is used; covariance
requires no extra propagation pass.

Count normalized singular values strictly above
EPSILON × max(rows, parameters) × largest normalized singular value.
Undo the column permutation and normalization to recover physical covariance.
Optimizer coordinate scales do not enter this calculation. This is a local
linearized covariance, not a guarantee of global nonlinear observability.

Empty/non-finite inputs, insufficient rank, failed factorization, and physical
covariance overflow or underflow produce errors. No pseudo-inverse,
regularization, singular-value truncation, or manufactured uncertainty is
returned. Existing contact-bias Varpro retains residual-scaled Schur covariance.

## Validation and downstream fixture

`cargo test --locked --test covariance` checks analytic linear and correlated
references, unit invariance, optimizer-scale invariance, six/seven parameters,
rank errors, large-batch memory behavior, and numerical-range failures.

To export a deterministic generation-2 DART normalized request:

```sh
DART_FIXTURE_OUTPUT=/tmp/dart-mixed-lm.json SATKIT_OFFLINE=1 \
  cargo test --locked --test covariance generation_two_fixture_has_later_held_out_observations
```

The fixture has 42 noisy five-component observations from two contacts.
Fit the first 39; retain the final three later epochs for independent EKF
updates. All components use explicit correlated absolute measurement covariance.
The fixture does not query providers, download data, or control an antenna.
