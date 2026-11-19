# Tychonic for Python

Tychonic 0.2.0 is the Python release candidate for the Rust 0.10.1 kernels.
The public API accepts NumPy columns, UTC epochs, and calibrated uncertainties.
The extension performs propagation, fitting, filtering, smoothing, and acquisition planning in Rust.

From a checkout with a Rust toolchain and CPython 3.11–3.14, build and install
with `python -m pip install .`. The optional extras `.[polars]` and `.[tdm]`
enable column export and CCSDS Doppler TDM generation. Set `SATKIT_DATA` to a
provisioned Satkit data directory for offline numerical work.

```python
from tychonic import CalibratedLM, EnsembleFilter, ForceModel, GaussianPrior, ODProblem

forces = ForceModel()
fit = ODProblem(
    historical_job,
    initial_guess=seed,
    force_model=forces,
    profile=CalibratedLM(),
).solve(output_epoch=publication_epoch)

orbit = fit.estimate.orbital_marginal()
catalogue = EnsembleFilter.ekf(force_model=forces, retain_history=True)
catalogue.add_spacecraft(
    historical_job.spacecraft_id,
    fit.estimate,
    bias_prior_hz=GaussianPrior(0, 5),
    last_contributing_epoch=fit.last_contributing_epoch,
)
updates = catalogue.update(next_contact)
checkpoint = catalogue.checkpoint(next_contact.spacecraft_id)
```

The [Python guide](https://github.com/Tycho887/tychonic/blob/main/docs/python.md)
describes input conventions, solver profiles, checkpoints, and acquisition.
The adapters recovered from the earlier 0.1.0 prototype are available under
`tychonic.low_level`.
