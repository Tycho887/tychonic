# Research and reproduction tools

Run these commands from the **repository root**. This is a separate unpublished
crate; tooling dependencies and executables are excluded from the library crate.
Its lockfile is retained for reproducibility. Satkit data must be provisioned
before offline propagation. Plotting requires Python, NumPy, and Matplotlib.

```sh
cargo check --manifest-path tools/Cargo.toml --locked --all-targets --all-features
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin phase_pass -- --trials 100
python3 tools/scripts/plot_phase_pass.py artifacts/phase_pass
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin differential_pass -- --trials 100
python3 tools/scripts/plot_differential_pass.py artifacts/differential_pass
```

Generated runs remain under ignored `artifacts/`; reports and plots stay beside
their run data. Do not add generated PNG/SVG/PDF files elsewhere. Selected
documentation images belong under `docs/assets/` and small regression fixtures
under `tests/fixtures/`. Existing local runs have been preserved.

The [phase guide](../docs/phase_difference.md) and
[differential guide](../docs/differential_doppler.md) describe experiment options.
[Profiling](profiling/README.md) retains the timing and allocation harnesses.
Shared benchmark inputs live in `benches/support/mod.rs` at the repository root.

Migration from earlier commands:

| Earlier command/path | v0.9.0 |
|---|---|
| `cargo run` | `cargo run --locked --example batch_fit` |
| `cargo run --bin phase_pass` | Add `--manifest-path tools/Cargo.toml` |
| `cargo run --bin differential_pass` | Add `--manifest-path tools/Cargo.toml` |
| `cargo build --bin kernel_profile` | Add `--manifest-path tools/Cargo.toml` |
| `scripts/plot_*.py` | `tools/scripts/plot_*.py` |
| `profiling/*.py` | `tools/profiling/*.py` |
| `target/release/kernel_profile` | `tools/target/release/kernel_profile` |

`CARGO_TARGET_DIR` can override build output locations. The profiling baseline
script obtains the tool target directory through Cargo metadata.
