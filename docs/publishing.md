# Tychonic crates.io releases

Version **0.9.7** was published to crates.io on **2026-11-11** from clean commit
`ce3d7bcb6dab25bc0321f3397be881a719441346`. Its release tag is `v0.9.7`.
The published archive's SHA-256 is
`0f5b6a4c16ee6bfcf09cfb7bc2d50934e44392e1c0d17e8fd8f00facad4bbbdf`.

The 0.10.0 workspace contains `tychonic` and `tychonic-constrained`. Verify
both source packages together with `cargo package --workspace --locked`.
For publication, publish the core first, then the companion after the registry
can resolve the matching core version. The companion's native IPOPT comparison
runs separately from its Rust build. See the [qualification commands](constrained_range_rate.md).

The canonical source checkout is `/home/michaeljohansen/tychonic`. Prepare and
verify future releases in this repository. Before the next publication, update
the package version, both lockfiles, installation example, and changelog.

Tychonic distributes portable Rust source as `tychonic-0.9.7.crate`. The manifest
allowlist uses repository-root paths and includes library source, tests,
benchmark support, concise examples, documentation, the citation index, and both
licenses. Research tooling, generated results, local
reference PDFs, credentials, compiled outputs, and release archives are excluded.

From the repository root, with Satkit data provisioned:

```sh
cargo fmt --all -- --check
SATKIT_OFFLINE=1 cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release --lib
cargo doc --locked --no-deps --lib
cargo check --manifest-path tools/Cargo.toml --locked --all-targets --all-features
cargo package --locked --list
cargo package --locked --offline
```

Package a clean release commit. Verification builds the extracted source.
The archive is `target/package/tychonic-0.9.7.crate` and generated API
documentation is `target/doc/tychonic/index.html`. A compiled `.rlib` is tied
to its compiler, platform and dependencies; distribute the source archive.

Prepare the archive and release notes under ignored `dist/`, with SHA-256
checksums. Validate a downstream project against the extracted crate before
publication. Test and benchmark fixtures must depend only on packaged files.
Simulation reproduction instructions are in the repository's
[tools guide](https://github.com/Tycho887/tychonic/blob/main/tools/README.md).

Version 0.9.7 is already published and cannot be published again. For a future
version, review the release commit and tag it with that version. Authenticate
with Cargo's credential flow, then verify and publish to crates.io:

```sh
cargo publish --locked --dry-run
cargo publish --locked
```

Each version can be published only once.

## GitHub release publishing

The workflows publish automatically when a release tag is pushed:

| Tag | Destination | Required validation |
| --- | --- | --- |
| `python-v<PYTHON_VERSION>` | TestPyPI | Python 3.11–3.14 wheels, source archive, tests, lint, and archive checks |
| `v<RUST_VERSION>` | crates.io | Rust contracts, long release regressions, native comparisons, and workspace package verification |

Python tags must match `pyproject.toml`. Rust tags must match the core and
constrained crate versions, including the companion's exact core dependency.
The Rust job publishes `tychonic` first, then `tychonic-constrained`. Branch
pushes and pull requests run validation without publishing. Production PyPI
publication remains a manual dispatch of `python.yml` from a Python release
tag with `publish_to=pypi`; `publish_to=testpypi` also permits manual TestPyPI
publication.

Configure these publishers before pushing a new release tag:

- On TestPyPI, register GitHub owner `Tycho887`, repository `tychonic`, workflow
  `python.yml`, and environment `testpypi` for project `tychonic`. Use the same
  fields with environment `pypi` on production PyPI if needed.
- On crates.io, register GitHub owner `Tycho887`, repository `tychonic`, workflow
  `rust.yml`, and environment `cratesio` for both Rust crates. The workflow uses
  `rust-lang/crates-io-auth-action@v1` to obtain a temporary upload token. For a
  crate's first publication or when Trusted Publishing is unavailable, provide
  an API token authorized for both crates as the GitHub Actions secret
  `CARGO_REGISTRY_TOKEN`; this takes precedence over Trusted Publishing.
- Create the matching GitHub environments `testpypi`, `pypi` (if used), and
  `cratesio`. Any configured environment approval rules still apply.

The release tag must point to a commit containing the corrected workflow and
matching package versions. Rerunning a failed run uses the workflow from its
original commit. When preparing a new Python version, also update
`python/Cargo.toml` and `python/Cargo.lock`. An uploaded distribution cannot be
overwritten, so use an unused version for subsequent TestPyPI releases.

Consumers install from crates.io:

```toml
[dependencies]
tychonic = "0.9.7"
```
