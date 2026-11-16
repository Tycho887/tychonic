# Tychonic crates.io releases

Version **0.9.7** was published to crates.io on **2026-11-11** from clean commit
`ce3d7bcb6dab25bc0321f3397be881a719441346`. Its release tag is `v0.9.7`.
The published archive's SHA-256 is
`0f5b6a4c16ee6bfcf09cfb7bc2d50934e44392e1c0d17e8fd8f00facad4bbbdf`.

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

Consumers install from crates.io:

```toml
[dependencies]
tychonic = "0.9.7"
```
