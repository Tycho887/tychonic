# Release v0.9.0 to crates.io

Tychonic distributes portable Rust source as `tychonic-0.9.0.crate`. The manifest
allowlist includes library source, tests, benchmark support, concise examples,
documentation, and both licenses. Research tooling, generated results, local
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
The archive is `target/package/tychonic-0.9.0.crate` and generated API
documentation is `target/doc/tychonic/index.html`. A compiled `.rlib` is tied
to its compiler, platform and dependencies; distribute the source archive.

Prepare the archive and release notes under ignored `dist/`, with SHA-256
checksums. Validate a downstream project against the extracted crate before
publication. Test and benchmark fixtures must depend only on packaged files.
Simulation reproduction instructions are in the repository's
[tools guide](https://github.com/Tycho887/tychonic/blob/main/tools/README.md).

After the release commit is reviewed, tag it `v0.9.0`. Authenticate with Cargo's
credential flow, then verify and publish to crates.io:

```sh
cargo publish --locked --dry-run
cargo publish --locked
```

Version 0.9.0 can be published only once.

Consumers install from crates.io:

```toml
[dependencies]
tychonic = "0.9.0"
```
