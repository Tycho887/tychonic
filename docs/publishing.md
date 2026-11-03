# Packaging v0.9.0

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
cargo package --locked --allow-dirty --list
cargo package --locked --offline --allow-dirty
```

`--allow-dirty` permits packaging a reviewed working tree before committing;
omit it for a release from a clean commit. Verification builds the extracted
source. The archive is `target/package/tychonic-0.9.0.crate` and local API
documentation is `target/doc/tychonic/index.html`. A compiled `.rlib` is tied
to its compiler, platform and dependencies; distribute the source archive.

Prepare the archive and release notes under ignored `dist/`, with SHA-256
checksums. Validate a downstream project against the extracted crate before
publication. Test and benchmark fixtures must depend only on packaged files.
Simulation reproduction instructions are in the repository's
[tools guide](https://github.com/Tycho887/tychonic/blob/main/tools/README.md).

After the release commit is reviewed, tag it `v0.9.0`. Registry publication is
a separate action. Authenticate locally with Cargo's credential flow, then run
`cargo publish --locked --dry-run` before `cargo publish --locked`. Version
0.9.0 can be published only once. A private registry uses Cargo's `--registry`
option and the corresponding configured credential provider.

Consumers may use `tychonic = "0.9.0"` after registry publication, a path to
the extracted source before publication, or a Git dependency on an existing tag:

```toml
[dependencies]
tychonic = { git = "https://github.com/Tycho887/tychonic", tag = "v0.9.0" }
```
