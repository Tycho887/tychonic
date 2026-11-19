"""Check the release version and required contents of local distribution archives."""

import argparse
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


def contents(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        return (
            names,
            {"tychonic/api.py", "tychonic/py.typed"},
            any(n.startswith("tychonic/_native.") and n.endswith(".so") for n in names),
        )
    with tarfile.open(path) as archive:
        names = {"/".join(PurePosixPath(n).parts[1:]) for n in archive.getnames()}
    required = {
        "Cargo.toml",
        "src/lib.rs",
        "constrained/src/lib.rs",
        "python/Cargo.lock",
        "python/Cargo.toml",
        "python/src/lib.rs",
        "python/tychonic/api.py",
        "python/tests/data/two_contacts.json",
        "pyproject.toml",
    }
    return names, required, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    version = tomllib.loads(project.read_text())["project"]["version"]
    if args.tag is not None and args.tag != f"python-v{version}":
        parser.error("release tag must match the Python project version")
    paths = [*args.directory.glob("*.whl"), *args.directory.glob("*.tar.gz")]
    if not paths:
        parser.error("no release archives found")
    for path in paths:
        if not path.name.startswith(
            (f"tychonic-{version}-", f"tychonic-{version}.tar.gz")
        ):
            parser.error(f"unexpected release archive {path.name}")
        names, required, extension = contents(path)
        if not required <= names or not extension:
            parser.error(f"incomplete {path.name}: {required - names}")
        if any(
            set(PurePosixPath(n).parts) & {".aws", ".codex", "target", ".venv"}
            for n in names
        ):
            parser.error(f"unexpected local files in {path.name}")
        print(path.name)


if __name__ == "__main__":
    main()
