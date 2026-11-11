#!/usr/bin/env python3
"""Create a read-only, checksummed archive of the DIMVA evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/State-Aware-ICS-Security"
DIRECTORIES = (
    "src",
    "baselines",
    "tests",
    "vendor",
    "data",
    "review",
    "docs/papers",
    "scripts/out",
    "paper/State-Aware-ICS-Security",
)
ROOT_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "TODO.md",
    "INSTALL.md",
)
EXCLUDED_PARTS = {".git", ".worktrees", "__pycache__", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for directory in DIRECTORIES:
        root = ROOT / directory
        if not root.is_dir():
            raise FileNotFoundError(f"required snapshot directory is missing: {root}")
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink() and not (
                EXCLUDED_PARTS & set(path.parts)
            ):
                files.add(path)
    files.update(ROOT.joinpath("scripts").glob("*.py"))
    for name in ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def create_snapshot(output: Path) -> dict:
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("snapshot output must be outside the repository")
    if output.suffix != ".tar":
        raise ValueError("snapshot output must end in .tar")
    if output.exists() or output.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = selected_files()
    records = []
    for index, path in enumerate(files, start=1):
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
        if index % 1000 == 0:
            print(f"hashed {index}/{len(files)} files", flush=True)

    manifest = {
        "schema": "stateaware-reproducibility-snapshot/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(ROOT),
        "paper_commit": git_commit(PAPER),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for index, path in enumerate(files, start=1):
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
            if index % 1000 == 0:
                print(f"archived {index}/{len(files)} files", flush=True)
        info = tarfile.TarInfo("REPRODUCIBILITY_MANIFEST.json")
        info.size = len(encoded)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        info.mode = 0o444
        archive.addfile(info, io.BytesIO(encoded))

    digest = sha256_file(output)
    checksum = output.with_suffix(".sha256")
    checksum.write_text(f"{digest}  {output.name}\n")
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_bytes(encoded)
    with tarfile.open(output, "r") as archive:
        archived = set(archive.getnames())
    expected = {record["path"] for record in records} | {"REPRODUCIBILITY_MANIFEST.json"}
    if archived != expected:
        raise RuntimeError("archive member verification failed")
    for path in (output, checksum, manifest_path):
        os.chmod(path, 0o444)
    print(f"snapshot: {output}")
    print(f"sha256: {digest}")
    print(f"files: {len(records)}; bytes: {manifest['total_bytes']}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="external .tar path")
    args = parser.parse_args()
    create_snapshot(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
