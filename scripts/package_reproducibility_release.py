#!/usr/bin/env python3
"""Split a reproducibility archive into GitHub Release-sized, hashed chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CHUNK_SIZE = 90 * 1024 * 1024
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def chunk_name(index: int) -> str:
    if index >= len(ALPHABET) ** 2:
        raise ValueError("too many archive chunks")
    return ALPHABET[index // len(ALPHABET)] + ALPHABET[index % len(ALPHABET)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(archive: Path, output_dir: Path, chunk_size: int = CHUNK_SIZE) -> dict:
    archive = archive.resolve()
    output_dir = output_dir.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if chunk_size < 1 or chunk_size > 2 * 1024 * 1024 * 1024:
        raise ValueError("chunk size must be between 1 byte and 2 GiB")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"release directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_digest = hashlib.sha256()
    chunks = []
    with archive.open("rb") as source:
        for index in range(len(ALPHABET) ** 2):
            name = f"{archive.name}.{chunk_name(index)}"
            path = output_dir / name
            digest = hashlib.sha256()
            size = 0
            with path.open("wb") as target:
                while True:
                    block = source.read(min(8 << 20, chunk_size - size))
                    if not block:
                        break
                    archive_digest.update(block)
                    digest.update(block)
                    target.write(block)
                    size += len(block)
                    if size == chunk_size:
                        break
            if size == 0:
                path.unlink()
                break
            chunks.append({"path": name, "bytes": size, "sha256": digest.hexdigest()})
            if size < chunk_size:
                break

    verify = hashlib.sha256()
    for record in chunks:
        path = output_dir / record["path"]
        verify.update(path.read_bytes())
    if verify.hexdigest() != archive_digest.hexdigest():
        raise RuntimeError("chunk concatenation hash mismatch")

    manifest = {
        "schema": "github-release-chunks/v1",
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_digest.hexdigest(),
        "chunk_bytes": chunk_size,
        "chunks": chunks,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_lines = [f"{archive_digest.hexdigest()}  {archive.name}"]
    checksum_lines.extend(f"{record['sha256']}  {record['path']}" for record in chunks)
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text("\n".join(checksum_lines) + "\n")
    for path in [output_dir / record["path"] for record in chunks] + [manifest_path, checksum_path]:
        os.chmod(path, 0o444)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    manifest = package(args.archive, args.output_dir, args.chunk_size)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
