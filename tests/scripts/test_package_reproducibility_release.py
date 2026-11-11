import hashlib
from pathlib import Path

from scripts.package_reproducibility_release import package


def test_release_chunks_roundtrip_and_hashes(tmp_path):
    archive = tmp_path / "evidence.tar"
    archive.write_bytes(b"abcdefghij")
    output = tmp_path / "release"
    manifest = package(archive, output, chunk_size=3)
    assert [record["path"] for record in manifest["chunks"]] == [
        "evidence.tar.aa", "evidence.tar.ab", "evidence.tar.ac", "evidence.tar.ad"
    ]
    joined = b"".join((output / record["path"]).read_bytes() for record in manifest["chunks"])
    assert joined == archive.read_bytes()
    assert manifest["archive_sha256"] == hashlib.sha256(joined).hexdigest()
    checksums = (output / "SHA256SUMS").read_text()
    assert f"{manifest['archive_sha256']}  evidence.tar" in checksums
