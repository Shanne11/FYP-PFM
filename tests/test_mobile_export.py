import hashlib

from deployment.export_mobile_package import sha256


def test_artifact_hash_is_sha256(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"mobile model")
    assert sha256(artifact) == hashlib.sha256(b"mobile model").hexdigest()
