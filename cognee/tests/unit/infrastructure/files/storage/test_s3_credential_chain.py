"""Unit tests for S3FileStorage credential handling.

Covers the IAM-role fallback (issue #3086): when no static AWS credentials are
configured, S3FileStorage must construct s3fs.S3FileSystem without key/secret so
boto3's default credential chain (ECS task role, EC2 instance metadata,
environment variables, ~/.aws/credentials) resolves them, instead of raising
ValueError. This is what makes S3 storage usable on ECS Fargate / EC2 / Lambda.

No real AWS credentials or network access are required: s3fs.S3FileSystem is
replaced with a recording stand-in.
"""

import pytest
import s3fs

import cognee.infrastructure.files.storage.S3FileStorage as s3_storage_module
from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage
from cognee.infrastructure.files.storage.s3_config import S3Config


class _RecordingS3FileSystem:
    """Stand-in for s3fs.S3FileSystem that records how it was constructed."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingS3FileSystem.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_recorder():
    _RecordingS3FileSystem.instances.clear()
    yield
    _RecordingS3FileSystem.instances.clear()


def _clear_aws_credential_env(monkeypatch):
    """Make the no-credentials case deterministic on hosts that have AWS env set."""
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_falls_back_to_boto3_credential_chain_when_no_static_credentials(monkeypatch):
    """With no key/secret configured, S3FileStorage must not raise and must let
    boto3 resolve credentials by omitting key/secret from s3fs.S3FileSystem."""
    _clear_aws_credential_env(monkeypatch)
    monkeypatch.setattr(
        s3_storage_module,
        "get_s3_config",
        lambda: S3Config(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
        ),
    )
    monkeypatch.setattr(s3fs, "S3FileSystem", _RecordingS3FileSystem)

    # Must not raise: IAM-role deployments (ECS/EC2/Lambda) depend on this path.
    storage = S3FileStorage("test-bucket")

    assert len(_RecordingS3FileSystem.instances) == 1
    used_kwargs = _RecordingS3FileSystem.instances[0].kwargs

    # Omitting key/secret is what lets boto3's credential chain run.
    assert "key" not in used_kwargs
    assert "secret" not in used_kwargs
    # anon must stay False: we want real credentials, not anonymous access.
    assert used_kwargs["anon"] is False
    # Connection settings from the explicit-credentials path are preserved.
    assert "endpoint_url" in used_kwargs
    assert "client_kwargs" in used_kwargs
    assert storage.storage_path == "test-bucket"


def test_uses_explicit_credentials_when_configured(monkeypatch):
    """The explicit-credentials path keeps passing key/secret/token through."""
    monkeypatch.setattr(
        s3_storage_module,
        "get_s3_config",
        lambda: S3Config(
            aws_access_key_id="KEYID",
            aws_secret_access_key="SECRET",
            aws_session_token="TOKEN",
            aws_region="us-east-1",
            aws_endpoint_url="https://example",
        ),
    )
    monkeypatch.setattr(s3fs, "S3FileSystem", _RecordingS3FileSystem)

    S3FileStorage("test-bucket")

    assert len(_RecordingS3FileSystem.instances) == 1
    used_kwargs = _RecordingS3FileSystem.instances[0].kwargs
    assert used_kwargs["key"] == "KEYID"
    assert used_kwargs["secret"] == "SECRET"
    assert used_kwargs["token"] == "TOKEN"
    assert used_kwargs["anon"] is False
