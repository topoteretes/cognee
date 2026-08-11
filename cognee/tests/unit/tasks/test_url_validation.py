import os
import pytest
from cognee.tasks.web_scraper.utils import validate_safe_url


def test_validate_safe_url_aws_metadata():
    """Verify that AWS instance metadata service IP is blocked by default."""
    with pytest.raises(ValueError, match="SSRF Protection"):
        validate_safe_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")


def test_validate_safe_url_loopback():
    """Verify that loopback IP addresses and localhost are blocked by default."""
    with pytest.raises(ValueError, match="SSRF Protection"):
        validate_safe_url("http://127.0.0.1:8000/admin")
    with pytest.raises(ValueError, match="SSRF Protection"):
        validate_safe_url("http://localhost:3000")


def test_validate_safe_url_private_networks():
    """Verify that RFC 1918 private IP address ranges are blocked."""
    with pytest.raises(ValueError, match="SSRF Protection"):
        validate_safe_url("http://10.0.0.1/secrets")
    with pytest.raises(ValueError, match="SSRF Protection"):
        validate_safe_url("http://172.16.0.100/config")
    with pytest.raises(ValueError, match="SSRF Protection"):
        validate_safe_url("http://192.168.1.1/")


def test_validate_safe_url_unsupported_scheme():
    """Verify that non-http/https schemes are rejected."""
    with pytest.raises(ValueError, match="SSRF Protection: Unsupported scheme"):
        validate_safe_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="SSRF Protection: Unsupported scheme"):
        validate_safe_url("ftp://example.com/file")


def test_validate_safe_url_allow_local_env_var(monkeypatch):
    """Verify that setting COGNEE_ALLOW_LOCAL_URLS allows local/private URLs."""
    monkeypatch.setenv("COGNEE_ALLOW_LOCAL_URLS", "true")
    # Should not raise any ValueError when allow_local is enabled via env var
    validate_safe_url("http://127.0.0.1:8000/admin")
    validate_safe_url("http://localhost:3000")


def test_validate_safe_url_allow_local_param():
    """Verify that explicit allow_local=True parameter bypasses SSRF check."""
    validate_safe_url("http://10.0.0.1/secrets", allow_local=True)
