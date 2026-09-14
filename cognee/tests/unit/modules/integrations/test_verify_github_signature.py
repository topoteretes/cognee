"""Unit tests for cognee.modules.integrations.github.verify_github_signature.

Pure function — no DB, no network. The invariant that matters: only an
HMAC-SHA256 keyed with the webhook secret over the exact raw bytes, carried
in GitHub's ``sha256=`` header format, passes.
"""

import hashlib
import hmac

import pytest

from cognee.modules.integrations.github.verify_github_signature import is_valid_github_signature

_SECRET = "It's a Secret to Everybody"


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.integrations.github.github_settings.github_settings.webhook_secret",
        _SECRET,
    )


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()


def test_valid_signature_passes():
    body = b'{"action":"created","installation":{"id":42}}'
    assert is_valid_github_signature(body, _sign(body))


def test_tampered_body_fails():
    signature = _sign(b'{"action":"created"}')
    assert not is_valid_github_signature(b'{"action":"deleted"}', signature)


def test_wrong_secret_fails():
    body = b"payload"
    wrong = "sha256=" + hmac.new(b"other-secret", body, hashlib.sha256).hexdigest()
    assert not is_valid_github_signature(body, wrong)


def test_missing_prefix_fails():
    body = b"payload"
    bare_hex = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert not is_valid_github_signature(body, bare_hex)
    # sha1= is GitHub's legacy header, deliberately unsupported.
    assert not is_valid_github_signature(body, "sha1=" + bare_hex)


def test_missing_signature_fails():
    assert not is_valid_github_signature(b"payload", "")
    assert not is_valid_github_signature(b"payload", None)
