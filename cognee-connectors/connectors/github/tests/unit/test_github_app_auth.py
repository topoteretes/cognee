"""Unit tests for cognee_connector_github.app_auth.build_app_jwt.

A real RSA keypair is generated per run and the token verified with the
public half — proving the hand-rolled JWT is a signature GitHub would
accept, not just three base64 blobs.
"""

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from cognee_connector_github.app_auth import build_app_jwt

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PEM = _PRIVATE_KEY.private_bytes(
    Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
).decode()


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    settings = "cognee_connector_github.settings.github_settings"
    monkeypatch.setattr(f"{settings}.app_id", "314159")
    # Stored with literal \n escapes, the way a single-line env file carries it.
    monkeypatch.setattr(f"{settings}.app_private_key", _PEM.replace("\n", "\\n"))


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def test_jwt_claims_and_signature():
    token = build_app_jwt(now=1_700_000_000)
    header_b64, payload_b64, signature_b64 = token.split(".")

    assert json.loads(_b64url_decode(header_b64)) == {"alg": "RS256", "typ": "JWT"}

    payload = json.loads(_b64url_decode(payload_b64))
    assert payload["iss"] == "314159"
    assert payload["iat"] == 1_700_000_000 - 60
    assert payload["exp"] == 1_700_000_000 + 9 * 60

    # raises InvalidSignature if the RS256 signature is wrong
    _PRIVATE_KEY.public_key().verify(
        _b64url_decode(signature_b64),
        f"{header_b64}.{payload_b64}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_missing_private_key_fails_loudly(monkeypatch):
    monkeypatch.setattr("cognee_connector_github.settings.github_settings.app_private_key", "")
    with pytest.raises(RuntimeError, match="GITHUB_APP_PRIVATE_KEY"):
        build_app_jwt()


def test_non_rsa_key_is_rejected(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    ed_pem = (
        ed25519.Ed25519PrivateKey.generate()
        .private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        .decode()
    )
    monkeypatch.setattr(
        "cognee_connector_github.settings.github_settings.app_private_key",
        ed_pem,
    )
    with pytest.raises(RuntimeError, match="RSA"):
        build_app_jwt()


class _FakeResponse:
    status = 201

    async def json(self):
        return {"token": "installation-token", "expires_at": "2026-10-28T10:00:00Z"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    def __init__(self, calls):
        self._calls = calls

    def post(self, url, **kwargs):
        self._calls.append((url, kwargs))
        return _FakeResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.mark.asyncio
async def test_a_connector_with_its_own_settings_authenticates_as_its_own_app(monkeypatch):
    """The app id and key come from the connector's settings, not the process-wide
    ones: empty globals must not fail the mint, and other globals must not be the
    app it signs as."""
    from cognee_connector_github import app_auth
    from cognee_connector_github.connector import GithubConnector
    from cognee_connector_github.settings import GithubSettings
    from cognee_connectors_core.testing import FakeCredential

    monkeypatch.setattr("cognee_connector_github.settings.github_settings.app_id", "")
    monkeypatch.setattr("cognee_connector_github.settings.github_settings.app_private_key", "")
    calls = []
    monkeypatch.setattr(app_auth.aiohttp, "ClientSession", lambda **_kwargs: _FakeSession(calls))
    monkeypatch.setattr(app_auth, "list_installation_repository_records", _async_return([]))
    connector = GithubConnector(settings=GithubSettings(app_id="271828", app_private_key=_PEM))

    await connector.list_resources(FakeCredential(provider="github", provider_account_id="42"), {})

    url, kwargs = calls[0]
    assert url.endswith("/app/installations/42/access_tokens")
    jwt = kwargs["headers"]["Authorization"].removeprefix("Bearer ")
    assert json.loads(_b64url_decode(jwt.split(".")[1]))["iss"] == "271828"


def _async_return(value):
    async def returns(*_args, **_kwargs):
        return value

    return returns
