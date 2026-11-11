"""Unit tests for cognee.modules.integrations.github.app_auth.build_app_jwt.

A real RSA keypair is generated per run and the token verified with the
public half — proving the hand-rolled JWT is a signature GitHub would
accept, not just three base64 blobs.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from cognee.modules.integrations.github import app_auth
from cognee.modules.integrations.github.app_auth import (
    build_app_jwt,
    user_administers_installation,
)

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PEM = _PRIVATE_KEY.private_bytes(
    Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
).decode()


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    settings = "cognee.modules.integrations.github.github_settings.github_settings"
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
    monkeypatch.setattr(
        "cognee.modules.integrations.github.github_settings.github_settings.app_private_key", ""
    )
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
        "cognee.modules.integrations.github.github_settings.github_settings.app_private_key",
        ed_pem,
    )
    with pytest.raises(RuntimeError, match="RSA"):
        build_app_jwt()


# --- user_administers_installation (SDK-832) ---------------------------------


def _session_answering(status: int, body: dict):
    """A fake aiohttp session whose one GET answers ``status``/``body``."""
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


_ORG_INSTALLATION = {"id": 1, "account": {"login": "acme org", "id": 99, "type": "Organization"}}
_USER_INSTALLATION = {"id": 2, "account": {"login": "jane", "id": 7, "type": "User"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, {"state": "active", "role": "admin"}, True),
        (200, {"state": "active", "role": "member"}, False),
        (200, {"state": "pending", "role": "admin"}, False),
        (404, {"message": "Not Found"}, False),
    ],
)
async def test_an_org_installation_needs_an_active_admin_membership(
    monkeypatch, status, body, expected
):
    session = _session_answering(status, body)
    monkeypatch.setattr(app_auth.aiohttp, "ClientSession", lambda **kwargs: session)

    assert await user_administers_installation("user-tok", _ORG_INSTALLATION) is expected

    url = session.get.call_args.args[0]
    # The org login is a path segment, encoded so it cannot re-aim the request.
    assert url == "https://api.github.com/user/memberships/orgs/acme%20org"
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert session.get.call_args.kwargs["headers"]["Authorization"] == "Bearer user-tok"


@pytest.mark.asyncio
async def test_a_rejected_membership_lookup_fails_the_install_instead_of_binding(monkeypatch):
    # 403 is what GitHub answers when the app lacks the Members permission:
    # that is a deployment gap to surface, never a reason to let the bind through.
    session = _session_answering(403, {"message": "Resource not accessible"})
    monkeypatch.setattr(app_auth.aiohttp, "ClientSession", lambda **kwargs: session)

    with pytest.raises(RuntimeError, match="HTTP 403"):
        await user_administers_installation("user-tok", _ORG_INSTALLATION)


@pytest.mark.asyncio
@pytest.mark.parametrize(("user_id", "expected"), [(7, True), (8, False)])
async def test_a_user_installation_must_be_the_users_own_account(monkeypatch, user_id, expected):
    session = _session_answering(200, {"id": user_id, "login": "jane"})
    monkeypatch.setattr(app_auth.aiohttp, "ClientSession", lambda **kwargs: session)

    assert await user_administers_installation("user-tok", _USER_INSTALLATION) is expected

    assert session.get.call_args.args[0] == "https://api.github.com/user"


@pytest.mark.asyncio
async def test_an_installation_without_an_account_is_refused():
    with pytest.raises(RuntimeError, match="no account"):
        await user_administers_installation("user-tok", {"id": 3})
