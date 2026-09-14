"""Unit tests for cognee.modules.integrations.github.app_auth.build_app_jwt.

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

from cognee.modules.integrations.github.app_auth import build_app_jwt

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
