"""Unit tests for cognee.modules.integrations.crypto.

Pure functions — no DB, no network. The invariants: roundtrip fidelity,
loud failure on a missing/malformed key (never silent fallback), version
dispatch, AES-GCM's tamper detection, and — the point of the keyring — that a
key rotation leaves old rows decryptable under their original key id.
"""

import base64
import json

import pytest

from cognee.modules.integrations.crypto import (
    CURRENT_ENCRYPTION_VERSION,
    decrypt_credentials,
    encrypt_credentials,
)

_KEY_1 = base64.b64encode(b"1" * 32).decode()
_KEY_2 = base64.b64encode(b"2" * 32).decode()


@pytest.fixture(autouse=True)
def _single_key(monkeypatch):
    # Legacy single-key config — loaded into the ring under id "1".
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEYS", raising=False)
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID", raising=False)
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", _KEY_1)


def test_roundtrip_preserves_payload():
    payload = {"access_token": "xoxb-secret", "refresh_token": "xoxe-1-secret"}
    ciphertext, nonce, version, key_id = encrypt_credentials(payload)

    assert version == CURRENT_ENCRYPTION_VERSION
    assert key_id == "1"
    assert decrypt_credentials(ciphertext, nonce, version, key_id) == payload


def test_ciphertext_is_not_plaintext():
    ciphertext, _, _, _ = encrypt_credentials({"access_token": "xoxb-secret"})
    assert b"xoxb-secret" not in ciphertext


def test_fresh_nonce_per_encryption():
    # Nonce reuse under the same AES-GCM key breaks confidentiality — two
    # encryptions of the same payload must never share a nonce.
    payload = {"access_token": "xoxb-secret"}
    _, nonce_one, _, _ = encrypt_credentials(payload)
    _, nonce_two, _, _ = encrypt_credentials(payload)
    assert nonce_one != nonce_two


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        encrypt_credentials({"access_token": "x"})


def test_wrong_size_key_raises(monkeypatch):
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"short").decode())
    with pytest.raises(RuntimeError, match="32 bytes"):
        encrypt_credentials({"access_token": "x"})


def test_unknown_version_raises():
    ciphertext, nonce, _, key_id = encrypt_credentials({"access_token": "x"})
    with pytest.raises(ValueError, match="Unsupported encryption_version"):
        decrypt_credentials(ciphertext, nonce, CURRENT_ENCRYPTION_VERSION + 1, key_id)


def test_tampered_ciphertext_raises():
    # GCM authenticates the ciphertext — a flipped byte must fail, not decode garbage.
    ciphertext, nonce, version, key_id = encrypt_credentials({"access_token": "x"})
    tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]
    with pytest.raises(Exception):
        decrypt_credentials(tampered, nonce, version, key_id)


def test_rotation_new_rows_use_active_key(monkeypatch):
    # Two keys in the ring, active id points at "2": new rows are written under 2.
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEYS", json.dumps({"1": _KEY_1, "2": _KEY_2}))
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID", "2")

    ciphertext, nonce, version, key_id = encrypt_credentials({"access_token": "new"})
    assert key_id == "2"
    assert decrypt_credentials(ciphertext, nonce, version, key_id) == {"access_token": "new"}


def test_rotation_old_rows_still_decrypt(monkeypatch):
    # A row written under the legacy single key (id "1")...
    ciphertext, nonce, version, key_id = encrypt_credentials({"access_token": "old"})
    assert key_id == "1"

    # ...stays readable after rotating the active key to "2", as long as key "1"
    # remains in the ring.
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEYS", json.dumps({"1": _KEY_1, "2": _KEY_2}))
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID", "2")

    assert decrypt_credentials(ciphertext, nonce, version, key_id) == {"access_token": "old"}


def test_decrypt_with_retired_key_raises(monkeypatch):
    # Key "1" written a row, then "1" was dropped from the ring entirely —
    # decryption must fail loudly, not silently return garbage.
    ciphertext, nonce, version, key_id = encrypt_credentials({"access_token": "old"})

    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEYS", json.dumps({"2": _KEY_2}))
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID", "2")

    with pytest.raises(RuntimeError, match="key_id"):
        decrypt_credentials(ciphertext, nonce, version, key_id)
