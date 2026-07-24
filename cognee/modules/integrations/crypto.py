"""Encryption seam for stored third-party integration credentials.

Version 1 encrypts with AES-256-GCM under a **keyring**: a set of 32-byte keys
(base64) each addressed by a short ``key_id``. New rows are written under the
active key id; every row stores the id it was written with, so decryption
always uses the right key. This is what makes **key rotation** possible without
re-encrypting existing rows: add a new key to the ring, point the active id at
it, and old rows keep decrypting under their original id until (optionally)
re-encrypted.

A future KMS envelope scheme (per-row data keys wrapped by a regional CMK)
becomes version 2: readers dispatch on the row's stored ``encryption_version``,
so the upgrade needs no schema migration and version-1 rows stay readable
until re-encrypted.

Provider-agnostic on purpose — every connector's tokens go through the same
two functions, so swapping the scheme is one change, not N.

Configuration (env):
* ``INTEGRATION_CREDENTIALS_KEYS`` — JSON object ``{"<key_id>": "<base64 key>"}``
  holding every currently-decryptable key. Preferred.
* ``INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID`` — the key id new rows are written
  under (must exist in the ring). Defaults to ``"1"``.
* ``INTEGRATION_CREDENTIALS_KEY`` — legacy single-key fallback used only when
  ``INTEGRATION_CREDENTIALS_KEYS`` is unset; loaded into the ring under id
  ``"1"`` so existing deployments keep working unchanged.
"""

import base64
import json
import os
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CURRENT_ENCRYPTION_VERSION = 1

# The key id a single-key (legacy) deployment writes under, and the default
# active id when none is configured.
_DEFAULT_KEY_ID = "1"

# 96-bit nonces are the AES-GCM standard; a fresh random nonce per encryption
# is what makes reusing the same key across rows safe.
_NONCE_SIZE_BYTES = 12


def _load_keyring() -> dict[str, bytes]:
    """Every currently-decryptable key, addressed by id.

    Refuses to run with a missing or wrong-size key: a derived or hardcoded
    fallback would silently produce ciphertext nobody can decrypt after a
    config fix — better to fail the write outright.
    """
    raw_ring = os.getenv("INTEGRATION_CREDENTIALS_KEYS")
    if raw_ring:
        encoded: dict[str, str] = json.loads(raw_ring)
    else:
        legacy = os.getenv("INTEGRATION_CREDENTIALS_KEY")
        if not legacy:
            raise RuntimeError(
                "INTEGRATION_CREDENTIALS_KEYS (or legacy INTEGRATION_CREDENTIALS_KEY) "
                "is not configured"
            )
        encoded = {_DEFAULT_KEY_ID: legacy}

    ring: dict[str, bytes] = {}
    for key_id, encoded_key in encoded.items():
        key = base64.b64decode(encoded_key)
        if len(key) != 32:
            raise RuntimeError(
                f"Integration credentials key '{key_id}' must decode to exactly 32 bytes"
            )
        ring[key_id] = key
    return ring


def _active_key_id() -> str:
    """The key id new rows are encrypted under."""
    return os.getenv("INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID", _DEFAULT_KEY_ID)


def _key_for(key_id: str, ring: dict[str, bytes]) -> bytes:
    key = ring.get(key_id)
    if key is None:
        raise RuntimeError(f"No integration credentials key configured for key_id '{key_id}'")
    return key


def encrypt_credentials(payload: dict[str, Any]) -> tuple[bytes, bytes, int, str]:
    """Serialize and encrypt a credentials payload under the active key.

    Returns ``(ciphertext, nonce, encryption_version, key_id)`` — exactly the
    four columns the caller persists.
    """
    key_id = _active_key_id()
    key = _key_for(key_id, _load_keyring())
    plaintext = json.dumps(payload).encode()
    nonce = os.urandom(_NONCE_SIZE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return ciphertext, nonce, CURRENT_ENCRYPTION_VERSION, key_id


def decrypt_credentials(
    ciphertext: bytes, nonce: bytes, encryption_version: int, key_id: Optional[str]
) -> dict[str, Any]:
    """Decrypt a stored payload, dispatching on its ``encryption_version`` and ``key_id``.

    An unknown version means the row was written by a newer scheme this code
    predates — refuse rather than misinterpret the bytes. A missing ``key_id``
    (a row predating rotation support) is read under the default key id.
    """
    if encryption_version != CURRENT_ENCRYPTION_VERSION:
        raise ValueError(f"Unsupported encryption_version: {encryption_version}")
    key = _key_for(key_id or _DEFAULT_KEY_ID, _load_keyring())
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext)
