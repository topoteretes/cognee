"""Key providers for file envelope encryption.

Every stored object is sealed under its own random 256-bit data key. The
provider's job is only to wrap that data key for the file header and unwrap it
again on read, so the key-encryption key (KEK) itself never touches disk:

* ``local`` holds the KEK in the process environment (a keyring addressed by
  id, like the integration-credential keyring, so a key can be rotated by
  adding a new id and pointing ``ACTIVE_KEY_ID`` at it while old files keep
  decrypting under theirs).
* ``aws_kms`` never sees the KEK at all: ``GenerateDataKey`` returns the data
  key already wrapped by the customer's KMS key and ``Decrypt`` unwraps it, so
  key custody stays in the customer's account.

Third parties register their own with :func:`register_file_key_provider`.
"""

import base64
import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives.keywrap import InvalidUnwrap, aes_key_unwrap, aes_key_wrap

from .config import FileEncryptionConfig, get_file_encryption_config
from .exceptions import FileDecryptionError, FileEncryptionConfigurationError

DATA_KEY_SIZE = 32

# Every KMS call carries this context, and KMS refuses to decrypt a data key
# whose context differs — so a data key wrapped for cognee's file storage cannot
# be unwrapped by a caller claiming another purpose, even with the same CMK.
KMS_ENCRYPTION_CONTEXT = {"cognee:purpose": "file-storage"}

_GENERATE_KEY_HINT = (
    "Generate a key with: python -c 'import os,base64;"
    "print(base64.b64encode(os.urandom(32)).decode())'"
)


@dataclass(frozen=True)
class DataKey:
    plaintext: bytes
    wrapped: bytes
    key_id: str


class KeyProvider(Protocol):
    name: str

    def generate_data_key(self) -> DataKey:
        """A fresh data key, plaintext for sealing plus its wrapped form for the header."""

    def unwrap_data_key(self, wrapped: bytes, key_id: str) -> bytes:
        """The plaintext data key a header's wrapped key stands for."""


class LocalKeyProvider:
    """A KEK ring held in the environment; data keys are AES-key-wrapped (RFC 3394)."""

    name = "local"

    def __init__(self, keyring: dict[str, bytes], active_key_id: str) -> None:
        if active_key_id not in keyring:
            raise FileEncryptionConfigurationError(
                f"FILE_ENCRYPTION_ACTIVE_KEY_ID '{active_key_id}' is not in the key ring.",
                remediation="Add the key under that id to FILE_ENCRYPTION_KEYS, or point "
                "FILE_ENCRYPTION_ACTIVE_KEY_ID at an id that exists.",
            )
        self._keyring = keyring
        self._active_key_id = active_key_id

    @classmethod
    def from_config(cls, config: FileEncryptionConfig) -> "LocalKeyProvider":
        encoded: dict[str, str]
        if config.keys:
            try:
                encoded = json.loads(config.keys)
            except json.JSONDecodeError as error:
                raise FileEncryptionConfigurationError(
                    "FILE_ENCRYPTION_KEYS is not a JSON object of key ids to base64 keys.",
                    remediation='Example: FILE_ENCRYPTION_KEYS=\'{"1": "<base64 32 bytes>"}\'',
                ) from error
            if not isinstance(encoded, dict) or not encoded:
                raise FileEncryptionConfigurationError(
                    "FILE_ENCRYPTION_KEYS must be a non-empty JSON object.",
                    remediation='Example: FILE_ENCRYPTION_KEYS=\'{"1": "<base64 32 bytes>"}\'',
                )
        elif config.key:
            encoded = {config.active_key_id: config.key}
        elif config.key_file:
            key_path = Path(os.path.expanduser(config.key_file))
            if not key_path.is_file():
                raise FileEncryptionConfigurationError(
                    f"FILE_ENCRYPTION_KEY_FILE '{config.key_file}' does not exist.",
                    remediation="Write a base64-encoded 32-byte key to that path "
                    "(python -c 'import os,base64;print(base64.b64encode(os.urandom(32)).decode())').",
                )
            encoded = {config.active_key_id: key_path.read_text().strip()}
        else:
            raise FileEncryptionConfigurationError(
                "FILE_ENCRYPTION_ENABLED is on with the local provider, but no key is configured.",
                remediation="Set FILE_ENCRYPTION_KEY to a base64 32-byte key (generate one with "
                "python -c 'import os,base64;print(base64.b64encode(os.urandom(32)).decode())'), "
                "or FILE_ENCRYPTION_KEY_FILE, or FILE_ENCRYPTION_KEYS for a ring; or set "
                "FILE_ENCRYPTION_PROVIDER=aws_kms.",
            )

        keyring: dict[str, bytes] = {}
        for key_id, encoded_key in encoded.items():
            try:
                key = base64.b64decode(encoded_key, validate=True)
            except (ValueError, TypeError) as error:
                raise FileEncryptionConfigurationError(
                    f"File encryption key '{key_id}' is not valid base64.",
                    remediation=_GENERATE_KEY_HINT,
                ) from error
            if len(key) != DATA_KEY_SIZE:
                raise FileEncryptionConfigurationError(
                    f"File encryption key '{key_id}' must decode to exactly {DATA_KEY_SIZE} bytes, "
                    f"got {len(key)}.",
                    remediation=_GENERATE_KEY_HINT,
                )
            keyring[str(key_id)] = key

        return cls(keyring, config.active_key_id)

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    def key_fingerprint(self, key_id: str) -> str:
        """A short, non-secret identifier of a ring key, for diagnostics."""
        return hashlib.sha256(self._keyring[key_id]).hexdigest()[:12]

    def generate_data_key(self) -> DataKey:
        data_key = os.urandom(DATA_KEY_SIZE)
        wrapped = aes_key_wrap(self._keyring[self._active_key_id], data_key)
        return DataKey(plaintext=data_key, wrapped=wrapped, key_id=self._active_key_id)

    def unwrap_data_key(self, wrapped: bytes, key_id: str) -> bytes:
        kek = self._keyring.get(key_id)
        if kek is None:
            raise FileDecryptionError(
                f"The file was encrypted under key id '{key_id}', which is not in the "
                "configured FILE_ENCRYPTION_KEYS ring."
            )
        try:
            return aes_key_unwrap(kek, wrapped)
        except InvalidUnwrap as error:
            raise FileDecryptionError(
                f"The data key could not be unwrapped with key id '{key_id}': the configured "
                "key differs from the one the file was written under."
            ) from error


class AwsKmsKeyProvider:
    """Data keys generated and unwrapped by AWS KMS under a customer-managed key."""

    name = "aws_kms"

    def __init__(
        self,
        kms_key_id: str,
        region: str | None = None,
        endpoint_url: str | None = None,
        client=None,
    ) -> None:
        if not kms_key_id:
            raise FileEncryptionConfigurationError(
                "FILE_ENCRYPTION_PROVIDER=aws_kms needs FILE_ENCRYPTION_KMS_KEY_ID.",
                remediation="Set FILE_ENCRYPTION_KMS_KEY_ID to the key ARN, key id or alias "
                "(alias/cognee-files) of a symmetric KMS key the runtime role may use for "
                "GenerateDataKey and Decrypt.",
            )
        self._kms_key_id = kms_key_id
        self._region = region
        self._endpoint_url = endpoint_url
        self._client = client
        self._client_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: FileEncryptionConfig) -> "AwsKmsKeyProvider":
        return cls(
            kms_key_id=config.kms_key_id or "",
            region=config.kms_region or os.getenv("AWS_REGION"),
            endpoint_url=config.kms_endpoint_url,
        )

    def _kms(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        import boto3
                    except ImportError as error:
                        raise FileEncryptionConfigurationError(
                            "The aws_kms file encryption provider needs boto3.",
                            remediation='Install it with: pip install cognee"[aws]"',
                        ) from error
                    self._client = boto3.client(
                        "kms", region_name=self._region, endpoint_url=self._endpoint_url
                    )
        return self._client

    def generate_data_key(self) -> DataKey:
        response = self._kms().generate_data_key(
            KeyId=self._kms_key_id,
            KeySpec="AES_256",
            EncryptionContext=KMS_ENCRYPTION_CONTEXT,
        )
        return DataKey(
            plaintext=bytes(response["Plaintext"]),
            wrapped=bytes(response["CiphertextBlob"]),
            # The ARN KMS answers with, not the alias we asked by: an alias can be
            # repointed, the header must name the key that actually wrapped this.
            key_id=response["KeyId"],
        )

    def unwrap_data_key(self, wrapped: bytes, key_id: str) -> bytes:
        # Errors propagate as they are. A missing boto3 is a
        # FileEncryptionConfigurationError from _kms() with its remediation; a
        # refused Decrypt is botocore's ClientError carrying AWS's own code
        # (AccessDeniedException, InvalidCiphertextException, ...). Wrapping
        # either as a FileDecryptionError would blame the file for a problem
        # with the host or the account.
        response = self._kms().decrypt(
            CiphertextBlob=wrapped,
            KeyId=key_id,
            EncryptionContext=KMS_ENCRYPTION_CONTEXT,
        )
        return bytes(response["Plaintext"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KeyProviderFactory = Callable[[FileEncryptionConfig], KeyProvider]

_PROVIDER_FACTORIES: dict[str, KeyProviderFactory] = {
    LocalKeyProvider.name: LocalKeyProvider.from_config,
    AwsKmsKeyProvider.name: AwsKmsKeyProvider.from_config,
}
_PROVIDER_INSTANCES: dict[str, KeyProvider] = {}
_PROVIDER_LOCK = threading.Lock()


def register_file_key_provider(name: str, factory: KeyProviderFactory) -> None:
    """Make ``FILE_ENCRYPTION_PROVIDER=<name>`` build providers with ``factory``.

    ``factory`` receives the :class:`FileEncryptionConfig`; extra settings a
    custom provider needs can ride on it as ``FILE_ENCRYPTION_*`` variables
    (the config accepts unknown fields).
    """
    with _PROVIDER_LOCK:
        _PROVIDER_FACTORIES[name] = factory
        _PROVIDER_INSTANCES.pop(name, None)


def get_key_provider(config: FileEncryptionConfig | None = None) -> KeyProvider:
    """The configured provider, built once per process.

    Building validates the configuration, so the first encrypted store or
    open raises :class:`FileEncryptionConfigurationError` with a remediation
    when the keys are missing or malformed — never a half-working layer.
    """
    config = config or get_file_encryption_config()
    with _PROVIDER_LOCK:
        instance = _PROVIDER_INSTANCES.get(config.provider)
        if instance is None:
            factory = _PROVIDER_FACTORIES.get(config.provider)
            if factory is None:
                raise FileEncryptionConfigurationError(
                    f"Unknown FILE_ENCRYPTION_PROVIDER '{config.provider}'.",
                    remediation=f"Use one of {sorted(_PROVIDER_FACTORIES)} or register the "
                    "provider with cognee.infrastructure.files.encryption."
                    "register_file_key_provider().",
                )
            instance = factory(config)
            _PROVIDER_INSTANCES[config.provider] = instance
        return instance


def reset_key_providers() -> None:
    """Drop cached provider instances and unwrapped keys (configuration changed, tests)."""
    with _PROVIDER_LOCK:
        _PROVIDER_INSTANCES.clear()
    with _UNWRAP_LOCK:
        _UNWRAPPED.clear()


# ---------------------------------------------------------------------------
# Unwrap cache
# ---------------------------------------------------------------------------

# Opening a file costs one unwrap. For KMS that is a network call, and the same
# object is opened several times per ingestion (hash, classify, load), so the
# unwrapped data key is kept for a bounded number of recent files. Plaintext
# data keys only ever live in process memory.
_UNWRAP_CACHE_SIZE = 256
_UNWRAPPED: OrderedDict[tuple[str, str, bytes], bytes] = OrderedDict()
_UNWRAP_LOCK = threading.Lock()


def unwrap_data_key_cached(provider: KeyProvider, wrapped: bytes, key_id: str) -> bytes:
    cache_key = (provider.name, key_id, wrapped)
    with _UNWRAP_LOCK:
        data_key = _UNWRAPPED.get(cache_key)
        if data_key is not None:
            _UNWRAPPED.move_to_end(cache_key)
            return data_key

    data_key = provider.unwrap_data_key(wrapped, key_id)

    with _UNWRAP_LOCK:
        _UNWRAPPED[cache_key] = data_key
        while len(_UNWRAPPED) > _UNWRAP_CACHE_SIZE:
            _UNWRAPPED.popitem(last=False)
    return data_key
