"""Application-layer envelope encryption for files under ``DATA_ROOT_DIRECTORY``.

See ``docs/file-encryption.md``. Entry points:

* :func:`wrap_storage_if_encrypted` — what ``get_file_storage`` and
  ``open_data_file`` call to put a backend behind the layer.
* :func:`is_encrypted_local_file` — for code that must hand a path to a
  reader it does not control (loaders), to know an unsealed copy is needed.
* :func:`register_file_key_provider` — plug in a key management service.
"""

from .config import FileEncryptionConfig, file_encryption_enabled, get_file_encryption_config
from .EncryptedFileStorage import EncryptedFileStorage
from .envelope import DecryptingReader, EnvelopeHeader, encrypt_stream, is_envelope
from .exceptions import FileDecryptionError, FileEncryptionConfigurationError
from .key_providers import (
    AwsKmsKeyProvider,
    DataKey,
    KeyProvider,
    LocalKeyProvider,
    get_key_provider,
    register_file_key_provider,
    reset_key_providers,
)
from .scope import is_encrypted_local_file, is_under_data_root, wrap_storage_if_encrypted

__all__ = [
    "AwsKmsKeyProvider",
    "DataKey",
    "DecryptingReader",
    "EncryptedFileStorage",
    "EnvelopeHeader",
    "FileDecryptionError",
    "FileEncryptionConfig",
    "FileEncryptionConfigurationError",
    "KeyProvider",
    "LocalKeyProvider",
    "encrypt_stream",
    "file_encryption_enabled",
    "get_file_encryption_config",
    "get_key_provider",
    "is_encrypted_local_file",
    "is_envelope",
    "is_under_data_root",
    "register_file_key_provider",
    "reset_key_providers",
    "wrap_storage_if_encrypted",
]
