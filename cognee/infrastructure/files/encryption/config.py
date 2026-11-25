from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Plaintext bytes per AES-GCM chunk. Each chunk is sealed on its own, so this
# is also the random-access granularity of the reader and the peak memory one
# read or write adds. 1 MiB keeps the per-chunk 16-byte tag overhead at
# ~0.0015% while a seek to the end of a 1 GiB PDF decrypts one chunk.
DEFAULT_CHUNK_SIZE = 1024 * 1024


class FileEncryptionConfig(BaseSettings):
    """Application-layer encryption of the objects cognee writes under
    ``DATA_ROOT_DIRECTORY`` (ingested files and the text loaders derive from
    them). Off by default: encryption at rest is then whatever the filesystem,
    volume or object store provides.

    Env prefix ``FILE_ENCRYPTION_``:

    * ``ENABLED`` — turn the layer on. Files written before that stay readable
      (the reader passes plaintext through) and are encrypted the next time
      they are written.
    * ``PROVIDER`` — ``local`` (default) or ``aws_kms``, or a name registered
      with ``register_file_key_provider``.
    * ``KEY`` / ``KEYS`` / ``ACTIVE_KEY_ID`` / ``KEY_FILE`` — the local
      provider's key-encryption keyring; see ``LocalKeyProvider``.
    * ``KMS_KEY_ID`` / ``KMS_REGION`` / ``KMS_ENDPOINT_URL`` — the AWS KMS
      provider's customer master key and client settings.
    * ``CHUNK_SIZE`` — plaintext bytes per sealed chunk.
    """

    model_config = SettingsConfigDict(env_prefix="FILE_ENCRYPTION_", extra="allow")

    enabled: bool = False
    provider: str = "local"

    # Local provider: one base64 32-byte key, or a JSON ring {"<id>": "<base64>"}
    # plus the id new files are written under, or a file holding the single key.
    key: str | None = None
    keys: str | None = None
    active_key_id: str = "1"
    key_file: str | None = None

    # AWS KMS provider.
    kms_key_id: str | None = None
    kms_region: str | None = None
    kms_endpoint_url: str | None = None

    chunk_size: int = DEFAULT_CHUNK_SIZE


@lru_cache
def get_file_encryption_config() -> FileEncryptionConfig:
    return FileEncryptionConfig()


def file_encryption_enabled() -> bool:
    return get_file_encryption_config().enabled
