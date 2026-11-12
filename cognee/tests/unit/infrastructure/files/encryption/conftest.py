import base64
import os

import pytest

from cognee.infrastructure.files.encryption import get_file_encryption_config, reset_key_providers


def fresh_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


@pytest.fixture
def encryption_env(monkeypatch):
    """Turn file encryption on with a fresh local key; rebuilds cached config/providers.

    Returns a function that applies further ``FILE_ENCRYPTION_*`` overrides and
    refreshes the caches again.
    """

    def configure(**overrides):
        monkeypatch.setenv("FILE_ENCRYPTION_ENABLED", "true")
        monkeypatch.setenv("FILE_ENCRYPTION_PROVIDER", "local")
        monkeypatch.setenv("FILE_ENCRYPTION_KEY", fresh_key())
        for name in (
            "FILE_ENCRYPTION_KEYS",
            "FILE_ENCRYPTION_KEY_FILE",
            "FILE_ENCRYPTION_KMS_KEY_ID",
        ):
            monkeypatch.delenv(name, raising=False)
        for name, value in overrides.items():
            env_name = f"FILE_ENCRYPTION_{name.upper()}"
            if value is None:
                monkeypatch.delenv(env_name, raising=False)
            else:
                monkeypatch.setenv(env_name, str(value))
        get_file_encryption_config.cache_clear()
        reset_key_providers()
        return get_file_encryption_config()

    configure()
    yield configure
    get_file_encryption_config.cache_clear()
    reset_key_providers()


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point DATA_ROOT_DIRECTORY at a temp dir and rebuild the base config."""
    from cognee.base_config import get_base_config

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(root))
    get_base_config.cache_clear()
    yield root
    get_base_config.cache_clear()
