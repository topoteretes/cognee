"""``AzureBlobFileStorage`` fails fast, with a configuration error, on missing settings, a
missing container or refused credentials, and checks each container once per process."""

import sys
from types import SimpleNamespace

import pytest

from cognee.infrastructure.files.storage import AzureBlobFileStorage as azure_module
from cognee.infrastructure.files.storage.azure_config import AzureStorageConfigurationError
from cognee.infrastructure.files.storage.AzureBlobFileStorage import AzureBlobFileStorage


class _FakeAdlfs:
    """Stands in for adlfs.AzureBlobFileSystem: one account, a set of existing containers."""

    def __init__(self, containers=("box",), error=None):
        self.service_client = SimpleNamespace(account_name="acct")
        self.containers = set(containers)
        self.error = error
        self.container_checks = 0

    def exists(self, path):
        if self.error is not None:
            raise self.error
        if "/" not in path:
            self.container_checks += 1
            return path in self.containers
        return False


@pytest.fixture(autouse=True)
def _forget_checked_containers():
    azure_module._CHECKED_CONTAINERS.clear()
    yield
    azure_module._CHECKED_CONTAINERS.clear()
    azure_module.get_azure_config.cache_clear()


def _storage(fs: _FakeAdlfs, root: str = "az://box/root") -> AzureBlobFileStorage:
    storage = AzureBlobFileStorage.__new__(AzureBlobFileStorage)  # skip building a real adlfs
    storage.storage_path, storage.fs = root, fs
    return storage


def test_missing_extra_names_the_azure_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "adlfs", None)

    with pytest.raises(ImportError, match=r"cognee\[azure\]"):
        AzureBlobFileStorage("az://box/root")


def test_no_account_is_a_configuration_error(monkeypatch):
    for name in ("CONNECTION_STRING", "ACCOUNT_NAME", "ACCOUNT_KEY", "SAS_TOKEN"):
        monkeypatch.delenv(f"AZURE_STORAGE_{name}", raising=False)
    azure_module.get_azure_config.cache_clear()

    with pytest.raises(AzureStorageConfigurationError, match="AZURE_STORAGE_ACCOUNT_NAME"):
        AzureBlobFileStorage("az://box/root")


@pytest.mark.asyncio
async def test_container_is_checked_once_per_process():
    fs = _FakeAdlfs()

    await _storage(fs).file_exists("a.txt")
    await _storage(fs, "az://box/other").file_exists("b.txt")
    await _storage(fs).file_exists("c.txt")

    assert fs.container_checks == 1


@pytest.mark.asyncio
async def test_missing_container_fails_the_first_read():
    storage = _storage(_FakeAdlfs(containers=()))

    with pytest.raises(
        AzureStorageConfigurationError, match="no container 'box' of account 'acct'"
    ):
        await storage.list_files("")
    assert azure_module._CHECKED_CONTAINERS == set()


def _azure_error(code: str):
    from azure.core.exceptions import HttpResponseError

    error = HttpResponseError(message=f"{code} from Azure")
    error.error_code = code
    return error


@pytest.mark.asyncio
async def test_refused_credentials_fail_the_container_check():
    storage = _storage(_FakeAdlfs(error=_azure_error("AuthorizationFailure")))

    with pytest.raises(AzureStorageConfigurationError, match="refused access to container 'box'"):
        await storage.file_exists("a.txt")


def test_write_errors_wrapped_by_adlfs_are_translated():
    storage = _storage(_FakeAdlfs())
    wrapped = RuntimeError("Failed to upload block")
    wrapped.__cause__ = _azure_error("AuthenticationFailed")

    assert isinstance(storage._translate_error(wrapped), AzureStorageConfigurationError)
    assert storage._translate_error(RuntimeError("other")).args == ("other",)
