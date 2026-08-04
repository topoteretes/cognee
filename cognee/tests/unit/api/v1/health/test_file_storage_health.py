import importlib
from types import SimpleNamespace

import pytest

from cognee.api.v1.health.health import HealthChecker, HealthStatus
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.files.storage.StorageManager import StorageManager


base_config_module = importlib.import_module("cognee.base_config")
file_storage_module = importlib.import_module(
    "cognee.infrastructure.files.storage.get_file_storage"
)
health_module = importlib.import_module("cognee.api.v1.health.health")


def _use_storage(monkeypatch, storage, root_path):
    monkeypatch.setattr(
        base_config_module,
        "get_base_config",
        lambda: SimpleNamespace(data_root_directory=str(root_path)),
    )
    monkeypatch.setattr(file_storage_module, "get_file_storage", lambda _: storage)


class _RecordingStorageManager:
    def __init__(self, fail_after_store=False):
        self.fail_after_store = fail_after_store
        self.objects = {}
        self.store_keys = []
        self.remove_keys = []

    async def store(self, key, data, overwrite=False):
        self.store_keys.append(key)
        data.seek(0)
        self.objects[key] = data.read()
        if self.fail_after_store:
            raise OSError("store failed after creating the probe")
        return key

    async def remove(self, key):
        self.remove_keys.append(key)
        self.objects.pop(key, None)


@pytest.mark.asyncio
async def test_file_storage_health_preserves_legacy_probe_file(monkeypatch, tmp_path):
    legacy_probe = tmp_path / "health_check_test"
    legacy_probe.write_text("real user data")
    storage = StorageManager(LocalFileStorage(str(tmp_path)))
    _use_storage(monkeypatch, storage, tmp_path)

    result = await HealthChecker().check_file_storage()

    assert result.status == HealthStatus.HEALTHY
    assert legacy_probe.read_text() == "real user data"
    assert [path.name for path in tmp_path.iterdir()] == ["health_check_test"]


@pytest.mark.asyncio
async def test_file_storage_health_uses_unique_keys_and_cleans_each_probe(monkeypatch):
    storage = _RecordingStorageManager()
    probe_ids = iter([SimpleNamespace(hex="probe-one"), SimpleNamespace(hex="probe-two")])
    monkeypatch.setattr(health_module, "uuid4", lambda: next(probe_ids))
    _use_storage(monkeypatch, storage, "s3://bucket/data")

    first = await HealthChecker().check_file_storage()
    second = await HealthChecker().check_file_storage()

    assert first.status == HealthStatus.HEALTHY
    assert second.status == HealthStatus.HEALTHY
    assert storage.store_keys == [".cognee-health-probe-one", ".cognee-health-probe-two"]
    assert storage.remove_keys == storage.store_keys
    assert storage.objects == {}


@pytest.mark.asyncio
async def test_file_storage_health_cleans_probe_when_store_partially_fails(monkeypatch):
    storage = _RecordingStorageManager(fail_after_store=True)
    monkeypatch.setattr(health_module, "uuid4", lambda: SimpleNamespace(hex="partial-write"))
    _use_storage(monkeypatch, storage, "s3://bucket/data")

    result = await HealthChecker().check_file_storage()

    assert result.status == HealthStatus.UNHEALTHY
    assert "store failed after creating the probe" in result.details
    assert storage.store_keys == [".cognee-health-partial-write"]
    assert storage.remove_keys == storage.store_keys
    assert storage.objects == {}
