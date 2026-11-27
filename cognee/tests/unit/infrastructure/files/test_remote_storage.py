"""The scheme helper decides which paths are remote and which backend serves them."""

import importlib
from types import SimpleNamespace

import pytest

from cognee.infrastructure.files.storage.remote_storage import (
    get_remote_file_storage,
    get_remote_scheme,
    get_remote_storage_class,
    is_remote_path,
)


@pytest.mark.parametrize(
    "path, scheme",
    [
        ("s3://bucket/root", "s3"),
        ("az://container/root/file.txt", "az"),
        ("file:///tmp/data", None),
        ("/tmp/data", None),
        ("relative/s3://not-a-scheme", None),
        ("gs://bucket/root", None),
    ],
)
def test_remote_scheme(path, scheme):
    assert get_remote_scheme(path) == scheme
    assert is_remote_path(path) is (scheme is not None)


def test_backend_lookup():
    from cognee.infrastructure.files.storage.AzureBlobFileStorage import AzureBlobFileStorage
    from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

    assert get_remote_storage_class("s3") is S3FileStorage
    assert get_remote_storage_class("az") is AzureBlobFileStorage
    with pytest.raises(ValueError):
        get_remote_storage_class("gs")
    with pytest.raises(ValueError):
        get_remote_file_storage("/tmp/data")


def _selected_scheme(monkeypatch, path, backend, system_root, data_root):
    """The scheme of the remote backend ``get_file_storage`` picks; None for local storage."""
    module = importlib.import_module("cognee.infrastructure.files.storage.get_file_storage")
    roots = SimpleNamespace(system_root_directory=system_root, data_root_directory=data_root)
    monkeypatch.setenv("STORAGE_BACKEND", backend)
    monkeypatch.setattr(module, "get_base_config", lambda: roots)
    stub = lambda scheme: lambda root: SimpleNamespace(scheme=scheme, storage_path=root)
    monkeypatch.setattr(module, "get_remote_storage_class", stub)
    return getattr(module.get_file_storage(path).storage, "scheme", None)


@pytest.mark.parametrize(
    "path, backend, system_root, data_root, expected",
    [
        ("az://box/root", "", "/sys", "/data", "az"),
        ("s3://bucket/root", "", "/sys", "/data", "s3"),
        ("/tmp/rel", "azure", "az://box/sys", "az://box/data", "az"),
        ("/tmp/rel", "AZURE", "az://box/sys", "az://box/data", "az"),
        ("/tmp/rel", "s3", "s3://b/sys", "s3://b/data", "s3"),
        ("/tmp/rel", "s3", "az://box/sys", "az://box/data", None),
        ("/tmp/rel", "azure", "az://box/sys", "/data", None),
        ("/tmp/rel", "", "/sys", "/data", None),
    ],
)
def test_get_file_storage_selects_the_backend(
    monkeypatch, path, backend, system_root, data_root, expected
):
    assert _selected_scheme(monkeypatch, path, backend, system_root, data_root) == expected


@pytest.mark.parametrize("backend, scheme", [("s3", "s3"), ("azure", "az"), ("AZURE", "az")])
def test_remote_backend_derives_the_cache_root_on_its_bucket(monkeypatch, backend, scheme):
    from cognee.base_config import BaseConfig

    monkeypatch.setenv("STORAGE_BACKEND", backend)
    monkeypatch.setenv("STORAGE_BUCKET_NAME", "box")
    monkeypatch.delenv("CACHE_ROOT_DIRECTORY", raising=False)
    config = BaseConfig(
        data_root_directory=f"{scheme}://box/data", system_root_directory=f"{scheme}://box/sys"
    )

    assert config.cache_root_directory == f"{scheme}://box/cognee/cache"
    assert config.data_root_directory == f"{scheme}://box/data"
