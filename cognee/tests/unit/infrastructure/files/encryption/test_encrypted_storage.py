"""EncryptedFileStorage against the real local backend, and where it is attached."""

import io
import os

import pytest

from cognee.infrastructure.files.encryption import (
    EncryptedFileStorage,
    FileDecryptionError,
    FileEncryptionConfigurationError,
    is_encrypted_local_file,
    is_envelope,
    is_under_data_root,
    wrap_storage_if_encrypted,
)
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.files.utils.open_data_file import open_data_file


def sealed_bytes(path) -> bytes:
    return path.read_bytes()


@pytest.mark.asyncio
async def test_store_seals_and_open_unseals(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    assert isinstance(storage.storage, EncryptedFileStorage)

    uri = await storage.store("doc.txt", "plain words\nsecond line\n")

    on_disk = sealed_bytes(data_root / "doc.txt")
    assert is_envelope(on_disk)
    assert b"plain words" not in on_disk
    assert uri == (data_root / "doc.txt").as_uri()

    async with storage.open("doc.txt", "rb") as file:
        assert file.read() == b"plain words\nsecond line\n"
        assert file.name == uri
    async with storage.open("doc.txt", "r", encoding="utf-8") as file:
        assert file.read() == "plain words\nsecond line\n"
        assert file.name == uri


@pytest.mark.asyncio
async def test_store_accepts_streams_and_bytes(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    payload = os.urandom(3 * 1024 * 1024 + 17)  # several default chunks plus a tail

    await storage.store("blob.bin", io.BytesIO(payload), overwrite=True)
    async with storage.open("blob.bin", "rb") as file:
        assert file.read() == payload

    await storage.store("raw.bin", payload[:100], overwrite=True)
    async with storage.open("raw.bin", "rb") as file:
        assert file.read() == payload[:100]


@pytest.mark.asyncio
async def test_each_store_uses_a_fresh_data_key(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    await storage.store("a.txt", "same content")
    await storage.store("b.txt", "same content")

    assert sealed_bytes(data_root / "a.txt") != sealed_bytes(data_root / "b.txt")


@pytest.mark.asyncio
async def test_get_size_reports_plaintext_size(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    await storage.store("doc.txt", "x" * 1000)

    assert (data_root / "doc.txt").stat().st_size > 1000
    assert await storage.get_size("doc.txt") == 1000


@pytest.mark.asyncio
async def test_plaintext_written_before_enabling_still_reads(encryption_env, data_root):
    legacy = data_root / "old.txt"
    legacy.write_text("written before the switch")

    storage = get_file_storage(str(data_root))
    assert not await storage.storage.is_sealed("old.txt")
    async with storage.open("old.txt", "rb") as file:
        assert file.read() == b"written before the switch"
    async with storage.open("old.txt", "r", encoding="utf-8") as file:
        assert file.read() == "written before the switch"
    assert await storage.get_size("old.txt") == len("written before the switch")

    # The next write seals it.
    await storage.store("old.txt", "rewritten", overwrite=True)
    assert is_envelope(legacy.read_bytes())


@pytest.mark.asyncio
async def test_open_refuses_write_modes(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    with pytest.raises(ValueError, match="read-only"):
        async with storage.open("doc.txt", "wb"):
            pass


@pytest.mark.asyncio
async def test_metadata_and_hash_are_of_the_plaintext(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    content = b"hash me as the user wrote me"
    await storage.store("doc.txt", io.BytesIO(content))

    async with storage.open("doc.txt", "rb") as file:
        metadata = await get_file_metadata(file)

    import hashlib

    assert metadata["content_hash"] == hashlib.md5(content).hexdigest()
    assert metadata["file_size"] == len(content)
    assert metadata["name"] == "doc"
    assert await get_file_content_hash(str(data_root / "doc.txt")) == metadata["content_hash"]


@pytest.mark.asyncio
async def test_open_data_file_unseals_by_uri_and_path(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    uri = await storage.store("doc.txt", "via open_data_file")

    async with open_data_file(uri, mode="r", encoding="utf-8") as file:
        assert file.read() == "via open_data_file"
    async with open_data_file(str(data_root / "doc.txt"), mode="rb") as file:
        assert file.read() == b"via open_data_file"


@pytest.mark.asyncio
async def test_copy_remove_list_pass_through(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    await storage.store("a/doc.txt", "content")
    await storage.storage.copy_file("a/doc.txt", "b/doc.txt")

    assert sorted(await storage.list_files("", recursive=True)) == ["a/doc.txt", "b/doc.txt"]
    async with storage.open("b/doc.txt", "r") as file:
        assert file.read() == "content"

    await storage.remove("a/doc.txt")
    assert not await storage.file_exists("a/doc.txt")
    await storage.remove_all()
    assert not data_root.exists()


@pytest.mark.asyncio
async def test_wrong_key_at_read_time_is_a_decryption_error(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    await storage.store("doc.txt", "sealed under key one")

    encryption_env()  # a different key, same id
    with pytest.raises(FileDecryptionError, match="differs"):
        async with get_file_storage(str(data_root)).open("doc.txt", "rb") as file:
            file.read()


@pytest.mark.asyncio
async def test_missing_key_fails_the_first_write_with_remediation(encryption_env, data_root):
    encryption_env(key=None)
    with pytest.raises(FileEncryptionConfigurationError, match="no key is configured"):
        get_file_storage(str(data_root))


# -- scope ------------------------------------------------------------------


def test_only_roots_under_the_data_root_are_wrapped(encryption_env, data_root, tmp_path):
    assert is_under_data_root(str(data_root))
    assert is_under_data_root(str(data_root / "dataset-1" / "nested"))
    assert is_under_data_root((data_root / "x").as_uri())
    assert not is_under_data_root(str(tmp_path / "system"))
    assert not is_under_data_root(str(tmp_path))

    assert isinstance(
        wrap_storage_if_encrypted(LocalFileStorage(str(data_root))), EncryptedFileStorage
    )
    assert isinstance(wrap_storage_if_encrypted(LocalFileStorage(str(tmp_path))), LocalFileStorage)
    assert isinstance(get_file_storage(str(tmp_path / "system")).storage, LocalFileStorage)


def test_s3_roots_compare_by_prefix(encryption_env, monkeypatch):
    from cognee.base_config import get_base_config

    monkeypatch.setenv("DATA_ROOT_DIRECTORY", "s3://bucket/cognee/data")
    get_base_config.cache_clear()
    try:
        assert is_under_data_root("s3://bucket/cognee/data")
        assert is_under_data_root("s3://bucket/cognee/data/abc/")
        assert not is_under_data_root("s3://bucket/cognee/data-other")
        assert not is_under_data_root("s3://bucket/cognee/system")
        assert not is_under_data_root("/local/path")
    finally:
        get_base_config.cache_clear()


def test_dataset_scoped_root_counts(encryption_env, data_root, tmp_path):
    from cognee.infrastructure.files.storage.config import file_storage_config

    scoped = tmp_path / "elsewhere" / "dataset-root"
    token = file_storage_config.set({"data_root_directory": str(scoped)})
    try:
        assert is_under_data_root(str(scoped / "files"))
    finally:
        file_storage_config.reset(token)


def test_disabled_layer_wraps_nothing(encryption_env, data_root):
    encryption_env(enabled="false")
    assert isinstance(wrap_storage_if_encrypted(LocalFileStorage(str(data_root))), LocalFileStorage)


@pytest.mark.asyncio
async def test_is_encrypted_local_file(encryption_env, data_root, tmp_path):
    storage = get_file_storage(str(data_root))
    await storage.store("doc.txt", "sealed")
    plain = tmp_path / "plain.txt"
    plain.write_text("plain")

    assert is_encrypted_local_file(str(data_root / "doc.txt"))
    assert not is_encrypted_local_file(str(plain))
    assert not is_encrypted_local_file(str(tmp_path / "missing.txt"))
