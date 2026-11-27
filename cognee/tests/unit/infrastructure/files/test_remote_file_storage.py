"""``RemoteFileStorage`` maps relative paths onto ``<scheme>://<bucket>/<prefix>`` the
same way for every object store (S3, Azure Blob), checked on an in-memory filesystem."""

from io import BytesIO

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from cognee.infrastructure.files.storage.RemoteFileStorage import RemoteFileStorage


class _ObjectStoreMemoryFS(MemoryFileSystem):
    """A private in-memory store that, like s3fs and adlfs, globs without a leading slash."""

    def glob(self, path, **kwargs):
        return [found.lstrip("/") for found in super().glob(path, **kwargs)]


class _MemoryStorage(RemoteFileStorage):
    def __init__(self, storage_path: str, scheme: str):
        self.scheme = scheme
        super().__init__(storage_path)

    def _create_filesystem(self):
        fs = _ObjectStoreMemoryFS(skip_instance_cache=True)
        # MemoryFileSystem keeps one class-wide store; give each storage its own.
        fs.store, fs.pseudo_dirs = {}, [""]
        return fs


@pytest.fixture(params=["s3", "az"])
def storage(request) -> _MemoryStorage:
    return _MemoryStorage(f"{request.param}://bucket/root", scheme=request.param)


@pytest.mark.asyncio
async def test_store_returns_a_uri_under_the_root_and_open_reads_it(storage):
    uri = await storage.store("docs/a.bin", BytesIO(b"payload"))

    assert uri == f"{storage.scheme}://bucket/root/docs/a.bin"
    async with storage.open("docs/a.bin", mode="rb") as file:
        assert file.read() == b"payload"


@pytest.mark.asyncio
async def test_store_keeps_an_existing_object_unless_overwrite(storage):
    await storage.store("a.txt", "first")
    await storage.store("a.txt", "second")
    async with storage.open("a.txt") as file:
        assert file.read() == "first"

    await storage.store("a.txt", "third", overwrite=True)
    async with storage.open("a.txt") as file:
        assert file.read() == "third"


@pytest.mark.asyncio
async def test_list_files_is_relative_to_the_root(storage):
    await storage.store("dir/a.txt", "a")
    await storage.store("dir/sub/b.txt", "b")

    assert await storage.list_files("dir") == ["dir/a.txt"]
    assert sorted(await storage.list_files("dir", recursive=True)) == ["dir/a.txt", "dir/sub/b.txt"]
    assert await storage.list_files("missing") == []


@pytest.mark.asyncio
async def test_remove_all_deletes_only_the_given_tree(storage):
    await storage.store("gone/a.txt", "a")
    await storage.store("kept/b.txt", "b")

    await storage.remove_all("gone")
    await storage.remove("never-stored.txt")

    assert not await storage.file_exists("gone/a.txt")
    assert await storage.file_exists("kept/b.txt")


@pytest.mark.asyncio
async def test_filesystem_errors_pass_through_translate_error():
    class _Translating(_MemoryStorage):
        def _translate_error(self, error):
            return ValueError("clear message") if isinstance(error, PermissionError) else error

    storage = _Translating("az://bucket/root", scheme="az")
    storage.fs.exists = lambda path: (_ for _ in ()).throw(PermissionError("denied"))

    with pytest.raises(ValueError, match="clear message") as raised:
        await storage.file_exists("a.txt")
    assert isinstance(raised.value.__cause__, PermissionError)
