"""Regression tests for zip extraction containment in StorageAwareCache (gh #6300).

``download_and_extract_zip()`` joined each archive member name straight onto
the requested cache directory. The local storage backend's containment check
guards the *storage root*, not the selected cache directory, so a member named
``../other-cache/state.json`` was written into a sibling cache directory under
the same root. Extraction must reject every member that does not resolve
beneath the requested cache directory, before any write happens.
"""

import zipfile
from io import BytesIO

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from cognee.base_config import get_base_config
from cognee.shared.cache import StorageAwareCache


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """A zip archive in memory with the given member names and contents."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def cache_manager(tmp_path, monkeypatch):
    """A StorageAwareCache whose storage root is a fresh temporary directory."""
    monkeypatch.setenv("CACHE_ROOT_DIRECTORY", str(tmp_path))
    # get_base_config is lru_cached; earlier imports may have pinned the
    # default root. Rebuild it against the temporary directory, and drop the
    # cached tmp-rooted config afterwards so other tests see the default.
    get_base_config.cache_clear()
    yield StorageAwareCache()
    get_base_config.cache_clear()


@pytest_asyncio.fixture
async def zip_server():
    """Local HTTP server handing out canned zip bodies registered by path."""
    payloads: dict[str, bytes] = {}

    async def serve_zip(request: web.Request) -> web.StreamResponse:
        body = payloads.get(request.path)
        if body is None:
            return web.Response(status=404)
        return web.Response(body=body)

    app = web.Application()
    app.router.add_route("GET", "/{tail:.*}", serve_zip)
    server = TestServer(app)
    await server.start_server()
    server.payloads = payloads
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_zip_member_traversal_is_rejected(cache_manager, zip_server, tmp_path):
    """A member named ``../data-b/state.json`` must not overwrite a sibling cache.

    gh #6300: the member path was joined onto the cache directory unchecked and
    the storage-root check in the local backend still allowed the write (a
    sibling cache directory is inside the root), so one cache's downloaded
    archive could corrupt a different cache under the same root.
    """
    sibling_file = tmp_path / "data-b" / "state.json"
    sibling_file.parent.mkdir()
    sibling_file.write_bytes(b"KEEP")

    zip_server.payloads["/evil.zip"] = _zip_bytes(
        {
            "ok.txt": b"legit",
            "../data-b/state.json": b"pwned",
        }
    )

    with pytest.raises(ValueError, match=r"\.\./data-b/state\.json"):
        await cache_manager.download_and_extract_zip(
            str(zip_server.make_url("/evil.zip")), "data-a", "v1", force=True
        )

    # The sibling cache is untouched, and a rejected archive is rejected whole:
    # validation runs before any extraction write, so nothing lands in data-a.
    assert sibling_file.read_bytes() == b"KEEP"
    assert not (tmp_path / "data-a" / "ok.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("member_name", ["/etc/cognee-pwned.txt", "C:\\Windows\\cognee-pwned.txt"])
async def test_zip_absolute_member_names_are_rejected(
    cache_manager, zip_server, tmp_path, member_name
):
    """Absolute and drive-qualified member names are not cache-relative paths."""
    zip_server.payloads["/abs.zip"] = _zip_bytes({member_name: b"x"})

    with pytest.raises(ValueError, match="cognee-pwned"):
        await cache_manager.download_and_extract_zip(
            str(zip_server.make_url("/abs.zip")), "data-a", "v1", force=True
        )

    assert list(tmp_path.rglob("cognee-pwned.txt")) == []


@pytest.mark.asyncio
async def test_benign_zip_still_extracts(cache_manager, zip_server, tmp_path):
    """Well-formed archives extract exactly as before, directories included."""
    zip_server.payloads["/good.zip"] = _zip_bytes(
        {
            "data/greeting.txt": b"hello",
            "notes/": b"",
            "top.txt": b"top",
        }
    )

    cache_dir = await cache_manager.download_and_extract_zip(
        str(zip_server.make_url("/good.zip")), "data-a", "v1", force=True
    )

    assert (tmp_path / "data-a" / "data" / "greeting.txt").read_bytes() == b"hello"
    assert (tmp_path / "data-a" / "top.txt").read_bytes() == b"top"
    assert (tmp_path / "data-a" / "notes").is_dir()
    assert (tmp_path / "data-a" / "version.txt").read_text() == "v1"
    assert cache_dir.endswith("data-a")
