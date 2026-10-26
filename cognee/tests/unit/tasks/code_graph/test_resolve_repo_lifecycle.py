"""Repository identity, lock ownership and commit checks on real Git objects."""

import asyncio
import hashlib
import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

resolve = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


@pytest.fixture(autouse=True)
def isolated_clone_root(tmp_path, monkeypatch):
    monkeypatch.setattr(resolve, "default_clones_dir", lambda: tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/Acme/API.git",
        "http://www.GitHub.com/acme/api/",
        "https://user:secret@github.com/acme/api?tab=readme#section",
        "git@github.com:acme/api.git",
        "ssh://git@github.com/acme/api.git",
    ],
)
def test_transport_and_credentials_do_not_change_repository_identity(url):
    assert resolve.repo_key(url) == "github.com/acme/api"
    assert resolve.repo_identity(url) == hashlib.sha256(b"github.com/acme/api").hexdigest()


def test_repository_identity_separates_owners_hosts_and_case_sensitive_paths():
    urls = [
        "https://github.com/acme/secret-api",
        "https://github.com/acme-secret/api",
        "https://gitlab.com/acme/api",
        "https://git.example.com/acme/api",
        "https://git.example.com/acme/API",
        "https://git.example.com:8443/acme/API",
    ]
    assert len({resolve.repo_identity(url) for url in urls}) == len(urls)


@pytest.mark.parametrize("url", [None, 1, "", "/local/api", "https://github.com", "git@github.com"])
def test_repository_key_requires_a_remote_path(url):
    with pytest.raises(ValueError):
        resolve.repo_key(url)


@pytest.mark.asyncio
async def test_lock_is_reentrant_only_in_its_own_task(tmp_path):
    clone = tmp_path / "clone"
    acquired = asyncio.Event()

    async def other_reader():
        async with resolve._clone_lock(clone):
            acquired.set()

    async with resolve._clone_lock(clone):
        async with resolve._clone_lock(clone):
            reader = asyncio.create_task(other_reader())
            await asyncio.sleep(0.1)
            assert not acquired.is_set()
        await asyncio.sleep(0.1)
        assert not acquired.is_set()
    await asyncio.wait_for(reader, 2)
    assert acquired.is_set()
    # POSIX keeps the inode to prevent locking a replacement file. Filelock's
    # Windows implementation removes its file on release; exclusion above is
    # the shared contract on both platforms.
    if sys.platform != "win32":
        assert (tmp_path / ".locks" / "clone.lock").is_file()


@pytest.mark.asyncio
async def test_cancelled_lock_waiter_does_not_acquire_later(tmp_path):
    clone = tmp_path / "clone"
    acquired = asyncio.Event()

    async def reader():
        async with resolve._clone_lock(clone):
            acquired.set()

    async with resolve._clone_lock(clone):
        waiter = asyncio.create_task(reader())
        await asyncio.sleep(0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    async with resolve._clone_lock(clone):
        assert not acquired.is_set()


@pytest.mark.asyncio
async def test_clone_lock_excludes_another_process(tmp_path):
    clone = tmp_path / "clone"
    lock_path = tmp_path / ".locks" / "clone.lock"
    code = (
        "from filelock import FileLock, Timeout\n"
        "import sys\n"
        "try:\n"
        "    with FileLock(sys.argv[1], timeout=0):\n"
        "        sys.exit(1)\n"
        "except Timeout:\n"
        "    sys.exit(0)\n"
    )
    async with resolve._clone_lock(clone):
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(lock_path))
        assert await asyncio.wait_for(process.wait(), 5) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["", 42, ["token"]])
async def test_invalid_credentials_do_not_start_git(monkeypatch, tmp_path, invalid):
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.web_scraper.ssrf_protection"),
        "validate_outbound_url",
        AsyncMock(),
    )
    run = AsyncMock()
    monkeypatch.setattr(resolve, "_run_git", run)
    with pytest.raises(ValueError, match="non-empty string"):
        await resolve.resolve_repo_source("https://github.com/acme/api", tmp_path, invalid)
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_clone_verification_never_publishes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.web_scraper.ssrf_protection"),
        "validate_outbound_url",
        AsyncMock(),
    )
    monkeypatch.setattr(resolve, "_run_git", AsyncMock(return_value=(0, "")))
    monkeypatch.setattr(
        resolve, "_origin_url", AsyncMock(return_value="https://github.com/acme/api")
    )
    monkeypatch.setattr(resolve, "_has_commit", AsyncMock(return_value=False))
    with pytest.raises(resolve.CodeRepositoryError, match="no checked-out commit"):
        await resolve.resolve_repo_source("https://github.com/acme/api", tmp_path)
    assert not resolve.clone_dir("https://github.com/acme/api", clones_dir=tmp_path).exists()
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.asyncio
async def test_git_diagnostics_redact_raw_and_encoded_credentials(monkeypatch, tmp_path):
    import base64

    secret = "test-only-token"
    encoded = base64.b64encode(f"x-access-token:{secret}".encode()).decode()
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.web_scraper.ssrf_protection"),
        "validate_outbound_url",
        AsyncMock(),
    )
    monkeypatch.setattr(resolve, "_run_git", AsyncMock(return_value=(128, f"{secret}: {encoded}")))
    with pytest.raises(resolve.CodeRepositoryError) as error:
        await resolve.resolve_repo_source("https://github.com/acme/api", tmp_path, secret)
    assert secret not in str(error.value)
    assert encoded not in str(error.value)


@pytest.mark.asyncio
async def test_clone_lock_rejects_a_target_outside_configured_storage(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside") / "clone"
    with pytest.raises(resolve.CodeRepositoryError, match="outside its storage root") as raised:
        async with resolve._clone_lock(outside):
            pytest.fail("Untrusted clone path acquired a lock")
    assert raised.value.status_code == 500
    assert not outside.parent.exists()
    assert asyncio.current_task() not in resolve._HELD_CLONE_LOCKS


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Creating symlinks requires Windows privileges")
@pytest.mark.parametrize("inside_storage", [False, True])
async def test_clone_lock_rejects_a_symlink_target(tmp_path, inside_storage):
    root = tmp_path / "clones"
    root.mkdir()
    destination = (root if inside_storage else tmp_path) / "real-clone"
    destination.mkdir()
    target = root / "clone"
    target.symlink_to(destination, target_is_directory=True)

    with pytest.raises(resolve.CodeRepositoryError, match="outside its storage root"):
        async with resolve._clone_lock(target, clones_dir=root):
            pytest.fail("Symlink clone target acquired a lock")

    assert not (root / ".locks").exists()
    assert asyncio.current_task() not in resolve._HELD_CLONE_LOCKS


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Creating symlinks requires Windows privileges")
@pytest.mark.parametrize("link_directory", [False, True])
async def test_clone_lock_refuses_symlinks_outside_storage(tmp_path, link_directory):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "clones"
    root.mkdir()
    if link_directory:
        (root / ".locks").symlink_to(outside, target_is_directory=True)
    else:
        (root / ".locks").mkdir()
        (root / ".locks" / "clone.lock").symlink_to(outside / "secret")
    with pytest.raises(resolve.CodeRepositoryError, match="outside"):
        async with resolve._clone_lock(root / "clone", clones_dir=root):
            pytest.fail("Escaped lock path was accepted")
    assert list(outside.iterdir()) == []
    assert asyncio.current_task() not in resolve._HELD_CLONE_LOCKS


@pytest.mark.asyncio
async def test_lock_wait_warning_carries_no_name_and_no_path(tmp_path, monkeypatch):
    from unittest.mock import Mock

    target = tmp_path / ("a1" * 32)
    warning = Mock()
    monkeypatch.setattr(resolve.logger, "warning", warning)
    monkeypatch.setattr(resolve, "_CLONE_LOCK_LOG_SECONDS", 0)
    observed = asyncio.Event()

    async def wait_for_lock():
        observed.set()
        async with resolve._clone_lock(target):
            pass

    async with resolve._clone_lock(target):
        waiter = asyncio.create_task(wait_for_lock())
        try:
            await observed.wait()

            async def warned():
                while not warning.called:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(warned(), timeout=5)
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
    assert warning.called
    assert target.name not in str(warning.call_args_list)
    assert str(tmp_path) not in str(warning.call_args_list)
