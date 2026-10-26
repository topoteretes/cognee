"""Repository identity, lock ownership and commit checks on real Git objects."""

import asyncio
import hashlib
import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

resolve = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


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
    with pytest.raises(
        resolve.CodeRepositoryError,
        match="(no checked-out commit|Could not resolve repository commit)",
    ):
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
