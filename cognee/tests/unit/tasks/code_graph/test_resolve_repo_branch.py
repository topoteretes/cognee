"""The rules around choosing the branch of a clone that need no real git.

A branch name git would misread never reaches it, a local path takes no branch, the
clone of another branch is removed even where its files are read-only, the URL token
stays out of errors, and the clones of one directory take turns. What git does with a
branch is in test_resolve_repo_real_git.py.
"""

import asyncio
import errno
import importlib
import os
import stat
from unittest.mock import AsyncMock

import pytest

from cognee.tests.utils.clone_state import write_clone_state, write_cloned_state

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
ssrf_module = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")

_URL = "https://github.com/org/repo"
_NAME = "github.com-org-repo"
_TOKEN_URL = "https://x-access-token:tok-SECRET@github.com/org/repo"


@pytest.fixture(autouse=True)
def _no_dns_lookup(monkeypatch):
    # resolve_repo_source resolves the host of an https URL before it clones. Git is
    # mocked here, so the lookup would only tie the tests to the network.
    monkeypatch.setattr(ssrf_module, "validate_outbound_url", AsyncMock())


def _fake_git(results=None):
    """A ``_run_git`` stand-in: a clone leaves what git leaves, ``results`` maps a git
    subcommand to the ``(returncode, stderr)`` it answers with."""
    results = results or {}

    async def run(args, cwd=None, env=None):
        if args[0] == "clone" and args[0] not in results:
            write_cloned_state(args)
        return results.get(args[0], (0, ""))

    return run


@pytest.mark.asyncio
async def test_only_a_usable_branch_name_reaches_git_and_a_local_path_takes_none(
    monkeypatch, tmp_path
):
    names_git_would_misread = ["", "-x", "--upload-pack=touch /tmp/x", "HEAD", "@", "a..b"]
    names_that_are_not_refs = ["a b", "a~1", "a:b", "a/", "a//b", ".a", "a.lock", "a@{b"]
    for branch in names_git_would_misread + names_that_are_not_refs:
        with pytest.raises(ValueError, match="Invalid branch name"):
            resolve_module.check_branch_name(branch)
    for branch in ["main", "feature/x", "release/1.2.3", "v1.0", "+dev", "tag", "feature/ü"]:
        resolve_module.check_branch_name(branch)

    monkeypatch.setattr(resolve_module, "_run_git", AsyncMock())
    with pytest.raises(resolve_module.CodeRepositoryError, match="only supported for remote"):
        await resolve_module.resolve_repo_source(str(tmp_path), branch="dev")
    resolve_module._run_git.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_clone_of_another_branch_is_removed_even_with_read_only_files(
    monkeypatch, tmp_path
):
    """git writes its object files read-only and Windows will not delete those, so a
    plain rmtree left the old clone in place and the new clone failed on it."""
    old = write_clone_state(tmp_path / _NAME, "dev")
    (old / "stale.py").write_text("x = 1\n")
    pack = old / ".git" / "objects" / "pack" / "pack-1.pack"
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_bytes(b"PACK")
    pack.chmod(stat.S_IRUSR)
    monkeypatch.setattr(resolve_module, "_run_git", _fake_git())
    real_unlink = os.unlink

    def unlink_like_windows(path, *, dir_fd=None):
        if not os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode & stat.S_IWUSR:
            raise PermissionError(errno.EACCES, "Access is denied", str(path))
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink_like_windows)

    resolved = await resolve_module.resolve_repo_source(_URL, clones_dir=tmp_path, branch="main")

    assert resolve_module.read_clone_branch(resolved) == "main"
    assert not (resolved / "stale.py").exists()
    assert not pack.exists()


@pytest.mark.asyncio
async def test_a_missing_branch_error_never_carries_the_url_token(monkeypatch, tmp_path):
    answers = {"clone": (128, f"fatal: no such ref at '{_TOKEN_URL}'"), "ls-remote": (2, "")}
    monkeypatch.setattr(resolve_module, "_run_git", _fake_git(answers))

    with pytest.raises(resolve_module.CodeRepositoryError) as error:
        await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path, branch="gone")

    assert "tok-SECRET" not in str(error.value)
    assert f"Branch 'gone' was not found in '{_URL}'" in str(error.value)


@pytest.mark.asyncio
async def test_clones_of_one_directory_take_turns_and_other_directories_do_not(
    monkeypatch, tmp_path
):
    active = {"now": 0, "most": 0}

    async def slow_git(args, cwd=None, env=None):
        active["now"] += 1
        active["most"] = max(active["most"], active["now"])
        await asyncio.sleep(0.01)
        if args[0] == "clone":
            write_cloned_state(args)
        active["now"] -= 1
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", slow_git)
    resolve = resolve_module.resolve_repo_source

    await asyncio.gather(
        resolve(_URL, clones_dir=tmp_path, branch="dev"),
        resolve(_URL, clones_dir=tmp_path, branch="main"),
        resolve(_URL, clones_dir=tmp_path),
    )
    assert active["most"] == 1

    await asyncio.gather(
        resolve("https://github.com/org/one", clones_dir=tmp_path),
        resolve("https://github.com/org/two", clones_dir=tmp_path),
    )
    assert active["most"] == 2
