import importlib
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


@pytest.fixture(autouse=True)
def _fake_clones_have_a_commit(monkeypatch):
    """The clones here are fake ``.git`` directories; real git would find no commit in them."""
    monkeypatch.setattr(resolve_module, "_has_commit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        resolve_module, "_origin_url", AsyncMock(return_value="https://github.com/org/repo")
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.web_scraper.ssrf_protection"),
        "validate_outbound_url",
        AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _fake_tree_snapshot(monkeypatch):
    original = resolve_module._exec_git

    async def tree_command(args, cwd=None, env=None):
        if args[0] == "rev-parse":
            return 0, "a" * 40, ""
        if args[0] == "ls-tree":
            return 0, "", ""
        return await original(args, cwd=cwd, env=env)

    monkeypatch.setattr(resolve_module, "_exec_git", tree_command)
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.code_graph.repo_selection"),
        "read_repository_snapshot",
        lambda path: {"commit": "a" * 40},
    )


def test_is_remote_repo_detection():
    assert resolve_module.is_remote_repo("https://github.com/org/repo")
    assert resolve_module.is_remote_repo("http://gitlab.example.com/org/repo.git")
    assert resolve_module.is_remote_repo("git@github.com:org/repo.git")
    assert resolve_module.is_remote_repo("ssh://git@github.com/org/repo")
    assert not resolve_module.is_remote_repo("/local/path/repo")
    assert not resolve_module.is_remote_repo("relative/path")
    assert not resolve_module.is_remote_repo(42)


def _origin_is(monkeypatch, url):
    """Stand in for the origin check on a fake ``.git`` directory."""

    async def fake_origin_url(clone):
        return url

    monkeypatch.setattr(resolve_module, "_origin_url", fake_origin_url)


def test_clone_slug_is_stable_and_filesystem_safe():
    slug = resolve_module._clone_slug("https://github.com/org/repo")
    assert (
        resolve_module._clone_slug("https://github.com/org/repo.git")
        == resolve_module._clone_slug("https://github.com/org/repo/")
        == slug
    )
    assert re.fullmatch(r"[0-9a-f]{64}", slug)
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        resolve_module._clone_slug("git@github.com:org/repo.git"),
    )


def test_clone_slugs_of_repositories_that_read_alike_are_apart():
    # Both read 'github.com-a-b-c'; sharing a directory would hand one owner's clone,
    # private or not, to callers of the other.
    first = resolve_module._clone_slug("https://github.com/a-b/c")
    second = resolve_module._clone_slug("https://github.com/a/b-c")

    assert first != second
    assert len(first) == len(second) == 64
    assert resolve_module._clone_slug("https://github.com/a/b-c", "dev") != (
        resolve_module._clone_slug("https://github.com/a-b/c", "dev")
    )


@pytest.mark.asyncio
async def test_local_path_is_returned_as_is(tmp_path):
    resolved = await resolve_module.resolve_repo_source(str(tmp_path))

    assert resolved == tmp_path


@pytest.mark.asyncio
async def test_missing_local_path_raises(tmp_path):
    with pytest.raises(resolve_module.CodeRepositoryError):
        await resolve_module.resolve_repo_source(str(tmp_path / "nope"))


@pytest.mark.asyncio
async def test_remote_refused_when_http_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")

    with pytest.raises(resolve_module.CodeRepositoryError) as exc_info:
        await resolve_module.resolve_repo_source("https://github.com/org/repo", clones_dir=tmp_path)

    assert "ALLOW_HTTP_REQUESTS" in str(exc_info.value)


@pytest.mark.asyncio
async def test_remote_is_shallow_cloned(monkeypatch, tmp_path):
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        "https://github.com/org/repo", clones_dir=tmp_path
    )

    assert resolved == resolve_module.clone_dir("https://github.com/org/repo", clones_dir=tmp_path)
    assert len(git_calls) == 6
    args, _cwd = git_calls[0]
    # core.symlinks=false is a security requirement, not incidental: without it a
    # symlink committed in the remote is materialized and later written through.
    assert args[:5] == ["clone", "-c", "core.symlinks=false", "--depth", "1"]
    assert "https://github.com/org/repo" in args


@pytest.mark.asyncio
async def test_existing_clone_is_fetched_and_reset(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir("https://github.com/org/repo", clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _origin_is(monkeypatch, "https://github.com/org/repo")
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        "https://github.com/org/repo", clones_dir=tmp_path
    )

    assert resolved == clone_dir
    assert git_calls == [
        (
            [
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-recurse-submodules",
                "https://github.com/org/repo",
                "HEAD",
            ],
            clone_dir,
        ),
        (["config", "remote.origin.promisor", "true"], clone_dir),
        (["config", "remote.origin.partialclonefilter", "blob:none"], clone_dir),
        (["config", "core.sparseCheckout", "true"], clone_dir),
        (["config", "core.sparseCheckoutCone", "false"], clone_dir),
        (["reset", "--hard", "a" * 40], clone_dir),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_command", ["fetch", "reset"])
async def test_failed_refresh_never_returns_stale_clone(monkeypatch, tmp_path, failed_command):
    clone_dir = resolve_module.clone_dir("https://github.com/org/repo", clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _origin_is(monkeypatch, "https://github.com/org/repo")

    async def fake_run_git(args, cwd=None, env=None):
        return (1, "fatal: refresh failed") if args[0] == failed_command else (0, "")

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    with pytest.raises(resolve_module.CodeRepositoryError, match="Could not (refresh|checkout)"):
        await resolve_module.resolve_repo_source("https://github.com/org/repo", clones_dir=tmp_path)


@pytest.mark.asyncio
async def test_failed_clone_raises_with_stderr(monkeypatch, tmp_path):
    async def fake_run_git(args, cwd=None, env=None):
        return 128, "fatal: repository not found"

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    with pytest.raises(resolve_module.CodeRepositoryError) as exc_info:
        await resolve_module.resolve_repo_source(
            "https://github.com/org/missing", clones_dir=tmp_path
        )

    assert "repository not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_missing_git_binary_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(resolve_module.shutil, "which", lambda name: None)

    with pytest.raises(resolve_module.CodeRepositoryError) as exc_info:
        await resolve_module.resolve_repo_source("https://github.com/org/repo", clones_dir=tmp_path)

    assert "git is required" in str(exc_info.value)


@pytest.mark.parametrize(
    "spec,clone_url",
    [
        ("https://github.com/org/repo", "https://github.com/org/repo"),
        ("https://github.com/org/repo.git", "https://github.com/org/repo.git"),
        ("https://github.com/org/repo/", "https://github.com/org/repo"),
        ("http://github.com/org/repo", "http://github.com/org/repo"),
        # Browser chrome is dropped so git gets a clean URL.
        (
            "https://www.github.com/org/repo?tab=readme-ov-file#readme",
            "https://www.github.com/org/repo",
        ),
        ("  https://github.com/org/repo  ", "https://github.com/org/repo"),
        ("https://gitlab.com/group/repo", "https://gitlab.com/group/repo"),
        ("https://gitlab.com/group/subgroup/repo", "https://gitlab.com/group/subgroup/repo"),
        # Any host, when the URL says so.
        ("https://git.example.com/team/repo.git", "https://git.example.com/team/repo.git"),
        # Userinfo (a token) stays: it is needed to clone, redacted before storing.
        (
            "https://x-access-token:tok@github.com/org/repo",
            "https://x-access-token:tok@github.com/org/repo",
        ),
    ],
)
def test_code_repo_clone_url_detects_repository_roots(spec, clone_url):
    assert resolve_module.code_repo_clone_url(spec) == clone_url


@pytest.mark.parametrize(
    "spec",
    [
        # Pages inside a repository stay on the web-page path.
        "https://github.com/org/repo/blob/main/README.md",
        "https://github.com/org/repo/tree/main/src",
        "https://github.com/org/repo/issues/12",
        "https://github.com/org/repo/pull/3",
        "https://gitlab.com/group/repo/-/blob/main/README.md",
        "https://gitlab.com/group/repo/-/merge_requests/1",
        # Forge site pages that happen to have two segments.
        "https://github.com/topics/python",
        "https://github.com/orgs/topoteretes",
        "https://gitlab.com/explore/projects",
        # Not a repository root.
        "https://github.com/org",
        "https://github.com",
        # Other hosts need the .git suffix to be treated as a repository.
        "https://git.example.com/team/repo",
        "https://example.com/some/page",
        # Non-http specs are not detected: clone them and pass the directory.
        "git@github.com:org/repo.git",
        "ssh://git@github.com/org/repo",
        "/local/path/repo",
        "Read https://github.com/org/repo tonight",
        "",
        42,
    ],
)
def test_code_repo_clone_url_leaves_pages_and_other_specs_alone(spec):
    assert resolve_module.code_repo_clone_url(spec) is None


def test_default_clones_dir_comes_from_base_config():
    from cognee.base_config import get_base_config

    assert resolve_module.default_clones_dir() == Path(get_base_config().repos_root_directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["https://github.com/someone/else", None])
async def test_a_clone_of_another_repository_is_never_reused(monkeypatch, tmp_path, origin):
    clone_dir = resolve_module.clone_dir("https://github.com/org/repo", clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _origin_is(monkeypatch, origin)
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append(args)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    with pytest.raises(resolve_module.CodeRepositoryError, match="does not match"):
        await resolve_module.resolve_repo_source("https://github.com/org/repo", clones_dir=tmp_path)

    assert git_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/org/repo.git",
        "https://github.com/org/repo/",
        "https://x-access-token:old@github.com/org/repo",  # a legacy remote with a token
    ],
)
async def test_other_spellings_of_the_same_origin_are_reused(monkeypatch, tmp_path, origin):
    clone_dir = resolve_module.clone_dir("https://github.com/org/repo", clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _origin_is(monkeypatch, origin)

    async def fake_run_git(args, cwd=None, env=None):
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        "https://github.com/org/repo", clones_dir=tmp_path
    )

    assert resolved == clone_dir


@pytest.mark.asyncio
async def test_git_never_prompts(monkeypatch):
    seen = {}

    async def fake_create_subprocess_exec(*args, env=None, **kwargs):
        seen.update(env)
        raise RuntimeError("stop")

    monkeypatch.setattr(
        resolve_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_SSH", raising=False)

    with pytest.raises(RuntimeError):
        await resolve_module._run_git(["status"])
    assert seen["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["GIT_SSH_COMMAND"] == "ssh -o BatchMode=yes"

    # A user's own ssh command is left alone.
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i ~/.ssh/deploy_key")
    with pytest.raises(RuntimeError):
        await resolve_module._run_git(
            ["status"],
            env={"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "a.b", "GIT_CONFIG_VALUE_0": "c"},
        )
    assert seen["GIT_SSH_COMMAND"] == "ssh -i ~/.ssh/deploy_key"
    # The caller's config entries come first; the no-stored-credentials ones, the Git LFS
    # and long path ones and the no-background-maintenance ones follow.
    assert (seen["GIT_CONFIG_COUNT"], seen["GIT_CONFIG_KEY_0"]) == ("10", "a.b")
    assert (seen["GIT_CONFIG_KEY_7"], seen["GIT_CONFIG_VALUE_7"]) == ("core.ignorecase", "false")
    assert (seen["GIT_CONFIG_KEY_8"], seen["GIT_CONFIG_VALUE_8"]) == ("maintenance.auto", "false")
    assert (seen["GIT_CONFIG_KEY_9"], seen["GIT_CONFIG_VALUE_9"]) == ("gc.auto", "0")


def test_a_www_prefix_never_makes_a_second_identity():
    from cognee.tasks.code_graph.resolve_repo import repo_key

    assert repo_key("https://www.git.example.com/a/b.git") == repo_key(
        "https://git.example.com/a/b.git"
    )
    assert repo_key("https://www.github.com/Org/Repo") == "github.com/org/repo"
