import importlib
from pathlib import Path

import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


def test_is_remote_repo_detection():
    assert resolve_module.is_remote_repo("https://github.com/org/repo")
    assert resolve_module.is_remote_repo("http://gitlab.example.com/org/repo.git")
    assert resolve_module.is_remote_repo("git@github.com:org/repo.git")
    assert resolve_module.is_remote_repo("ssh://git@github.com/org/repo")
    assert not resolve_module.is_remote_repo("/local/path/repo")
    assert not resolve_module.is_remote_repo("relative/path")
    assert not resolve_module.is_remote_repo(42)


def test_clone_slug_is_stable_and_filesystem_safe():
    assert (
        resolve_module._clone_slug("https://github.com/org/repo.git")
        == resolve_module._clone_slug("https://github.com/org/repo")
        == "github.com-org-repo"
    )
    assert resolve_module._clone_slug("git@github.com:org/repo.git") == "git-github.com-org-repo"


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

    assert resolved == tmp_path / "github.com-org-repo"
    assert len(git_calls) == 1
    args, _cwd = git_calls[0]
    # core.symlinks=false is a security requirement, not incidental: without it a
    # symlink committed in the remote is materialized and later written through.
    assert args[:5] == ["clone", "-c", "core.symlinks=false", "--depth", "1"]
    assert "https://github.com/org/repo" in args


@pytest.mark.asyncio
async def test_existing_clone_is_reused_with_pull(monkeypatch, tmp_path):
    clone_dir = tmp_path / "github.com-org-repo"
    (clone_dir / ".git").mkdir(parents=True)
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        "https://github.com/org/repo", clones_dir=tmp_path
    )

    assert resolved == clone_dir
    assert git_calls == [(["pull", "--ff-only"], clone_dir)]


@pytest.mark.asyncio
async def test_failed_pull_still_reuses_stale_clone(monkeypatch, tmp_path):
    clone_dir = tmp_path / "github.com-org-repo"
    (clone_dir / ".git").mkdir(parents=True)

    async def fake_run_git(args, cwd=None, env=None):
        return 1, "fatal: not fast-forward"

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        "https://github.com/org/repo", clones_dir=tmp_path
    )

    assert resolved == clone_dir


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
        # Non-http specs stay explicit (remember(content_type="code")).
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
