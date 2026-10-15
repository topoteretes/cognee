"""The rules around choosing the branch of a clone that need no real git.

A branch name git would misread never reaches it, a local path takes no branch, each
branch gets a directory of its own, and the URL token stays out of errors. What git does
with a branch is in test_resolve_repo_real_git.py.
"""

import importlib
from unittest.mock import AsyncMock

import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
ssrf_module = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")

_URL = "https://github.com/org/repo"
_TOKEN_URL = "https://x-access-token:tok-SECRET@github.com/org/repo"


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


def test_each_branch_has_a_directory_of_its_own(tmp_path):
    default = resolve_module.clone_dir(_URL, clones_dir=tmp_path)
    # The directory every clone of the URL had before a branch could be chosen.
    assert default == tmp_path / "github.com-org-repo"

    branches = ["dev", "Dev", "feature/x", "feature-x", "release/" + "x" * 300]
    dirs = [resolve_module.clone_dir(_URL, branch, tmp_path) for branch in branches]

    # Apart even where the file system folds case or the name had to be rewritten.
    assert len({str(path).lower() for path in [default, *dirs]}) == len(branches) + 1
    assert all(path.parent == tmp_path and len(path.name) < 100 for path in dirs)


@pytest.mark.asyncio
async def test_a_missing_branch_is_reported_before_cloning_without_the_url_token(
    monkeypatch, tmp_path
):
    # resolve_repo_source resolves the host of an https URL before it clones. Git is
    # mocked here, so the lookup would only tie the test to the network.
    monkeypatch.setattr(ssrf_module, "validate_outbound_url", AsyncMock())
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append(args[0])
        return (2, "") if args[0] == "ls-remote" else (0, "")

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    with pytest.raises(resolve_module.CodeRepositoryError) as error:
        await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path, branch="gone")

    assert "tok-SECRET" not in str(error.value)
    assert f"Branch 'gone' was not found in '{_URL}'" in str(error.value)
    assert git_calls == ["ls-remote"]
