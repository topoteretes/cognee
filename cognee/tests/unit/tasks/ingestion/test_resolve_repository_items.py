"""resolve_data_directories: which items become repositories (SDK-793).

Without ``treat_as_repository`` add() recognises repositories itself: forge and
``.git`` URLs, ssh remotes, URLs a git server answers for, and directories with
a project marker. With it, every path or URL item is one. Cloning is stubbed.
"""

import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

code_repo_module = importlib.import_module("cognee.tasks.code_graph.code_repo")
resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
storage_module = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")


@pytest.fixture
def clone(monkeypatch):
    """resolve_code_repository_url stubbed: returns a manifest marker per spec."""

    async def fake(spec, **kwargs):
        return f"manifest:{spec}", [Path("/clones/repo/README.md")], 0

    stub = AsyncMock(side_effect=fake)
    monkeypatch.setattr(code_repo_module, "resolve_code_repository_url", stub)
    return stub


def _source_tree(tmp_path) -> Path:
    """A source tree with no build file and no .git: not detected as a project."""
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "main.py").write_text("def main():\n    return 1\n")
    (tree / "notes.md").write_text("# Notes\n")
    return tree


@pytest.mark.asyncio
async def test_an_ssh_remote_is_cloned_as_a_repository(clone):
    resolved = await resolve_data_directories(
        ["git@github.com:org/repo.git", "a note"], credentials=None
    )

    assert resolved == ["manifest:git@github.com:org/repo.git", "/clones/repo/README.md", "a note"]
    assert clone.await_args.kwargs["clone_url"] == "git@github.com:org/repo.git"


@pytest.mark.asyncio
async def test_a_url_a_git_server_answers_for_is_cloned(clone, monkeypatch):
    monkeypatch.setattr(resolve_module, "_answers_as_git_server", AsyncMock(return_value=True))

    resolved = await resolve_data_directories(["https://git.example.com/team/repo"])

    assert resolved[0] == "manifest:https://git.example.com/team/repo"
    assert clone.await_args.kwargs["clone_url"] == "https://git.example.com/team/repo"


@pytest.mark.asyncio
async def test_a_web_page_stays_a_web_page(clone):
    # The unit conftest's probe answers "not a repository".
    assert await resolve_data_directories(["https://example.com/article"]) == [
        "https://example.com/article"
    ]
    clone.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_probe_does_not_run_for_forge_urls(clone, monkeypatch):
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(resolve_module, "_answers_as_git_server", probe)

    await resolve_data_directories(["https://github.com/org/repo"])

    probe.assert_not_awaited()
    assert clone.await_args.kwargs["clone_url"] == "https://github.com/org/repo"


@pytest.mark.asyncio
async def test_repository_options_reach_the_clone(clone):
    await resolve_data_directories(
        ["https://github.com/org/repo"],
        branch="dev",
        credentials="token",
        include_documents=False,
    )

    kwargs = clone.await_args.kwargs
    assert kwargs["branch"] == "dev"
    assert kwargs["credentials"] == "token"
    assert kwargs["include_documents"] is False


@pytest.mark.asyncio
async def test_a_source_tree_without_a_marker_is_split_into_files(tmp_path):
    tree = _source_tree(tmp_path)

    resolved = await resolve_data_directories([str(tree)])

    assert sorted(Path(item).name for item in resolved) == ["main.py", "notes.md"]


@pytest.mark.asyncio
async def test_treat_as_repository_makes_it_one_repository(tmp_path):
    tree = _source_tree(tmp_path)

    resolved = await resolve_data_directories(
        [Path(tree)], treat_as_repository=True, include_documents=False
    )

    (manifest,) = resolved
    assert manifest.system_metadata["source"] == "code_repo"
    assert manifest.system_metadata["repo_path"] == str(tree)


@pytest.mark.asyncio
async def test_treat_as_repository_clones_any_remote(clone):
    # No .git suffix, an unknown host: only the caller's word makes it a repository.
    await resolve_data_directories(
        ["https://bitbucket.org/team/repo"], treat_as_repository=True, credentials="token"
    )

    assert clone.await_args.kwargs["clone_url"] == "https://bitbucket.org/team/repo"
    assert clone.await_args.kwargs["credentials"] == "token"


@pytest.mark.asyncio
async def test_treat_as_repository_passes_resolved_files_through(tmp_path, clone):
    """The runner resolves items before the add task resolves them again: a
    repository's document paths from the first pass must stay themselves."""
    document = tmp_path / "README.md"
    document.write_text("# Readme\n")

    assert await resolve_data_directories([str(document)], treat_as_repository=True) == [
        str(document)
    ]
    clone.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_repositories_follow_accept_local_file_path(tmp_path, monkeypatch):
    from cognee.modules.ingestion.exceptions import IngestionError

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    monkeypatch.setattr(storage_module.settings, "accept_local_file_path", False)

    with pytest.raises(IngestionError, match="ACCEPT_LOCAL_FILE_PATH"):
        await resolve_data_directories([str(project)])
    with pytest.raises(IngestionError, match="ACCEPT_LOCAL_FILE_PATH"):
        await resolve_data_directories([str(_source_tree(tmp_path))], treat_as_repository=True)
