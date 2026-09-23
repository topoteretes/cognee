"""Code-project detection and partition for directory adds.

A directory carrying a project marker resolves to ONE repo-level manifest
item plus its document files; junk is skipped. These tests lock in the
partition semantics: code and manifests go to the repo item, prose stays in
the document pipeline, and binaries/dotfiles can never abort an add.
"""

import json
from pathlib import Path

import pytest

from cognee.tasks.code_graph.code_repo import (
    build_repo_manifest,
    detect_code_project,
    partition_repo_files,
)


def _make_repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "sample"\n')
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "util.py").write_text("def helper():\n    pass\n")
    (tmp_path / "README.md").write_text("# Sample\n\nDocs live here.\n")
    (tmp_path / "notes.txt").write_text("plain notes")
    (tmp_path / ".env").write_text("SECRET_KEY=never-ingest-this")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\xa7\x00binary")
    (tmp_path / "model.bin").write_bytes(b"\x00\x01\x02binary-blob")
    # Silent decorative video: no audio stream, whisper transcription would
    # crash on it — repo partition must skip media outright.
    (tmp_path / "logo-loop.mp4").write_bytes(b"\x00\x00\x00 ftypisom")
    return tmp_path


def test_project_marker_detection(tmp_path):
    assert not detect_code_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("")
    assert detect_code_project(tmp_path)


def test_git_directory_is_a_project_marker(tmp_path):
    (tmp_path / ".git").mkdir()
    assert detect_code_project(tmp_path)


def test_partition_buckets(tmp_path):
    repo = _make_repo(tmp_path)

    covered, documents, skipped = partition_repo_files(repo)

    covered_names = {path.relative_to(repo).as_posix() for path in covered}
    document_names = {path.relative_to(repo).as_posix() for path in documents}
    skipped_names = {path.relative_to(repo).as_posix() for path in skipped}

    # Code + project manifests are covered by the single repo item.
    assert covered_names == {"pyproject.toml", "app.py", "lib/util.py"}
    # Prose stays in the document pipeline, individually.
    assert document_names == {"README.md", "notes.txt"}
    # Dotfiles (secrets), caches, and binaries are never ingested — so one
    # .pyc can no longer abort a directory add.
    assert skipped_names == {
        ".env",
        "__pycache__/app.cpython-311.pyc",
        "model.bin",
        "logo-loop.mp4",
    }


def test_manifest_hash_tracks_code_content(tmp_path):
    repo = _make_repo(tmp_path)
    covered, _documents, _skipped = partition_repo_files(repo)

    manifest_before = json.loads(build_repo_manifest(repo, covered))
    manifest_same = json.loads(build_repo_manifest(repo, covered))
    (repo / "app.py").write_text("def main():\n    return 1\n")
    manifest_after = json.loads(build_repo_manifest(repo, covered))

    assert manifest_before["content_hash"] == manifest_same["content_hash"]
    assert manifest_before["content_hash"] != manifest_after["content_hash"]
    assert manifest_before["repo_path"] == str(repo)
    assert manifest_before["file_count"] == 3


@pytest.mark.asyncio
async def test_directory_with_project_resolves_to_repo_item_plus_documents(tmp_path):
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    repo = _make_repo(tmp_path)

    resolved = await resolve_data_directories([str(repo)])

    manifest_items = [item for item in resolved if isinstance(item, DataItem)]
    file_items = [item for item in resolved if isinstance(item, str)]
    assert len(manifest_items) == 1
    assert manifest_items[0].system_metadata["source"] == "code_repo"
    assert manifest_items[0].system_metadata["file_count"] == 3
    assert {Path(item).name for item in file_items} == {"README.md", "notes.txt"}


@pytest.mark.asyncio
async def test_directory_without_project_still_flattens(tmp_path):
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    (tmp_path / "a.md").write_text("# a")
    (tmp_path / "b.txt").write_text("b")

    resolved = await resolve_data_directories([str(tmp_path)])

    assert sorted(Path(item).name for item in resolved) == ["a.md", "b.txt"]


@pytest.mark.asyncio
async def test_documents_excluded_without_llm_api_key(tmp_path, monkeypatch):
    """A key-less repo add must not emit document items that would only fail
    later in LLM pipelines; the code graph itself needs no key."""
    from types import SimpleNamespace

    import cognee.infrastructure.llm.config as llm_config_module
    from cognee.tasks.code_graph.code_repo import resolve_code_repository

    repo = _make_repo(tmp_path)
    monkeypatch.setattr(
        llm_config_module, "get_llm_config", lambda: SimpleNamespace(llm_api_key=None)
    )

    manifest_item, documents, skip_count = await resolve_code_repository(repo)

    assert manifest_item.system_metadata["source"] == "code_repo"
    assert documents == []
    assert skip_count == 6  # .env, pyc, bin, mp4 + README.md + notes.txt


@pytest.mark.asyncio
async def test_documents_kept_with_llm_api_key(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cognee.infrastructure.llm.config as llm_config_module
    from cognee.tasks.code_graph.code_repo import resolve_code_repository

    repo = _make_repo(tmp_path)
    monkeypatch.setattr(
        llm_config_module, "get_llm_config", lambda: SimpleNamespace(llm_api_key="sk-set")
    )

    _manifest_item, documents, skip_count = await resolve_code_repository(repo)

    assert {path.name for path in documents} == {"README.md", "notes.txt"}
    assert skip_count == 4


@pytest.mark.asyncio
async def test_documents_omitted_for_a_code_graph_only_caller(tmp_path, llm_key_set):
    """remember(content_type="code") indexes the code graph only: the repo item
    covers the same files as a repo add, and no document paths come back even
    with an LLM key."""
    from cognee.tasks.code_graph.code_repo import resolve_code_repository

    repo = _make_repo(tmp_path)

    manifest_item, documents, _skip_count = await resolve_code_repository(
        repo, include_documents=False
    )

    assert documents == []
    assert manifest_item.system_metadata["file_count"] == 3


@pytest.mark.asyncio
async def test_add_code_repository_stores_the_pinned_repo_item(tmp_path, monkeypatch):
    """The Data row remember(content_type="code") needs is the same pinned manifest
    item add(<repo>) ingests, added without its documents and returned by id."""
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from cognee.tasks.code_graph.code_repo import add_code_repository

    # importlib, not `import a.b as m`: these packages re-export a function under
    # the submodule's own name, which the attribute lookup would bind instead.
    add_module = importlib.import_module("cognee.api.v1.add")
    data_methods_module = importlib.import_module("cognee.modules.data.methods")
    unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_data_id")

    repo = _make_repo(tmp_path)
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), name="my_code")
    pinned_id = uuid4()
    data_row = SimpleNamespace(id=pinned_id)

    monkeypatch.setattr(unique_id_module, "get_unique_data_id", AsyncMock(return_value=pinned_id))
    calls = []
    reset_mock = AsyncMock(side_effect=lambda *_args, **_kwargs: calls.append("reset"))
    monkeypatch.setattr(data_methods_module, "reset_data_pipeline_status", reset_mock)
    add_mock = AsyncMock(side_effect=lambda *_args, **_kwargs: calls.append("add"))
    monkeypatch.setattr(add_module, "add", add_mock)
    get_data_mock = AsyncMock(return_value=data_row)
    monkeypatch.setattr(data_methods_module, "get_data", get_data_mock)

    stored = await add_code_repository(
        repo, user=user, dataset=dataset, source_url="https://github.com/org/repo"
    )

    assert stored is data_row
    # Only the pinned row's add stamp is cleared before add(), or its
    # incremental check would skip a changed repo's manifest as already added;
    # the cognify stamps stay for ingestion's content comparison to decide.
    reset_mock.assert_awaited_once_with(pinned_id, dataset.id, pipeline_names=("add_pipeline",))
    assert calls == ["reset", "add"]
    manifest_item = add_mock.await_args.args[0]
    assert manifest_item.data_id == pinned_id
    assert manifest_item.system_metadata["source"] == "code_repo"
    assert manifest_item.system_metadata["repo_url"] == "https://github.com/org/repo"
    assert add_mock.await_args.kwargs["dataset_id"] == dataset.id
    assert add_mock.await_args.kwargs["user"] is user
    get_data_mock.assert_awaited_once_with(user.id, pinned_id, dataset.id)


@pytest.mark.asyncio
async def test_add_code_repository_surfaces_an_errored_ingest(tmp_path, monkeypatch):
    """With RAISE_INCREMENTAL_LOADING_ERRORS=false add() returns an errored run
    instead of raising; the caller must see its cause, not a missing row."""
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
    from cognee.tasks.code_graph.code_repo import add_code_repository
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    add_module = importlib.import_module("cognee.api.v1.add")
    data_methods_module = importlib.import_module("cognee.modules.data.methods")
    unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_data_id")

    dataset = SimpleNamespace(id=uuid4(), name="my_code")
    monkeypatch.setattr(unique_id_module, "get_unique_data_id", AsyncMock(return_value=uuid4()))
    monkeypatch.setattr(data_methods_module, "reset_data_pipeline_status", AsyncMock())
    errored = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        error_class="IngestionError",
        error_message="disk full",
    )
    monkeypatch.setattr(add_module, "add", AsyncMock(return_value=errored))
    get_data_mock = AsyncMock()
    monkeypatch.setattr(data_methods_module, "get_data", get_data_mock)

    with pytest.raises(CodeRepositoryError, match="disk full"):
        await add_code_repository(
            _make_repo(tmp_path), user=SimpleNamespace(id=uuid4()), dataset=dataset
        )
    get_data_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_symlinks_are_not_followed_into_the_manifest(tmp_path):
    """rglob + is_file() both follow symlinks, and read_bytes() would then hash and
    index the TARGET. A repo containing 'creds.py -> ~/.aws/credentials' must not
    pull that file's contents into the code graph."""
    from cognee.tasks.code_graph.code_repo import partition_repo_files

    secret = tmp_path / "outside_secret.py"
    secret.write_text("AWS_SECRET_ACCESS_KEY = 'leaked'")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'")
    (repo / "real.py").write_text("x = 1")
    (repo / "creds.py").symlink_to(secret)

    covered, documents, _skipped = partition_repo_files(repo)

    indexed = {p.name for p in covered} | {p.name for p in documents}
    assert "real.py" in indexed
    assert "creds.py" not in indexed, "symlink was followed into the manifest"


@pytest.fixture
def llm_key_set(monkeypatch):
    """The document half of a repo partition is only emitted with an LLM key."""
    from types import SimpleNamespace

    import cognee.infrastructure.llm.config as llm_config_module

    monkeypatch.setattr(
        llm_config_module, "get_llm_config", lambda: SimpleNamespace(llm_api_key="sk-set")
    )


@pytest.mark.asyncio
async def test_repository_url_is_cloned_and_resolved_like_a_project(
    tmp_path, monkeypatch, llm_key_set
):
    """A GitHub URL in add() is a code project, not a web page: it is cloned and
    then partitioned exactly like a local project directory."""
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    clone = tmp_path / "github.com-org-repo"
    clone.mkdir()
    _make_repo(clone)
    cloned_specs = []

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        cloned_specs.append(spec)
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    resolved = await resolve_data_directories(
        ["https://github.com/org/repo?tab=readme-ov-file", "plain text note"]
    )

    # git gets the normalised URL, not the browser one.
    assert cloned_specs == ["https://github.com/org/repo"]
    manifest_items = [item for item in resolved if isinstance(item, DataItem)]
    assert len(manifest_items) == 1
    manifest = manifest_items[0]
    assert manifest.system_metadata["source"] == "code_repo"
    assert manifest.system_metadata["repo_path"] == str(clone)
    assert manifest.system_metadata["repo_url"] == "https://github.com/org/repo"
    assert manifest.system_metadata["file_count"] == 3
    # The repo's documents ride along individually; unrelated items pass through.
    string_items = [item for item in resolved if isinstance(item, str)]
    assert {Path(item).name for item in string_items if item != "plain text note"} == {
        "README.md",
        "notes.txt",
    }
    assert "plain text note" in string_items


@pytest.mark.asyncio
async def test_repository_url_credentials_are_redacted_from_the_manifest(
    tmp_path, monkeypatch, llm_key_set
):
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    clone = _make_repo(tmp_path)
    cloned_specs = []

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        cloned_specs.append(spec)
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    resolved = await resolve_data_directories(["https://x-access-token:tok@github.com/org/repo"])

    # The token reaches git and nothing else.
    assert cloned_specs == ["https://x-access-token:tok@github.com/org/repo"]
    manifest = next(item for item in resolved if isinstance(item, DataItem))
    assert manifest.system_metadata["repo_url"] == "https://github.com/org/repo"
    assert "tok" not in json.dumps(manifest.system_metadata)


@pytest.mark.asyncio
async def test_forge_page_urls_pass_through_to_the_web_page_path(monkeypatch):
    """A link into a repository (blob, issue, ...) is a page: no clone is attempted."""
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    async def refuse(*_args, **_kwargs):
        raise AssertionError("a page URL must not be cloned")

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", refuse)
    urls = [
        "https://github.com/org/repo/blob/main/README.md",
        "https://gitlab.com/group/repo/-/issues/1",
        "https://example.com/article",
    ]

    assert await resolve_data_directories(urls) == urls


@pytest.mark.asyncio
async def test_resolve_code_repository_url_rejects_non_repository_specs():
    from cognee.tasks.code_graph.code_repo import resolve_code_repository_url

    with pytest.raises(ValueError, match="not a repository URL"):
        await resolve_code_repository_url("https://github.com/org/repo/blob/main/README.md")
