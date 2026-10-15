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
    # The repo's documents ride along individually, marked as files of a clone
    # cognee made; unrelated items pass through.
    from cognee.tasks.ingestion.repo_clone_file import RepoCloneFile

    documents = [item for item in resolved if isinstance(item, RepoCloneFile)]
    assert {document.path.name for document in documents} == {"README.md", "notes.txt"}
    assert [item for item in resolved if isinstance(item, str)] == ["plain text note"]


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


@pytest.mark.asyncio
async def test_repository_urls_are_cloned_with_credentials(tmp_path, monkeypatch, llm_key_set):
    """add(codegraph_config={"repo_credentials": ...}) hands the token to the cloner."""
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    clone = _make_repo(tmp_path)
    clones = []

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        clones.append((spec, credentials))
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    resolved = await resolve_data_directories(
        ["plain text note", "https://github.com/org/private"], credentials="tok123"
    )

    assert clones == [("https://github.com/org/private", "tok123")]
    assert resolved[0] == "plain text note"
    manifest = resolved[1]
    assert isinstance(manifest, DataItem)
    assert manifest.system_metadata["repo_url"] == "https://github.com/org/private"
    assert "tok123" not in json.dumps(manifest.system_metadata)
    assert {document.path.name for document in resolved[2:]} == {"README.md", "notes.txt"}


@pytest.mark.asyncio
async def test_credentials_do_not_make_a_web_page_a_repository(monkeypatch):
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    async def refuse(*_args, **_kwargs):
        raise AssertionError("nothing here is a repository URL")

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", refuse)
    data = ["https://example.com/article"]

    assert await resolve_data_directories(data, credentials="tok") == data


@pytest.mark.asyncio
async def test_resolution_is_idempotent_on_its_own_output(tmp_path, monkeypatch, llm_key_set):
    """add() resolves ahead of the pipeline runner, which resolves again with defaults."""
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    clone = _make_repo(tmp_path)
    clones = []

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        clones.append(spec)
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    first = await resolve_data_directories(
        ["note", "https://github.com/org/private", str(clone)],
        credentials="tok",
        include_documents=False,
    )
    second = await resolve_data_directories(first)

    assert second == first
    # The second pass neither clones again nor re-emits the suppressed documents.
    assert clones == ["https://github.com/org/private"]


@pytest.fixture
def local_paths_disabled(monkeypatch, tmp_path):
    import importlib

    # The directory must resolve as a local path for the check to be reached.
    monkeypatch.setenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", str(tmp_path))

    # The package re-exports a function of the same name, so import the module.
    storage_module = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")

    monkeypatch.setattr(storage_module.settings, "accept_local_file_path", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_code_project", [True, False])
async def test_local_directories_are_refused_when_local_paths_are_disabled(
    tmp_path, local_paths_disabled, is_code_project
):
    """A code project would otherwise become a manifest that skips the per-file check."""
    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    if is_code_project:
        _make_repo(tmp_path)
    else:
        (tmp_path / "a.md").write_text("# a")

    with pytest.raises(IngestionError, match="ACCEPT_LOCAL_FILE_PATH=false"):
        await resolve_data_directories([str(tmp_path)])


@pytest.mark.asyncio
async def test_repository_urls_still_resolve_when_local_paths_are_disabled(
    tmp_path, monkeypatch, local_paths_disabled
):
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    clone = _make_repo(tmp_path)

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    resolved = await resolve_data_directories(["https://github.com/org/repo"])

    assert any(isinstance(item, DataItem) for item in resolved)


@pytest.mark.asyncio
async def test_clone_documents_are_stored_when_local_paths_are_disabled(
    tmp_path, monkeypatch, llm_key_set, local_paths_disabled
):
    """A cloned repository's documents are cognee's files, not caller-supplied paths."""
    import importlib

    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.tasks.ingestion.repo_clone_file import RepoCloneFile

    storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")
    clone = _make_repo(tmp_path)

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    _manifest, documents, _skipped = await code_repo_module.resolve_code_repository_url(
        "https://github.com/org/repo"
    )
    readme = next(document for document in documents if document.path.name == "README.md")

    stored = await storage.save_data_item_to_storage(readme)
    assert stored == (clone / "README.md").as_uri()

    # The same file named as a plain path is still refused.
    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await storage.save_data_item_to_storage(str(clone / "README.md"))

    with pytest.raises(IngestionError, match="does not exist"):
        await storage.save_data_item_to_storage(RepoCloneFile(clone / "missing.md"))


@pytest.mark.parametrize("index_vectors", [False, True])
def test_code_task_lists_forward_index_vectors(index_vectors):
    """cognify(index_vectors=...) reaches both code adapters as a task param, and
    the default builds graph-only."""
    from cognee.tasks.code_graph.code_files import get_code_file_tasks
    from cognee.tasks.code_graph.code_repo import get_code_repo_tasks

    for tasks in (
        get_code_repo_tasks(index_vectors=index_vectors),
        get_code_file_tasks(index_vectors=index_vectors),
    ):
        [task] = tasks
        assert task.default_params["kwargs"]["index_vectors"] is index_vectors

    for tasks in (get_code_repo_tasks(), get_code_file_tasks()):
        [task] = tasks
        assert task.default_params["kwargs"]["index_vectors"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("index_vectors", [False, True])
async def test_code_repo_route_embeds_only_with_index_vectors(tmp_path, monkeypatch, index_vectors):
    """extract_code_repo_graph stores graph-only by default and embeds with index_vectors."""
    import importlib
    import json
    from contextlib import asynccontextmanager
    from io import StringIO
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cognee.tasks.code_graph.code_repo import extract_code_repo_graph

    # importlib, not `import a.b as m`: these packages re-export a function under
    # the submodule's own name, which the attribute lookup would bind instead.
    open_data_file_module = importlib.import_module(
        "cognee.infrastructure.files.utils.open_data_file"
    )
    extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")

    repo = _make_repo(tmp_path)

    @asynccontextmanager
    async def fake_open(*_args, **_kwargs):
        yield StringIO(json.dumps({"repo_path": str(repo)}))

    monkeypatch.setattr(open_data_file_module, "open_data_file", fake_open)
    monkeypatch.setattr(extract_module, "extract_code_graph", AsyncMock(return_value=[]))
    add_points = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(extract_module, "add_code_graph_data_points", add_points)
    monkeypatch.setattr(extract_module, "add_code_graph_edges", AsyncMock())

    data_item = SimpleNamespace(
        id="d1", raw_data_location="unused", system_metadata={"source": "code_repo"}
    )
    await extract_code_repo_graph([data_item], index_vectors=index_vectors)

    assert add_points.await_args.kwargs["graph_only"] is (not index_vectors)


@pytest.mark.asyncio
async def test_documents_omitted_for_a_code_graph_only_caller(tmp_path, llm_key_set):
    """include_documents=False indexes the code graph alone, LLM key or not."""
    repo = _make_repo(tmp_path)

    from cognee.tasks.code_graph.code_repo import resolve_code_repository

    manifest_item, documents, _skipped = await resolve_code_repository(
        repo, include_documents=False
    )

    assert documents == []
    # The manifest is unaffected: the code graph is built from the same files.
    assert manifest_item.system_metadata["source"] == "code_repo"
    assert manifest_item.system_metadata["file_count"] == 3


@pytest.mark.asyncio
async def test_local_project_resolves_without_documents(tmp_path, llm_key_set):
    """include_documents=False leaves a local code project's README/docs out."""
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    repo = _make_repo(tmp_path)

    resolved = await resolve_data_directories(
        ["plain text note", str(repo)], include_documents=False
    )

    assert resolved[0] == "plain text note"
    assert len(resolved) == 2, resolved
    assert isinstance(resolved[1], DataItem)
    assert resolved[1].system_metadata["source"] == "code_repo"


@pytest.mark.asyncio
async def test_local_project_resolves_with_documents(tmp_path, llm_key_set):
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    repo = _make_repo(tmp_path)

    resolved = await resolve_data_directories([str(repo)], include_documents=True)

    assert {Path(item).name for item in resolved[1:]} == {"README.md", "notes.txt"}


@pytest.mark.asyncio
async def test_repository_url_resolves_without_documents(tmp_path, monkeypatch):
    import cognee.tasks.code_graph.code_repo as code_repo_module
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    clone = _make_repo(tmp_path)

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        return clone

    monkeypatch.setattr(code_repo_module, "resolve_repo_source", fake_resolve_repo_source)

    resolved = await resolve_data_directories(
        ["https://github.com/org/private"], credentials="tok123", include_documents=False
    )

    assert len(resolved) == 1, resolved
    assert resolved[0].system_metadata["repo_url"] == "https://github.com/org/private"


@pytest.mark.asyncio
async def test_plain_directory_is_not_a_repository(tmp_path):
    """include_documents applies to repositories; an ordinary folder still flattens."""
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("# a")

    assert await resolve_data_directories([str(folder)], include_documents=False) == [
        str(folder / "a.md")
    ]
