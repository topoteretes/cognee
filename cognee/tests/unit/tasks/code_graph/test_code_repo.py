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

    covered, documents, skipped = partition_repo_files(repo)

    indexed = {p.name for p in covered} | {p.name for p in documents}
    assert "real.py" in indexed
    assert "creds.py" not in indexed, "symlink was followed into the manifest"
