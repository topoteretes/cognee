"""Folder uploads inside the add pipeline.

Multipart parts with relative filenames are grouped by top-level folder and
written under ``uploads/<user>/<dataset>/<folder>`` by
``resolve_data_directories`` (through ``materialize_folder_uploads``), which
then resolves that directory like any local one: a code project becomes one
``code_repo`` manifest plus its documents, a plain folder is flattened. Flat
parts pass through untouched. Names are validated before any byte lands, a
re-upload replaces the folder wholesale, and the boundary validator gives the
API a 400 for the same mistakes. Background buffering must keep the relative
name or the pipeline never sees the folder.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee.tasks.ingestion.folder_uploads as folder_uploads
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.exceptions import InvalidFolderUploadError
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories
from cognee.tasks.ingestion.utils import materialize_stream_for_background

USER = SimpleNamespace(id="user-1", tenant_id="tenant-1")
DATASET_ID = "dataset-a"


def upload(name, content=b"x"):
    return SimpleNamespace(filename=name, read=AsyncMock(return_value=content))


@pytest.fixture
def uploads_root(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    monkeypatch.setattr(folder_uploads, "folder_uploads_root", lambda: root)
    return root


@pytest.fixture
def llm_key_set(monkeypatch):
    """A repo partition only emits its document half with an LLM key configured."""
    import cognee.infrastructure.llm.config as llm_config_module

    monkeypatch.setattr(
        llm_config_module, "get_llm_config", lambda: SimpleNamespace(llm_api_key="sk-set")
    )


# ── materialize_folder_uploads ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_folder_parts_become_one_directory_each_in_place(uploads_root):
    flat = upload("notes.txt")

    resolved = await folder_uploads.materialize_folder_uploads(
        [
            upload("proj/pyproject.toml", b"[project]"),
            flat,
            upload("proj/src/app.py", b"print(1)"),
            upload("other/readme.md", b"# other"),
        ],
        USER,
        DATASET_ID,
    )

    dataset_root = uploads_root / "user-1" / DATASET_ID
    # The directory takes its first part's position; later parts are dropped.
    assert resolved == [str(dataset_root / "proj"), flat, str(dataset_root / "other")]
    assert (dataset_root / "proj" / "pyproject.toml").read_bytes() == b"[project]"
    assert (dataset_root / "proj" / "src" / "app.py").read_bytes() == b"print(1)"
    assert (dataset_root / "other" / "readme.md").read_bytes() == b"# other"


@pytest.mark.asyncio
async def test_same_folder_name_in_two_datasets_stays_apart(uploads_root):
    await folder_uploads.materialize_folder_uploads([upload("proj/a.py", b"A")], USER, "ds-a")
    await folder_uploads.materialize_folder_uploads([upload("proj/a.py", b"B")], USER, "ds-b")

    assert (uploads_root / "user-1" / "ds-a" / "proj" / "a.py").read_bytes() == b"A"
    assert (uploads_root / "user-1" / "ds-b" / "proj" / "a.py").read_bytes() == b"B"


@pytest.mark.asyncio
async def test_backslash_separators_are_normalised(uploads_root):
    resolved = await folder_uploads.materialize_folder_uploads(
        [upload("proj\\src\\app.py", b"win")], USER, DATASET_ID
    )

    assert resolved == [str(uploads_root / "user-1" / DATASET_ID / "proj")]
    assert (uploads_root / "user-1" / DATASET_ID / "proj" / "src" / "app.py").read_bytes() == b"win"


@pytest.mark.asyncio
async def test_without_folder_parts_items_and_disk_are_untouched(uploads_root):
    a, b, handle = upload("a.txt"), upload("b.pdf"), open(__file__, "rb")
    try:
        items = [a, "some text", b, "/abs/path/file.txt", handle]

        assert await folder_uploads.materialize_folder_uploads(items, None, None) is items
    finally:
        handle.close()
    assert not uploads_root.exists()
    a.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_reupload_replaces_the_folder_wholesale(uploads_root):
    await folder_uploads.materialize_folder_uploads(
        [upload("proj/keep.py", b"v1"), upload("proj/stale.py", b"gone")], USER, DATASET_ID
    )

    await folder_uploads.materialize_folder_uploads(
        [upload("proj/keep.py", b"v2")], USER, DATASET_ID
    )

    folder = uploads_root / "user-1" / DATASET_ID / "proj"
    assert (folder / "keep.py").read_bytes() == b"v2"
    assert not (folder / "stale.py").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../x.py",
        "/etc/passwd",
        "proj//x.py",
        "proj/../x.py",
        "proj/./x.py",
        "C:\\proj\\x.py",
        "proj/",
    ],
)
@pytest.mark.asyncio
async def test_unsafe_names_reject_the_batch_before_anything_is_written(uploads_root, name):
    with pytest.raises(InvalidFolderUploadError):
        await folder_uploads.materialize_folder_uploads(
            [upload("proj/ok.py"), upload(name)], USER, DATASET_ID
        )

    assert not uploads_root.exists()


@pytest.mark.asyncio
async def test_folder_parts_need_the_ingestion_context(uploads_root):
    with pytest.raises(InvalidFolderUploadError, match="user and dataset"):
        await folder_uploads.materialize_folder_uploads([upload("proj/a.py")], None, None)

    assert not uploads_root.exists()


@pytest.mark.asyncio
async def test_labelled_folder_parts_are_rejected(uploads_root):
    wrapped = DataItem(data=upload("proj/a.py"), label="code")

    with pytest.raises(InvalidFolderUploadError, match="labels"):
        await folder_uploads.materialize_folder_uploads([wrapped], USER, DATASET_ID)


# ── through resolve_data_directories ───────────────────────────────────────


@pytest.mark.asyncio
async def test_uploaded_code_project_resolves_to_a_repo_item_plus_documents(
    uploads_root, llm_key_set, monkeypatch
):
    import importlib
    import uuid

    # The package re-exports the function under the module's name; import the
    # real module to patch it. The manifest's stable id is derived through the
    # relational store, and this stays a unit test.
    unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_data_id")
    monkeypatch.setattr(
        unique_id_module, "get_unique_data_id", AsyncMock(return_value=uuid.uuid4())
    )

    resolved = await resolve_data_directories(
        [
            upload("proj/pyproject.toml", b'[project]\nname = "proj"\n'),
            upload("proj/src/app.py", b"def main():\n    pass\n"),
            upload("proj/README.md", b"# proj\n"),
            "a plain note",
        ],
        user=USER,
        dataset_id=DATASET_ID,
    )

    folder = uploads_root / "user-1" / DATASET_ID / "proj"
    manifests = [item for item in resolved if isinstance(item, DataItem)]
    assert len(manifests) == 1
    assert manifests[0].system_metadata["source"] == "code_repo"
    assert manifests[0].system_metadata["repo_path"] == str(folder)
    assert manifests[0].system_metadata["file_count"] == 2
    strings = [item for item in resolved if isinstance(item, str)]
    assert str(folder / "README.md") in strings
    assert "a plain note" in strings


@pytest.mark.asyncio
async def test_uploaded_plain_folder_is_flattened_to_its_files(uploads_root):
    resolved = await resolve_data_directories(
        [upload("docs/a.md", b"# a"), upload("docs/sub/b.txt", b"b")],
        user=USER,
        dataset_id=DATASET_ID,
    )

    folder = uploads_root / "user-1" / DATASET_ID / "docs"
    assert sorted(Path(item).relative_to(folder).as_posix() for item in resolved) == [
        "a.md",
        "sub/b.txt",
    ]


# ── boundary validator (what the routers call) ─────────────────────────────


def test_validate_reports_folder_parts_and_rejects_attributes():
    assert folder_uploads.validate_folder_uploads([upload("proj/a.py"), upload("b.txt")]) is True
    assert folder_uploads.validate_folder_uploads([upload("b.txt")]) is False
    assert folder_uploads.validate_folder_uploads(None) is False
    with pytest.raises(InvalidFolderUploadError, match="labels"):
        folder_uploads.validate_folder_uploads([upload("proj/a.py")], with_attributes=True)
    with pytest.raises(InvalidFolderUploadError):
        folder_uploads.validate_folder_uploads([upload("../escape.py")])
    # Attributes with flat uploads only are fine.
    assert folder_uploads.validate_folder_uploads([upload("b.txt")], with_attributes=True) is False


def test_flat_names_and_non_uploads_are_not_folder_parts():
    assert folder_uploads.relative_upload_parts("report.pdf") is None
    assert folder_uploads.relative_upload_parts("") is None
    assert folder_uploads.relative_upload_parts(None) is None
    assert folder_uploads.relative_upload_parts("proj/src/app.py") == ("proj", "src", "app.py")
    # A local file handle's name is a path on this machine, not a relative name.
    with open(__file__, "rb") as handle:
        assert folder_uploads.folder_parts_of(handle) is None
    assert folder_uploads.folder_parts_of("proj/src/app.py") is None


# ── background buffering keeps relative names ──────────────────────────────


@pytest.mark.asyncio
async def test_background_buffering_keeps_the_relative_upload_name():
    from io import BytesIO

    buffered = await materialize_stream_for_background(
        SimpleNamespace(file=BytesIO(b"x"), filename="proj\\src\\app.py")
    )

    assert buffered.filename == "proj/src/app.py"
    assert buffered.file.read() == b"x"


@pytest.mark.asyncio
async def test_background_buffering_keeps_only_the_basename_of_a_local_handle():
    with open(__file__, "rb") as handle:
        buffered = await materialize_stream_for_background(handle)

    assert buffered.filename == Path(__file__).name
