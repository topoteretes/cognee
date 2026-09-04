"""Folder uploads: multipart parts with relative filenames become one directory
per top-level folder, written server-side; flat parts pass through untouched.
Every name is validated before any byte lands, and a re-upload replaces the
folder wholesale.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee.tasks.ingestion.folder_uploads as folder_uploads
from cognee.tasks.ingestion.exceptions import InvalidFolderUploadError

USER = SimpleNamespace(id="user-1")


def upload(name, content=b"x"):
    return SimpleNamespace(filename=name, read=AsyncMock(return_value=content))


@pytest.fixture
def uploads_root(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    monkeypatch.setattr(folder_uploads, "folder_uploads_root", lambda: root)
    return root


@pytest.mark.asyncio
async def test_folder_parts_are_written_per_top_level_folder(uploads_root):
    flat = upload("notes.txt")

    flat_out, folders = await folder_uploads.materialize_folder_uploads(
        [
            upload("proj/pyproject.toml", b"[project]"),
            flat,
            upload("proj/src/app.py", b"print(1)"),
            upload("other/readme.md", b"# other"),
        ],
        USER,
    )

    assert flat_out == [flat]
    assert folders == [
        str(uploads_root / "user-1" / "proj"),
        str(uploads_root / "user-1" / "other"),
    ]
    assert (uploads_root / "user-1" / "proj" / "pyproject.toml").read_bytes() == b"[project]"
    assert (uploads_root / "user-1" / "proj" / "src" / "app.py").read_bytes() == b"print(1)"
    assert (uploads_root / "user-1" / "other" / "readme.md").read_bytes() == b"# other"


@pytest.mark.asyncio
async def test_backslash_separators_are_normalised(uploads_root):
    _flat, folders = await folder_uploads.materialize_folder_uploads(
        [upload("proj\\src\\app.py", b"win")], USER
    )

    assert folders == [str(uploads_root / "user-1" / "proj")]
    assert (uploads_root / "user-1" / "proj" / "src" / "app.py").read_bytes() == b"win"


@pytest.mark.asyncio
async def test_without_folder_parts_nothing_is_written(uploads_root):
    a, b = upload("a.txt"), upload("b.pdf")

    flat_out, folders = await folder_uploads.materialize_folder_uploads([a, b], USER)

    assert flat_out == [a, b]
    assert folders == []
    assert not uploads_root.exists()
    a.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_none_uploads_are_fine(uploads_root):
    assert await folder_uploads.materialize_folder_uploads(None, USER) == ([], [])


@pytest.mark.asyncio
async def test_reupload_replaces_the_folder_wholesale(uploads_root):
    await folder_uploads.materialize_folder_uploads(
        [upload("proj/keep.py", b"v1"), upload("proj/stale.py", b"gone")], USER
    )

    await folder_uploads.materialize_folder_uploads([upload("proj/keep.py", b"v2")], USER)

    folder = uploads_root / "user-1" / "proj"
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
async def test_unsafe_names_reject_the_request_before_anything_is_written(uploads_root, name):
    with pytest.raises(InvalidFolderUploadError):
        await folder_uploads.materialize_folder_uploads([upload("proj/ok.py"), upload(name)], USER)

    assert not uploads_root.exists()


def test_flat_names_are_not_folder_parts():
    assert folder_uploads.relative_upload_parts("report.pdf") is None
    assert folder_uploads.relative_upload_parts("") is None
    assert folder_uploads.relative_upload_parts(None) is None
    assert folder_uploads.relative_upload_parts("proj/src/app.py") == ("proj", "src", "app.py")
