"""The origin lookup add() and update() share: which stored document an input came from."""

import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.modules.ingestion.identify_by_origin  # bind the real submodule
from cognee.modules.ingestion.identify_by_origin import (
    Origin,
    OriginMatch,
    expand_directories,
    find_by_origin,
    origin_of,
)

origin_module = sys.modules["cognee.modules.ingestion.identify_by_origin"]

pytestmark = pytest.mark.asyncio


def _upload(content: bytes, filename: str):
    from starlette.datastructures import UploadFile

    spooled = tempfile.SpooledTemporaryFile()  # noqa: SIM115 - handed to UploadFile, which owns it
    spooled.write(content)
    spooled.seek(0)
    return UploadFile(file=spooled, filename=filename)


async def test_a_local_file_is_matched_by_its_location(tmp_path):
    report = tmp_path / "report.txt"
    report.write_text("Quarterly report, version one.\n")

    origin = await origin_of(str(report))
    again = await origin_of(report.as_uri())

    assert origin.location == report.as_uri() and origin.name is None
    assert origin.label == "report.txt"
    assert origin.content_hash and origin == again, "path and file URI are one origin"


async def test_an_upload_is_matched_by_filename():
    uploaded = _upload(b"Meeting notes, version one.\n", "notes.txt")

    upload = await origin_of(uploaded)

    assert upload.name == ("notes", "txt") and upload.location is None
    assert upload.label == "notes.txt" and upload.content_hash
    assert uploaded.file.tell() == 0, "the stream is rewound for ingestion"


async def test_text_and_unknown_inputs_have_no_origin():
    assert await origin_of("plain text, not a path") is None
    assert await origin_of(object()) is None


def test_directories_expand_to_their_files_and_other_inputs_pass_through(tmp_path):
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "b.txt").write_text("B")
    (folder / "a.txt").write_text("A")
    (folder / "sub" / "c.txt").write_text("C")

    expanded = expand_directories(["raw text", str(folder), 42])

    assert expanded[0] == "raw text" and expanded[-1] == 42
    assert [p.rsplit("/", 1)[-1] for p in expanded[1:-1]] == ["a.txt", "b.txt", "c.txt"]


def _engine_returning(rows):
    session = MagicMock()
    session.execute = AsyncMock(return_value=SimpleNamespace(fetchall=lambda: rows))
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.get_async_session = MagicMock(return_value=context)
    return engine, session


async def test_rows_are_assigned_by_location_first_then_by_name():
    by_path = Origin(label="report.txt", content_hash="p", location="file:///docs/report.txt")
    by_name = Origin(label="notes.txt", content_hash="n", name=("notes", "txt"))
    unmatched = Origin(label="gone.txt", content_hash="g", location="file:///docs/gone.txt")
    report_id, notes_id, other_notes = uuid4(), uuid4(), uuid4()
    rows = [
        (report_id, "report", "txt", "file:///docs/report.txt", "stored-p"),
        (notes_id, "notes", "txt", "file:///uploads/abc/notes.txt", "stored-n"),
        (other_notes, "notes", "txt", "file:///uploads/def/notes.txt", "stored-n2"),
        (uuid4(), "notes", "md", "file:///uploads/ghi/notes.md", "different extension"),
    ]
    engine, session = _engine_returning(rows)

    with patch.object(origin_module, "get_relational_engine", MagicMock(return_value=engine)):
        matches = await find_by_origin(
            [by_path, by_name, unmatched], SimpleNamespace(id=uuid4(), tenant_id=None), uuid4()
        )

    assert matches[by_path] == [OriginMatch(data_id=report_id, content_hash="stored-p")]
    assert [m.data_id for m in matches[by_name]] == [notes_id, other_notes]
    assert matches[unmatched] == [], "every origin is a key, matched or not"
    session.execute.assert_awaited_once()


async def test_no_origins_means_no_query():
    engine, session = _engine_returning([])

    with patch.object(origin_module, "get_relational_engine", MagicMock(return_value=engine)):
        assert await find_by_origin([], SimpleNamespace(id=uuid4(), tenant_id=None), uuid4()) == {}

    session.execute.assert_not_awaited()
