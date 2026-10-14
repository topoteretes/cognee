"""Saved payloads are keyed by content, and describe themselves accurately.

Two independent things are pinned here:

* Storage keys used to be the caller's filename, written with ``overwrite=True``
  — so two different uploads that happened to share a name silently clobbered
  each other. Keys are content addressed now.
* The returned ``metadata`` is computed from the payload in hand rather than by
  reading the object back, which is what lets ingestion skip a HEAD + GET per
  file. It has to agree with what reading it back would have said.
"""

import importlib
import io
from urllib.parse import unquote

import pytest

from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules import ingestion

# The package __init__ rebinds the name ``save_data_to_file`` to the function,
# shadowing the module it lives in — import the module explicitly to patch it.
save_module = importlib.import_module("cognee.modules.ingestion.save_data_to_file")
save_data_to_file_detailed = save_module.save_data_to_file_detailed


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        save_module, "get_storage_config", lambda: {"data_root_directory": str(tmp_path)}
    )
    return tmp_path


def _upload(content: bytes):
    return io.BytesIO(content)


def _stored_files(root):
    return [p for p in root.rglob("*") if p.is_file()]


@pytest.mark.asyncio
async def test_same_name_different_content_does_not_clobber(storage_root):
    first = await save_data_to_file_detailed(_upload(b"first payload"), filename="report.txt")
    second = await save_data_to_file_detailed(_upload(b"second payload"), filename="report.txt")

    assert first.file_path != second.file_path
    assert len(_stored_files(storage_root)) == 2

    async with open_data_file(first.file_path) as file:
        assert file.read() == b"first payload"
    async with open_data_file(second.file_path) as file:
        assert file.read() == b"second payload"


@pytest.mark.asyncio
async def test_readding_the_same_file_is_idempotent(storage_root):
    first = await save_data_to_file_detailed(_upload(b"same bytes"), filename="a.txt")
    second = await save_data_to_file_detailed(_upload(b"same bytes"), filename="a.txt")

    assert first.file_path == second.file_path
    assert len(_stored_files(storage_root)) == 1


@pytest.mark.asyncio
async def test_same_content_under_two_names_stays_two_objects(storage_root):
    # Parity with the old filename-keyed layout: the DB dedups by content, but
    # the raw copies live under each name the user gave them.
    first = await save_data_to_file_detailed(_upload(b"same bytes"), filename="a.txt")
    second = await save_data_to_file_detailed(_upload(b"same bytes"), filename="b.txt")

    assert first.file_path != second.file_path
    assert len(_stored_files(storage_root)) == 2


@pytest.mark.asyncio
async def test_key_keeps_the_basename_downstream_consumers_read(storage_root):
    # The code-graph route stages files under basename(original_data_location)
    # and keys node identity on it; loaders select by suffix; dlt names its
    # source from it. The key must therefore end with the real filename.
    stored = await save_data_to_file_detailed(_upload(b"%PDF-1.4 fake"), filename="paper.pdf")

    assert stored.file_path.endswith("/paper.pdf")


@pytest.mark.asyncio
async def test_metadata_matches_reading_the_object_back(storage_root):
    stored = await save_data_to_file_detailed(_upload(b"hello world"), filename="Report Q1.txt")

    async with open_data_file(stored.file_path) as file:
        read_back = await ingestion.classify(file).aget_metadata()

    for field in ("content_hash", "mime_type", "extension", "file_size"):
        assert stored.metadata[field] == read_back[field], field


@pytest.mark.asyncio
async def test_name_is_the_document_name_not_the_content_addressed_key(storage_root):
    # Data.name is user-facing. The key is a hash now, so the name has to come
    # from the filename the caller supplied.
    stored = await save_data_to_file_detailed(_upload(b"hello"), filename="Quarterly Report.pdf")

    assert stored.metadata["name"] == "Quarterly Report"
    assert stored.metadata["file_path"] == stored.file_path


@pytest.mark.asyncio
async def test_text_payloads_keep_their_text_named_key(storage_root):
    # text_<md5>.txt is constructed by hand elsewhere in cognee (the cloud
    # client asserts on it), so text stays the one naming source it always was.
    stored = await save_data_to_file_detailed("a plain note")

    assert stored.file_path.rsplit("/", 1)[-1].startswith("text_")


@pytest.mark.asyncio
async def test_text_metadata_is_complete(storage_root):
    # TextData used to report only name + content_hash; ingestion builds a row
    # straight from this dict, so a partial one silently dropped the row's
    # extension, mime type and size.
    stored = await save_data_to_file_detailed("a plain note")

    assert stored.metadata["extension"] == "txt"
    assert stored.metadata["mime_type"] == "text/plain"
    assert stored.metadata["file_size"] == len(b"a plain note")


# --- Named text (CLO-1023) ----------------------------------------------------
# Text a caller names (a connector row's title, DataItem.name) is a document
# with that name, stored under a content-addressed key like any file. Only
# nameless text keeps the text_<md5>.txt convention.


@pytest.mark.asyncio
async def test_named_text_is_stored_under_its_name(storage_root):
    stored = await save_data_to_file_detailed("Ship the thing.", filename="Q3 Plan")

    assert unquote(stored.file_path).endswith("/Q3 Plan.txt")
    assert stored.metadata["name"] == "Q3 Plan"
    assert stored.metadata["extension"] == "txt"
    assert stored.metadata["mime_type"] == "text/plain"
    assert stored.metadata["file_path"] == stored.file_path


@pytest.mark.asyncio
async def test_named_text_keeps_the_source_suffix_in_the_document_name(storage_root):
    # A Drive export's text is still text: read back as .txt, shown as roadmap.docx.
    stored = await save_data_to_file_detailed("exported body", filename="roadmap.docx")

    assert stored.file_path.endswith("/roadmap.docx.txt")
    assert stored.metadata["name"] == "roadmap.docx"
    assert stored.metadata["extension"] == "txt"


@pytest.mark.asyncio
async def test_two_named_texts_sharing_a_name_do_not_clobber(storage_root):
    first = await save_data_to_file_detailed("first notes", filename="Notes")
    second = await save_data_to_file_detailed("second notes", filename="Notes")

    assert first.file_path != second.file_path
    assert first.metadata["name"] == second.metadata["name"] == "Notes"
    assert len(_stored_files(storage_root)) == 2

    async with open_data_file(first.file_path) as file:
        assert file.read() == b"first notes"
    async with open_data_file(second.file_path) as file:
        assert file.read() == b"second notes"


@pytest.mark.asyncio
async def test_readding_the_same_named_text_is_idempotent(storage_root):
    first = await save_data_to_file_detailed("same words", filename="Notes")
    second = await save_data_to_file_detailed("same words", filename="Notes")

    assert first.file_path == second.file_path
    assert len(_stored_files(storage_root)) == 1


@pytest.mark.asyncio
async def test_a_name_cannot_pick_the_storage_directory(storage_root):
    stored = await save_data_to_file_detailed("x", filename="../../etc/passwd")

    assert stored.file_path.endswith("/passwd.txt")
    assert stored.metadata["name"] == "passwd"
    assert all(path.is_relative_to(storage_root) for path in _stored_files(storage_root))


@pytest.mark.asyncio
async def test_an_unusable_name_falls_back_to_the_content_hash(storage_root):
    stored = await save_data_to_file_detailed("a plain note", filename="   ")

    assert stored.file_path.rsplit("/", 1)[-1].startswith("text_")
