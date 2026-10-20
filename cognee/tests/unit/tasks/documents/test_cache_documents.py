"""The caching pipeline's task: one CachedDocument per readable document (SDK-397)."""

import sys
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import (
    CACHED_DOCUMENT_MAX_CHARS,
    CachedDocument,
    cached_document_id,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.documents.cache_documents import build_cached_document, cache_documents

cache_module = sys.modules["cognee.tasks.documents.cache_documents"]


def _document(name="notes.txt", location="/tmp/notes.txt"):
    return TextDocument(
        name=name,
        raw_data_location=location,
        external_metadata=None,
        mime_type="text/plain",
    )


def test_the_cached_document_is_stable_and_carries_the_text():
    document = _document()
    entry = build_cached_document(document, "Alice met the rabbit.")

    assert entry.id == cached_document_id(document.id)
    assert build_cached_document(document, "other").id == entry.id, "re-caching upserts"
    assert entry.text == "Alice met the rabbit."
    assert entry.document_id == str(document.id)
    assert entry.name == "notes.txt" and entry.mime_type == "text/plain"
    assert entry.char_count == len("Alice met the rabbit.") and entry.truncated is False
    assert entry.type == "CachedDocument"
    # The text is a lexical field served from the vector store's keyword index,
    # not an embedded one, and leaves the graph node when that index exists.
    assert entry.metadata["index_fields"] == []
    assert entry.metadata["lexical_fields"] == ["text"]
    assert entry.metadata["vector_only_fields"] == ["text"]


def test_an_oversized_text_is_cut_and_flagged():
    entry = build_cached_document(_document(), "x" * (CACHED_DOCUMENT_MAX_CHARS + 5))
    assert entry.truncated is True
    assert len(entry.text) == CACHED_DOCUMENT_MAX_CHARS
    assert entry.char_count == CACHED_DOCUMENT_MAX_CHARS + 5


@pytest.mark.asyncio
@patch.object(cache_module, "add_data_points", new_callable=AsyncMock)
async def test_writes_one_entry_per_readable_document(mock_add):
    readable, unreadable, empty = (
        _document("a.txt", "/a"),
        _document("b.txt", "/b"),
        _document("c.txt", "/c"),
    )

    async def read(document):
        if document is unreadable:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "binary")
        if document is empty:
            return "   \n"
        return "readable text"

    ctx = PipelineContext()
    with patch.object(cache_module, "read_document_text", side_effect=read):
        cached = await cache_documents([readable, unreadable, empty], ctx=ctx)

    assert [entry.name for entry in cached] == ["a.txt"]
    assert isinstance(cached[0], CachedDocument)
    mock_add.assert_awaited_once()
    written, kwargs = mock_add.await_args.args[0], mock_add.await_args.kwargs
    assert written == cached
    assert kwargs["ctx"] is ctx
    assert kwargs["index_edges"] is False, "no structural edge is embedded by the cache write"


@pytest.mark.asyncio
@patch.object(cache_module, "add_data_points", new_callable=AsyncMock)
async def test_nothing_readable_writes_nothing(mock_add):
    not_utf8 = UnicodeDecodeError("utf-8", b"", 0, 1, "binary")
    with patch.object(cache_module, "read_document_text", AsyncMock(side_effect=not_utf8)):
        assert await cache_documents([_document()]) == []
    mock_add.assert_not_awaited()


@pytest.mark.asyncio
@patch.object(cache_module, "add_data_points", new_callable=AsyncMock)
async def test_an_io_error_fails_the_item_so_the_next_run_caches_it(mock_add):
    """Skipping would stamp the item cached, and nothing would ever cache it again."""
    io_error = AsyncMock(side_effect=OSError(5, "EIO"))
    with patch.object(cache_module, "read_document_text", io_error), pytest.raises(OSError):
        await cache_documents([_document()])
    mock_add.assert_not_awaited()


@pytest.mark.asyncio
@patch.object(cache_module, "add_data_points", new_callable=AsyncMock)
async def test_update_caches_the_new_text_without_reading_the_file(mock_add):
    """update() holds the edited text; a read could only fail after its chunks were written."""
    read = AsyncMock(side_effect=AssertionError("the file must not be read"))
    with patch.object(cache_module, "read_document_text", read):
        entry = await cache_module.cache_document_text(_document(), "the edited text")
        blank = await cache_module.cache_document_text(_document(), "  \n")

    assert entry.text == "the edited text"
    assert blank is None
    mock_add.assert_awaited_once()
    assert mock_add.await_args.args[0] == [entry]
