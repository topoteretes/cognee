"""Unit tests for the cognee-backed adapter (issue #3609, commit 2).

Every cognee call (add / cognify / search / forget) is mocked — real cognify is
expensive and needs API keys, so it is never run here. No Slack, no keys.

The critical test is the citation round-trip: it mocks ``cognee.search`` to
return chunk payloads shaped EXACTLY like cognee's real ``search(CHUNKS)`` output
(a flat ``list[dict]`` of DocumentChunk payloads carrying ``document_id``) and
asserts the resulting ``Answer`` citations carry the correct channel/ts/permalink
joined back from that ``document_id``.
"""

import asyncio
from unittest.mock import AsyncMock

import cognee
import pytest

from src.cognee_memory import (
    CitationStore,
    CogneeChatMemory,
    _first_text,
    _normalize_chunk_payloads,
)
from src.memory_adapter import Answer, ConversationRef, message_data_id

REF = ConversationRef(team_id="T1", channel_id="C42")


@pytest.fixture
def index():
    return CitationStore()


@pytest.fixture
def memory(index):
    return CogneeChatMemory(index, top_k=5)


def _real_chunk_payload(document_id: str, text: str, chunk_index: int = 0) -> dict:
    """A dict shaped like a real search(SearchType.CHUNKS) result item.

    Keys mirror the DocumentChunk vector payload kept by LanceDB
    (``_build_data_point_schema`` keeps scalar fields + id + belongs_to_set,
    drops ``metadata`` and nested models). ``document_id`` is the join key.
    """
    return {
        "id": f"chunk-{document_id}-{chunk_index}",
        "text": text,
        "chunk_size": len(text),
        "chunk_index": chunk_index,
        "cut_type": "sentence_end",
        "document_id": document_id,
        "document_name": "slack_message.txt",
        "belongs_to_set": ["C42"],
    }


# --------------------------------------------------------------------------- #
# ingest                                                                      #
# --------------------------------------------------------------------------- #


def test_ingest_adds_with_deterministic_id_and_records_citation(memory, index, monkeypatch):
    add_mock = AsyncMock()
    monkeypatch.setattr(cognee, "add", add_mock)

    asyncio.run(
        memory.ingest(
            REF,
            ts="1700000000.000100",
            text="We decided to ship on Friday.",
            permalink="https://slack.example/archives/C42/p1700000000000100",
            author="alice",
        )
    )

    add_mock.assert_awaited_once()
    args, kwargs = add_mock.await_args
    data_item = args[0]
    expected_id = message_data_id("C42", "1700000000.000100")
    assert data_item.data == "We decided to ship on Friday."
    assert data_item.data_id == expected_id
    assert kwargs["dataset_name"] == "slack_C42"
    assert kwargs["node_set"] == ["C42"]

    record = index.get(str(expected_id))
    assert record is not None
    assert record.channel_id == "C42"
    assert record.ts == "1700000000.000100"
    assert record.permalink == "https://slack.example/archives/C42/p1700000000000100"
    assert record.author == "alice"
    assert record.snippet == "We decided to ship on Friday."


def test_ingest_does_not_cognify(memory, monkeypatch):
    add_mock = AsyncMock()
    cognify_mock = AsyncMock()
    monkeypatch.setattr(cognee, "add", add_mock)
    monkeypatch.setattr(cognee, "cognify", cognify_mock)

    asyncio.run(memory.ingest(REF, ts="1.0", text="hi", permalink="https://x", author="bob"))

    cognify_mock.assert_not_awaited()


# --------------------------------------------------------------------------- #
# flush                                                                       #
# --------------------------------------------------------------------------- #


def test_flush_triggers_cognify_for_channel_dataset(memory, monkeypatch):
    cognify_mock = AsyncMock()
    monkeypatch.setattr(cognee, "cognify", cognify_mock)

    asyncio.run(memory.flush(REF))

    cognify_mock.assert_awaited_once()
    _, kwargs = cognify_mock.await_args
    assert kwargs["datasets"] == ["slack_C42"]


# --------------------------------------------------------------------------- #
# answer — the critical citation round-trip                                   #
# --------------------------------------------------------------------------- #


def _patch_search(monkeypatch, *, prose, chunks):
    def fake_search(query_text, *, query_type, **kwargs):
        if query_type == cognee.SearchType.GRAPH_COMPLETION:
            return prose
        if query_type == cognee.SearchType.CHUNKS:
            return chunks
        raise AssertionError(f"unexpected query_type {query_type}")

    search_mock = AsyncMock(side_effect=fake_search)
    monkeypatch.setattr(cognee, "search", search_mock)
    return search_mock


def test_answer_round_trip_maps_chunks_to_citations(memory, index, monkeypatch):
    # Ingest a real message so the side index holds its citation row.
    add_mock = AsyncMock()
    monkeypatch.setattr(cognee, "add", add_mock)
    ts = "1700000000.000100"
    permalink = "https://slack.example/archives/C42/p1700000000000100"
    asyncio.run(
        memory.ingest(
            REF, ts=ts, text="We decided to ship on Friday.", permalink=permalink, author="alice"
        )
    )
    document_id = str(message_data_id("C42", ts))

    # Mock search: prose from GRAPH_COMPLETION, real-shaped chunk from CHUNKS.
    search_mock = _patch_search(
        monkeypatch,
        prose=["The team decided to ship on Friday."],
        chunks=[_real_chunk_payload(document_id, "We decided to ship on Friday.")],
    )

    answer = asyncio.run(memory.answer(REF, query="what did we decide?"))

    assert isinstance(answer, Answer)
    assert answer.text == "The team decided to ship on Friday."
    assert len(answer.citations) == 1
    cite = answer.citations[0]
    assert cite.ok is True
    assert cite.permalink == permalink
    assert cite.channel_id == "C42"
    assert cite.ts == ts
    assert cite.author == "alice"

    # Confirm the two searches used the right types / scope.
    assert search_mock.await_count == 2
    query_types = {call.kwargs["query_type"] for call in search_mock.await_args_list}
    assert query_types == {cognee.SearchType.GRAPH_COMPLETION, cognee.SearchType.CHUNKS}
    chunks_call = next(
        c for c in search_mock.await_args_list if c.kwargs["query_type"] == cognee.SearchType.CHUNKS
    )
    assert chunks_call.kwargs["datasets"] == ["slack_C42"]
    assert chunks_call.kwargs["node_name"] == ["C42"]


def test_answer_dedupes_multiple_chunks_from_one_message(memory, index, monkeypatch):
    ts = "1700000000.000100"
    document_id = str(message_data_id("C42", ts))
    index.put(
        document_id,
        channel_id="C42",
        ts=ts,
        permalink="https://slack.example/x",
        author="alice",
        snippet="msg",
    )

    _patch_search(
        monkeypatch,
        prose=["answer"],
        chunks=[
            _real_chunk_payload(document_id, "chunk one", chunk_index=0),
            _real_chunk_payload(document_id, "chunk two", chunk_index=1),
        ],
    )

    answer = asyncio.run(memory.answer(REF, query="q"))

    assert len(answer.citations) == 1
    assert answer.citations[0].permalink == "https://slack.example/x"


def test_answer_missing_permalink_falls_back_gracefully(memory, monkeypatch):
    # No index row for this document_id → must not crash, must degrade to text.
    unknown_document_id = str(message_data_id("C42", "9999.0001"))
    _patch_search(
        monkeypatch,
        prose=["here is what I found"],
        chunks=[_real_chunk_payload(unknown_document_id, "orphan chunk text")],
    )

    answer = asyncio.run(memory.answer(REF, query="q"))

    assert answer.text == "here is what I found"
    assert len(answer.citations) == 1
    cite = answer.citations[0]
    assert cite.ok is False
    assert cite.permalink == ""
    assert cite.snippet == "orphan chunk text"


def test_answer_blank_permalink_in_index_falls_back(memory, index, monkeypatch):
    ts = "1700000000.000100"
    document_id = str(message_data_id("C42", ts))
    index.put(
        document_id,
        channel_id="C42",
        ts=ts,
        permalink="",  # stale/blank permalink
        author="alice",
        snippet="stored snippet",
    )
    _patch_search(
        monkeypatch,
        prose=["a"],
        chunks=[_real_chunk_payload(document_id, "chunk text")],
    )

    answer = asyncio.run(memory.answer(REF, query="q"))

    cite = answer.citations[0]
    assert cite.ok is False
    assert cite.permalink == ""
    assert cite.snippet == "stored snippet"


def test_answer_returns_empty_when_channel_has_no_data(memory, monkeypatch):
    # Fresh/empty channel: cognee.search raises DatasetNotFoundError. answer() must
    # degrade to an empty Answer (the renderer shows a calm "no memory yet" reply)
    # rather than propagate and leave the user with no response.
    from cognee.modules.data.exceptions import DatasetNotFoundError

    monkeypatch.setattr(cognee, "search", AsyncMock(side_effect=DatasetNotFoundError()))

    answer = asyncio.run(memory.answer(REF, query="what did we decide?"))

    assert isinstance(answer, Answer)
    assert answer.text == ""
    assert answer.citations == []


def test_answer_handles_access_control_wrapper_shape(memory, index, monkeypatch):
    # Prove _first_text / _normalize handle the ENABLE_BACKEND_ACCESS_CONTROL shape:
    # search returns [{"dataset_id":..., "search_result": <result>}].
    ts = "1700000000.000100"
    document_id = str(message_data_id("C42", ts))
    index.put(
        document_id,
        channel_id="C42",
        ts=ts,
        permalink="https://slack.example/x",
        author="alice",
        snippet="msg",
    )
    _patch_search(
        monkeypatch,
        prose=[
            {"dataset_id": "d1", "dataset_name": "slack_C42", "search_result": "wrapped answer"}
        ],
        chunks=[
            {
                "dataset_id": "d1",
                "dataset_name": "slack_C42",
                "search_result": [_real_chunk_payload(document_id, "chunk text")],
            }
        ],
    )

    answer = asyncio.run(memory.answer(REF, query="q"))

    assert answer.text == "wrapped answer"
    assert len(answer.citations) == 1
    assert answer.citations[0].permalink == "https://slack.example/x"


# --------------------------------------------------------------------------- #
# forget                                                                      #
# --------------------------------------------------------------------------- #


def test_forget_deletes_channel_dataset_and_citations(memory, index, monkeypatch):
    document_id = str(message_data_id("C42", "1.0"))
    index.put(
        document_id,
        channel_id="C42",
        ts="1.0",
        permalink="https://slack.example/x",
        author="alice",
        snippet="msg",
    )
    forget_mock = AsyncMock()
    monkeypatch.setattr(cognee, "forget", forget_mock)

    asyncio.run(memory.forget(REF))

    forget_mock.assert_awaited_once()
    _, kwargs = forget_mock.await_args
    assert kwargs["dataset"] == "slack_C42"
    assert index.get(document_id) is None


def test_forget_is_idempotent_for_never_used_channel(memory, index, monkeypatch):
    # cognee resolves an unknown dataset name to None and dereferences it, raising
    # AttributeError. forget() must swallow it and still clear local citation rows.
    document_id = str(message_data_id("C42", "1.0"))
    index.put(
        document_id,
        channel_id="C42",
        ts="1.0",
        permalink="https://slack.example/x",
        author="alice",
        snippet="msg",
    )
    monkeypatch.setattr(
        cognee, "forget", AsyncMock(side_effect=AttributeError("'NoneType' has no attribute 'id'"))
    )

    asyncio.run(memory.forget(REF))  # must not raise

    assert index.get(document_id) is None


# --------------------------------------------------------------------------- #
# shape helpers (direct)                                                       #
# --------------------------------------------------------------------------- #


def test_normalize_chunk_payloads_flat_and_wrapped():
    flat = [{"document_id": "a"}, {"document_id": "b"}]
    assert _normalize_chunk_payloads(flat) == flat

    wrapped = [{"dataset_id": "d1", "search_result": [{"document_id": "a"}]}]
    assert _normalize_chunk_payloads(wrapped) == [{"document_id": "a"}]

    assert _normalize_chunk_payloads(None) == []


def test_first_text_variants():
    assert _first_text(["hello"]) == "hello"
    assert _first_text("hello") == "hello"
    assert _first_text([{"search_result": "wrapped"}]) == "wrapped"
    assert _first_text([]) == ""
