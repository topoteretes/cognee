"""summarize_text: source-chunk reference fields, and the eval-capture provenance
(SDK-529) it adds to every TextSummary only while capture is active.

``extract_summary_with_provenance`` is patched throughout — no LLM, no network.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.observability import capture
from cognee.modules.observability.capture import KIND_RUN_MANIFEST, KIND_SUMMARY_GENERATED
from cognee.tasks.summarization.models import TextSummary

summarize_text_module = importlib.import_module("cognee.tasks.summarization.summarize_text")

pytestmark = pytest.mark.usefixtures("capture_reset")

PROMPT = "Summarize the following content."
MODEL = "openai/gpt-test"


def _document():
    return TextDocument(
        name="notes.txt",
        raw_data_location="/tmp/notes.txt",
        external_metadata="",
        mime_type="text/plain",
    )


def _chunk(text="Chunk text", document=None, belongs_to_set=None, chunk_index=0):
    return DocumentChunk(
        text=text,
        chunk_size=len(text.split()),
        chunk_index=chunk_index,
        cut_type="sentence_end",
        is_part_of=document or _document(),
        contains=[],
        belongs_to_set=belongs_to_set,
    )


@pytest.fixture
def extract(monkeypatch):
    """Stand-in for extract_summary_with_provenance: (llm_output, prompt_text, model_name)."""
    mock = AsyncMock(
        side_effect=lambda content, response_model: (
            SimpleNamespace(summary=f"Summary of {content}"),
            PROMPT,
            MODEL,
        )
    )
    monkeypatch.setattr(summarize_text_module, "extract_summary_with_provenance", mock)
    return mock


def _summary_events(sink):
    return [record for record in sink.records if record["kind"] == KIND_SUMMARY_GENERATED]


@pytest.mark.asyncio
async def test_summarize_text_sets_source_chunk_reference_fields(monkeypatch, extract):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    chunk = _chunk(belongs_to_set=["KEEP"])

    summaries = await summarize_text_module.summarize_text([chunk], summarization_model=object)

    assert len(summaries) == 1
    assert summaries[0].source_chunk_id == str(chunk.id)
    assert summaries[0].belongs_to_set == ["KEEP"]
    assert summaries[0].made_from == chunk
    assert summaries[0].text == f"Summary of {chunk.text}"
    extract.assert_awaited_once_with(chunk.text, object)


@pytest.mark.asyncio
async def test_summarize_text_leaves_provenance_unset_when_capture_is_off(monkeypatch, extract):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    # A structural no-op when off: nothing is hashed, nothing is buffered.
    hashing = MagicMock(side_effect=capture.prompt_fingerprint)
    monkeypatch.setattr(capture, "prompt_fingerprint", hashing)

    [summary] = await summarize_text_module.summarize_text([_chunk()], summarization_model=object)

    assert capture.is_active() is False
    assert summary.text == f"Summary of {summary.made_from.text}"
    hashing.assert_not_called()
    assert not capture.hook._buffer


@pytest.mark.asyncio
async def test_summarize_text_populates_provenance_and_emits_one_event_per_chunk(
    fake_capture_sink, extract
):
    chunks = [_chunk("First chunk text", chunk_index=0), _chunk("Second chunk text", chunk_index=1)]

    summaries = await summarize_text_module.summarize_text(chunks, summarization_model=object)

    assert len(summaries) == 2
    for chunk, summary in zip(chunks, summaries):
        assert summary.text == f"Summary of {chunk.text}"
        # Provenance rides the event, never the persisted node.
        for field in ("model", "prompt_fingerprint", "source_text_hash"):
            assert not hasattr(summary, field)

    await capture.drain()

    events = _summary_events(fake_capture_sink)
    assert len(events) == 2
    for chunk, summary, event in zip(chunks, summaries, events):
        assert event["stage"] == "summarize_text"
        # Exact payload: ids, fingerprints and a size — never the chunk or summary text.
        assert event["payload"] == {
            "chunk_id": str(chunk.id),
            "summary_id": str(summary.id),
            "model": MODEL,
            "prompt_fingerprint": capture.prompt_fingerprint(PROMPT),
            "source_text_hash": capture.prompt_fingerprint(chunk.text),
            "summary_chars": len(summary.text),
        }


@pytest.mark.asyncio
async def test_summarize_text_notes_model_and_prompt_once_per_run(fake_capture_sink, extract):
    run_id, dataset_id = uuid4(), uuid4()
    chunks = [_chunk("a", chunk_index=0), _chunk("b", chunk_index=1), _chunk("c", chunk_index=2)]

    with capture.run_scope(run_id, dataset_id, kind="pipeline") as scope:
        await summarize_text_module.summarize_text(chunks, summarization_model=object)

    assert scope.fields["summarization.model"] == MODEL
    assert scope.fields["summarization.prompt_fingerprint"] == capture.prompt_fingerprint(PROMPT)

    await capture.drain()

    [manifest] = [r for r in fake_capture_sink.records if r["kind"] == KIND_RUN_MANIFEST]
    assert manifest["run_id"] == str(run_id)
    assert manifest["payload"]["summarization.model"] == MODEL
    assert manifest["payload"]["summarization.prompt_fingerprint"] == capture.prompt_fingerprint(
        PROMPT
    )
    # Every event is attributed to the run; still exactly one per chunk.
    events = _summary_events(fake_capture_sink)
    assert len(events) == 3
    assert {event["run_id"] for event in events} == {str(run_id)}
    assert {event["dataset_id"] for event in events} == {str(dataset_id)}


@pytest.mark.asyncio
async def test_summarize_text_returns_empty_input_without_touching_capture(
    fake_capture_sink, extract
):
    assert await summarize_text_module.summarize_text([], summarization_model=object) == []

    extract.assert_not_awaited()
    assert not capture.hook._buffer


def test_text_summary_node_carries_no_capture_provenance():
    """Graph content must not depend on whether capture happened to be on.

    The provenance the design partner needs is on the ``summary.generated`` event,
    keyed by ``summary_id``; the node schema stays free of telemetry fields.
    """
    summary = TextSummary(text="Short summary", made_from=_chunk())

    for name in ("model", "prompt_fingerprint", "source_text_hash"):
        assert name not in TextSummary.model_fields

    # The embedding index is unchanged: only the summary text is embedded.
    assert TextSummary.model_fields["metadata"].default == {"index_fields": ["text"]}
    assert summary.metadata["index_fields"] == ["text"]
    assert DataPoint.get_embeddable_property_names(summary) == ["text"]
    assert DataPoint.get_embeddable_properties(summary) == ["Short summary"]
