from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.modules.provenance.edge_evidence import capture
from cognee.tasks.summarization.models import TextSummary


def _ctx():
    return SimpleNamespace(
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_run_id=uuid4(),
        provenance_buffer=None,
    )


def _chunk(document_id):
    document = Document(
        id=document_id,
        name="report.txt",
        raw_data_location="file:///tmp/report.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunk = DocumentChunk(
        text="Alice works at Acme.",
        chunk_size=20,
        chunk_index=0,
        cut_type="paragraph",
        is_part_of=document,
        contains=[],
        document_id=str(document.id),
        document_name=document.name,
    )
    chunk._provenance_edges.append(EdgeIdentity(str(uuid4()), str(uuid4()), "works_at"))
    return chunk


def _enabled(monkeypatch):
    monkeypatch.setattr(
        capture,
        "get_provenance_config",
        lambda: SimpleNamespace(edge_evidence_enabled=True, edge_evidence_flush_threshold=10_000),
    )


@pytest.mark.asyncio
async def test_disabled_capture_is_zero_cost(monkeypatch):
    monkeypatch.setattr(
        capture,
        "get_provenance_config",
        lambda: SimpleNamespace(
            edge_evidence_enabled=False,
            edge_evidence_flush_threshold=10_000,
        ),
    )
    ctx = SimpleNamespace(provenance_buffer=None)

    assert await capture.capture_graph_provenance([object()], [], ctx) == 0
    assert ctx.provenance_buffer is None


def test_env_flag_names_are_edge_evidence_specific(monkeypatch):
    """The flags must not collide with PROVENANCE_TRACKING / COGNEE_PROVENANCE_MODE."""
    from cognee.modules.provenance.edge_evidence.config import ProvenanceConfig

    monkeypatch.setenv("EDGE_EVIDENCE_ENABLED", "false")
    monkeypatch.setenv("EDGE_EVIDENCE_FLUSH_THRESHOLD", "250")
    config = ProvenanceConfig()

    assert config.edge_evidence_enabled is False
    assert config.edge_evidence_flush_threshold == 250


def test_collect_document_chunks_finds_chunks_nested_under_summaries():
    """cognify's default batch is TextSummary objects; their chunks sit under made_from."""
    ctx = _ctx()
    chunk = _chunk(ctx.data_item.id)
    summary = TextSummary(text="s", made_from=chunk)

    found = capture.collect_document_chunks([summary])

    assert found == [chunk]
    assert found[0]._provenance_edges == chunk._provenance_edges


def test_collect_document_chunks_dedupes_shared_chunks_and_ignores_non_datapoints():
    ctx = _ctx()
    chunk = _chunk(ctx.data_item.id)
    summaries = [TextSummary(text="a", made_from=chunk), TextSummary(text="b", made_from=chunk)]

    assert capture.collect_document_chunks([*summaries, chunk, object(), "text"]) == [chunk]


@pytest.mark.asyncio
async def test_nested_chunk_evidence_is_captured(monkeypatch):
    """Regression: with the merged extract+summarize task, no chunk is top-level."""
    _enabled(monkeypatch)
    ctx = _ctx()
    chunk = _chunk(ctx.data_item.id)

    captured = await capture.capture_graph_provenance(
        [TextSummary(text="s", made_from=chunk)], graph_edges=[], ctx=ctx
    )

    assert captured == 1
    (row,) = ctx.provenance_buffer.snapshot().evidence_rows
    assert row["chunk_id"] == chunk.id
    assert row["relationship_name"] == "works_at"
    assert row["evidence_kind"] == "extracted"
