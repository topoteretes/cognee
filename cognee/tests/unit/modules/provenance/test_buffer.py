from types import SimpleNamespace
from uuid import UUID, uuid4

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.provenance.buffer import ProvenanceBuffer


def _context():
    return PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
        pipeline_name="cognify_pipeline",
    )


def _chunk(ctx, source_id, destination_id):
    document = Document(
        id=ctx.data_item.id,
        name="report.txt",
        raw_data_location="file:///tmp/report.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunk = DocumentChunk(
        text="Alice works at Acme. " * 20,
        chunk_size=80,
        chunk_index=0,
        cut_type="paragraph",
        is_part_of=document,
        contains=[],
        document_id=str(document.id),
        document_name=document.name,
    )
    chunk._provenance_edges.append(EdgeIdentity(str(source_id), str(destination_id), "works_at"))
    return chunk


def test_buffer_captures_extracted_and_structural_edge_evidence():
    ctx = _context()
    source_id = uuid4()
    destination_id = uuid4()
    chunk = _chunk(ctx, source_id, destination_id)
    mentioned_id = uuid4()
    structural_edge = (
        str(chunk.id),
        str(mentioned_id),
        "contains",
        {"edge_text": "chunk mentions Alice"},
    )
    buffer = ProvenanceBuffer()

    captured = buffer.capture(chunks=[chunk], graph_edges=[structural_edge], ctx=ctx)

    assert captured == 2
    assert len(buffer.evidence_rows) == 2
    assert {row["evidence_kind"] for row in buffer.evidence_rows.values()} == {
        "extracted",
        "structural",
    }
    assert {row["chunk_id"] for row in buffer.evidence_rows.values()} == {chunk.id}
    assert {row["data_id"] for row in buffer.evidence_rows.values()} == {ctx.data_item.id}
    assert chunk.text not in str(buffer.evidence_rows)

    expected_edge_id = UUID(generate_edge_object_id(source_id, destination_id, "works_at"))
    assert expected_edge_id in {row["edge_id"] for row in buffer.evidence_rows.values()}


def test_buffer_deduplicates_retries_within_the_same_run():
    ctx = _context()
    chunk = _chunk(ctx, uuid4(), uuid4())
    buffer = ProvenanceBuffer()

    assert buffer.capture(chunks=[chunk], graph_edges=[], ctx=ctx) == 1
    assert buffer.capture(chunks=[chunk], graph_edges=[], ctx=ctx) == 0
    assert len(buffer.evidence_rows) == 1


def test_same_graph_assertion_keeps_support_from_each_chunk():
    ctx = _context()
    source_id = uuid4()
    destination_id = uuid4()
    first = _chunk(ctx, source_id, destination_id)
    second = _chunk(ctx, source_id, destination_id)
    second.chunk_index = 1
    buffer = ProvenanceBuffer()

    assert buffer.capture(chunks=[first, second], graph_edges=[], ctx=ctx) == 2
    assert {row["chunk_id"] for row in buffer.evidence_rows.values()} == {
        first.id,
        second.id,
    }


def test_mark_persisted_releases_buffered_rows():
    ctx = _context()
    chunk = _chunk(ctx, uuid4(), uuid4())
    buffer = ProvenanceBuffer()
    buffer.capture(chunks=[chunk], graph_edges=[], ctx=ctx)
    batch = buffer.snapshot()

    buffer.mark_persisted(batch)

    assert not buffer.evidence_rows
