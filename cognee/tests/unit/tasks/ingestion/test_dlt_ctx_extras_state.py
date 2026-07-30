"""Cross-batch dedup state for extract_dlt_source_edges lives in ctx.extras.

The state used to be mutable ``set()`` kwargs baked into the Task object,
which the old cognify executor shared across datasets — dataset B skipped schema
emission because dataset A had already "emitted" it, writing edges to nodes
that only exist in A's graph. ctx.extras is per data item (one manifest item
IS one source), so isolation holds by construction. These tests pin that.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import cognee.tasks.ingestion.extract_dlt_source_edges as edges_module
from cognee.modules.data.processing.document_types import DltSourceDocument
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.tasks.ingestion.extract_dlt_source_edges import extract_dlt_source_edges


def _source_doc():
    doc = MagicMock(spec=DltSourceDocument)
    doc.id = uuid4()
    doc.raw_data_location = "unused://manifest.json"
    return doc


def _manifest(row_id):
    return {
        "tables": {
            "people": {"schema_info": {"columns": {}}, "foreign_keys": [], "dlt_db_name": "db"}
        },
        "rows": [
            {
                "node_id": str(row_id),
                "table_name": "people",
                "fk_references": [],
                "column_values": {},
            }
        ],
    }


def _batch(doc, row_id):
    return [SimpleNamespace(id=row_id, is_part_of=doc)]


@pytest.fixture
def emit_mock(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(edges_module, "emit_dlt_schema_graph", mock)
    return mock


@pytest.mark.asyncio
async def test_two_contexts_do_not_share_dedup_state(monkeypatch, emit_mock):
    """Two data items (two sources/datasets) must both emit their schema."""
    doc = _source_doc()
    row_id = uuid4()
    monkeypatch.setattr(
        edges_module, "load_dlt_manifest", AsyncMock(return_value=_manifest(row_id))
    )

    ctx_a, ctx_b = PipelineContext(), PipelineContext()
    await extract_dlt_source_edges(_batch(doc, row_id), ctx=ctx_a)
    await extract_dlt_source_edges(_batch(doc, row_id), ctx=ctx_b)

    # Both calls emitted the schema tables — B was not suppressed by A's state.
    first_tables = emit_mock.await_args_list[0].args[0]
    second_tables = emit_mock.await_args_list[1].args[0]
    assert "people" in first_tables
    assert "people" in second_tables

    # And the dedup sets are distinct objects per context.
    assert ctx_a.extras["dlt_emitted_schema_docs"] is not ctx_b.extras["dlt_emitted_schema_docs"]
    assert (
        ctx_a.extras["dlt_emitted_value_node_ids"] is not ctx_b.extras["dlt_emitted_value_node_ids"]
    )


@pytest.mark.asyncio
async def test_same_context_dedups_schema_across_batches(monkeypatch, emit_mock):
    """Within one data item's run, schema is emitted for the first batch only."""
    doc = _source_doc()
    row_a, row_b = uuid4(), uuid4()
    manifest = _manifest(row_a)
    manifest["rows"].append(
        {"node_id": str(row_b), "table_name": "people", "fk_references": [], "column_values": {}}
    )
    monkeypatch.setattr(edges_module, "load_dlt_manifest", AsyncMock(return_value=manifest))

    ctx = PipelineContext()
    await extract_dlt_source_edges(_batch(doc, row_a), ctx=ctx)
    await extract_dlt_source_edges(_batch(doc, row_b), ctx=ctx)

    assert "people" in emit_mock.await_args_list[0].args[0]
    assert emit_mock.await_args_list[1].args[0] == {}  # schema deduped
    # Row records still emitted for the second batch.
    second_rows = emit_mock.await_args_list[1].args[1]
    assert [record["source_id"] for record in second_rows] == [str(row_b)]
    assert str(doc.id) in ctx.extras["dlt_emitted_schema_docs"]


@pytest.mark.asyncio
async def test_ctx_none_direct_call_uses_fresh_state(monkeypatch, emit_mock):
    """Direct invocation without a pipeline context works, with per-call sets."""
    doc = _source_doc()
    row_id = uuid4()
    monkeypatch.setattr(
        edges_module, "load_dlt_manifest", AsyncMock(return_value=_manifest(row_id))
    )

    await extract_dlt_source_edges(_batch(doc, row_id), ctx=None)
    await extract_dlt_source_edges(_batch(doc, row_id), ctx=None)

    # No cross-call state: both calls emit the schema.
    assert "people" in emit_mock.await_args_list[0].args[0]
    assert "people" in emit_mock.await_args_list[1].args[0]


@pytest.mark.asyncio
async def test_relational_rows_index_into_their_own_collection(monkeypatch):
    """Rows are their own type AND their own collection: chunk search
    (DocumentChunk_text) is documents-only; row text lands in
    RelationalRow_text for row-aware retrieval. The pipeline rebuilds
    data-point classes (copy_model keeps only the class NAME and fields),
    so the collection name must survive that rebuild too."""
    from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
    from cognee.modules.chunking.models.RelationalRow import RelationalRow
    from cognee.modules.data.processing.document_types import DltSourceDocument
    from cognee.modules.storage.utils import copy_model
    from cognee.tasks.storage.index_data_points import index_data_points as run_indexing

    chunk = RelationalRow(
        id=uuid4(),
        text="row text",
        chunk_size=2,
        chunk_index=0,
        cut_type="dlt_row",
        is_part_of=DltSourceDocument(
            name="src", raw_data_location="unused", external_metadata="{}"
        ),
        contains=[],
    )
    assert isinstance(chunk, DocumentChunk)  # field-shape reuse only
    assert type(chunk).__name__ == "RelationalRow"

    rebuilt = copy_model(type(chunk))(**chunk.model_dump())

    created = []

    class _FakeEngine:
        class embedding_engine:
            @staticmethod
            def get_batch_size():
                return 10

        @staticmethod
        async def create_vector_index(type_name, field_name):
            created.append((type_name, field_name))

        @staticmethod
        async def index_data_points(type_name, field_name, points):
            return None

    await run_indexing([chunk], vector_engine=_FakeEngine())
    await run_indexing([rebuilt], vector_engine=_FakeEngine())
    assert created == [("RelationalRow", "text"), ("RelationalRow", "text")], (
        "row vectors must land in RelationalRow_text, original and rebuilt alike"
    )
