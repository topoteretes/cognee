from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import importlib

import pytest

from cognee.modules.pipelines.models import PipelineContext

chunk_module = importlib.import_module("cognee.tasks.chunks.create_chunk_associations")


@pytest.mark.asyncio
async def test_create_chunk_associations_folds_provenance(monkeypatch):
    graph = SimpleNamespace(add_edges=AsyncMock())
    vector = SimpleNamespace(
        search=AsyncMock(
            side_effect=[
                [SimpleNamespace(id="chunk-a")],
                [SimpleNamespace(id="chunk-b")],
                [SimpleNamespace(id="chunk-a"), SimpleNamespace(id="chunk-b")],
                [SimpleNamespace(id="chunk-b"), SimpleNamespace(id="chunk-a")],
            ]
        )
    )
    ctx = PipelineContext(
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )

    monkeypatch.setattr(chunk_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(chunk_module, "get_vector_engine_async", AsyncMock(return_value=vector))
    monkeypatch.setattr(
        chunk_module,
        "_compare_chunks",
        AsyncMock(
            return_value=chunk_module.ChunkSimilarity(
                are_similar=True,
                similarity_score=0.9,
                reasoning="related",
                association_type="topical",
            )
        ),
    )
    monkeypatch.setattr(
        chunk_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": "source-ref", "pipeline_run_id": "run-id"}),
    )
    monkeypatch.setattr(chunk_module, "index_graph_edges", AsyncMock())

    result = [
        item
        async for item in chunk_module.create_chunk_associations(
            ["climate impacts ecosystems", "weather events are connected"],
            similarity_threshold=0.5,
            ctx=ctx,
        )
    ]

    assert result == ["climate impacts ecosystems", "weather events are connected"]
    graph.add_edges.assert_awaited_once()
    assert graph.add_edges.await_args.kwargs == {
        "source_ref_key": "source-ref",
        "pipeline_run_id": "run-id",
    }
