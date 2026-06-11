from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.utils.access_tracking import (
    _extract_access_node_ids,
    update_node_access_timestamps,
)


def test_extract_access_node_ids_from_hybrid_retrieval_dict():
    chunk = MagicMock()
    chunk.id = "chunk-result"
    chunk.payload = {"id": "chunk-1", "text": "Chunk"}

    node_ids = _extract_access_node_ids(
        {
            "chunks": [chunk, {"id": "chunk-2", "text": "Chunk 2"}],
            "entities": [
                {
                    "id": "entity-1",
                    "edges": [
                        {"source_id": "source-1", "target_id": "target-1"},
                        "not-an-edge",
                    ],
                },
                "not-an-entity",
            ],
        }
    )

    assert node_ids == ["chunk-1", "chunk-2", "entity-1", "source-1", "target-1"]


def test_extract_access_node_ids_ignores_unknown_shapes():
    assert _extract_access_node_ids({"unrelated": ["value"]}) == []
    assert _extract_access_node_ids(["not-an-edge"]) == []


@pytest.mark.asyncio
async def test_update_node_access_timestamps_handles_hybrid_retrieval_dict(monkeypatch):
    monkeypatch.setenv("ENABLE_LAST_ACCESSED", "true")
    graph_engine = MagicMock()

    with (
        patch(
            "cognee.modules.retrieval.utils.access_tracking.get_graph_engine",
            new_callable=AsyncMock,
            return_value=graph_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.access_tracking._find_origin_documents_via_projection",
            new_callable=AsyncMock,
            return_value=["doc-1"],
        ) as find_origin_documents,
        patch(
            "cognee.modules.retrieval.utils.access_tracking._update_sql_records",
            new_callable=AsyncMock,
        ) as update_sql_records,
    ):
        await update_node_access_timestamps(
            {
                "chunks": [{"id": "chunk-1", "text": "Chunk"}],
                "entities": [
                    {
                        "id": "entity-1",
                        "edges": [{"source_id": "source-1", "target_id": "target-1"}],
                    }
                ],
            }
        )

    find_origin_documents.assert_awaited_once_with(
        graph_engine,
        ["chunk-1", "entity-1", "source-1", "target-1"],
    )
    update_sql_records.assert_awaited_once()
    assert update_sql_records.await_args.args[0] == ["doc-1"]
    assert isinstance(update_sql_records.await_args.args[1], datetime)
