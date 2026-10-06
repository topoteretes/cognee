"""legacy_delete still clears vectors of DataPoint models removed from code (SDK-981)."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

legacy_delete_module = importlib.import_module("cognee.modules.graph.methods.legacy_delete")


class _VectorEngine:
    def __init__(self, existing: set[str]):
        self.existing = existing
        self.deleted: dict[str, list[str]] = {}

    async def has_collection(self, name: str) -> bool:
        return name in self.existing

    async def delete_data_points(self, name: str, ids: list[str]) -> None:
        self.deleted[name] = ids


@pytest.mark.asyncio
async def test_event_vectors_left_by_the_removed_temporal_pipeline_are_deleted():
    node_id = uuid4()
    vector_engine = _VectorEngine({"Event_name", "Entity_name"})

    with (
        patch.object(
            legacy_delete_module,
            "delete_document_subgraph",
            AsyncMock(return_value=[node_id]),
        ),
        patch.object(
            legacy_delete_module,
            "get_vector_engine_async",
            AsyncMock(return_value=vector_engine),
        ),
    ):
        await legacy_delete_module.legacy_delete(SimpleNamespace(id=uuid4()))

    assert vector_engine.deleted["Event_name"] == [str(node_id)]
    assert vector_engine.deleted["Entity_name"] == [str(node_id)]
