"""TRIPLET_EMBEDDING was removed (SDK-1115): a stale setting warns once and writes nothing."""

import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.cognify.config import CognifyConfig, get_cognify_config
from cognee.modules.engine.models import Triplet
from cognee.modules.improve import stage_names
from cognee.tasks.storage.add_data_points import add_data_points

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
config_module = importlib.import_module("cognee.modules.cognify.config")
adp_module = sys.modules["cognee.tasks.storage.add_data_points"]


class _Point(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
async def test_set_triplet_embedding_warns_once_and_leaves_the_writer_unchanged(monkeypatch):
    monkeypatch.delenv("CONTRADICTION_DETECTION", raising=False)
    monkeypatch.setenv("TRIPLET_EMBEDDING", "true")
    get_cognify_config.cache_clear()
    try:
        with patch.object(config_module.logger, "warning") as warning:
            config = get_cognify_config()
            assert get_cognify_config() is config
            warning.assert_called_once()
            assert "TRIPLET_EMBEDDING is ignored" in warning.call_args.args[0]

        assert not hasattr(config, "triplet_embedding")
        assert "triplet_embedding" not in config.to_dict()
        with patch.object(cognify_module, "get_cognify_config", return_value=config):
            tasks = await cognify_module.get_default_tasks(
                config={"ontology_config": {"ontology_resolver": None}}, chunk_size=1024
            )
        [store] = [task for task in tasks if task.executable is add_data_points]
        assert "embed_triplets" not in store.default_params["kwargs"]
    finally:
        get_cognify_config.cache_clear()


def test_unset_triplet_embedding_does_not_warn(monkeypatch):
    monkeypatch.delenv("TRIPLET_EMBEDDING", raising=False)
    monkeypatch.delenv("CONTRADICTION_DETECTION", raising=False)
    get_cognify_config.cache_clear()
    try:
        with patch.object(config_module.logger, "warning") as warning:
            get_cognify_config()
        warning.assert_not_called()
    finally:
        get_cognify_config.cache_clear()


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_deprecated_embed_triplets_warns_and_indexes_no_triplets(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    """embed_triplets=True still runs, but writes the nodes and edges only."""
    source, target = _Point(text="source"), _Point(text="target")
    edge = (str(source.id), str(target.id), "relates", {"edge_text": "describes"})
    mock_get_graph.side_effect = [([source], [edge]), ([target], [])]
    mock_dedup.side_effect = lambda nodes, edges: (nodes, edges)
    unified = AsyncMock()
    unified.vector = MagicMock()
    unified.has_capability = MagicMock(return_value=False)
    unified.graph.is_empty = AsyncMock(return_value=False)
    unified.graph.get_graph_metadata = AsyncMock(return_value={})
    mock_get_unified.return_value = unified

    with pytest.warns(FutureWarning, match="embed_triplets"):
        await add_data_points([source, target], embed_triplets=True)

    mock_index_nodes.assert_awaited_once()
    indexed = mock_index_nodes.await_args.args[0]
    assert indexed == [source, target]
    assert not any(isinstance(point, Triplet) for point in indexed)
    mock_index_edges.assert_awaited_once()


def test_no_triplet_writer_is_left():
    assert "triplet_embedding" not in CognifyConfig.model_fields
    assert "triplet_enrichment" not in stage_names()


def test_add_data_points_keeps_its_positional_order():
    """embed_triplets keeps its old third place for the deprecation release, so a
    positional ``add_data_points(dps, edges, False, ctx)`` still binds ctx."""
    import inspect

    assert list(inspect.signature(add_data_points).parameters)[:5] == [
        "data_points",
        "custom_edges",
        "embed_triplets",
        "ctx",
        "graph_only",
    ]
