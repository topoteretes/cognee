"""The per-adapter capability probe (plan Part 5.6)."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.improve import GraphCapabilities, probe_graph_capabilities
from cognee.modules.improve import capabilities as capabilities_mod


def _concrete(cls):
    """Let an incomplete GraphDBInterface subclass be instantiated in tests."""
    cls.__abstractmethods__ = frozenset()
    return cls


@_concrete
class StubAdapter(GraphDBInterface):
    """Overrides nothing beyond the interface: every stub raises NotImplementedError."""


@_concrete
class FeedbackOnlyAdapter(GraphDBInterface):
    async def set_node_feedback_weights(self, node_feedback_weights: Dict[str, float]):
        return {}

    async def set_edge_feedback_weights(self, edge_feedback_weights: Dict[str, float]):
        return {}


@_concrete
class HalfFeedbackAdapter(GraphDBInterface):
    async def set_node_feedback_weights(self, node_feedback_weights: Dict[str, float]):
        return {}


@_concrete
class FullAdapter(GraphDBInterface):
    supports_incremental_chunk_updates = True

    async def set_node_feedback_weights(self, node_feedback_weights):
        return {}

    async def set_edge_feedback_weights(self, edge_feedback_weights):
        return {}

    async def get_node_truth_state(self, node_ids: List[str]):
        return {}

    async def set_node_truth_state(self, node_truth_state: Dict[str, Dict[str, Any]]):
        return {}


@_concrete
class ExplicitFlagAdapter(GraphDBInterface):
    # Declares the answer instead of overriding the methods.
    supports_feedback_weights = True
    supports_truth_state = False

    async def get_node_truth_state(self, node_ids):
        return {}

    async def set_node_truth_state(self, node_truth_state):
        return {}


def test_interface_stubs_do_not_count_as_support():
    caps = probe_graph_capabilities(StubAdapter())
    assert caps.supports_feedback_weights is False
    assert caps.supports_truth_state is False
    assert caps.supports_incremental_chunk_updates is False
    assert caps.adapter == "StubAdapter"


def test_overriding_both_feedback_setters_gives_feedback_weights():
    caps = probe_graph_capabilities(FeedbackOnlyAdapter())
    assert caps.supports_feedback_weights is True
    assert caps.supports_truth_state is False


def test_overriding_only_one_feedback_setter_is_not_enough():
    assert probe_graph_capabilities(HalfFeedbackAdapter()).supports_feedback_weights is False


def test_full_adapter_supports_everything():
    caps = probe_graph_capabilities(FullAdapter())
    assert caps == GraphCapabilities(
        supports_feedback_weights=True,
        supports_truth_state=True,
        supports_incremental_chunk_updates=True,
        adapter="FullAdapter",
    )


def test_explicit_class_attribute_wins_over_method_inspection():
    caps = probe_graph_capabilities(ExplicitFlagAdapter())
    assert caps.supports_feedback_weights is True  # declared, no methods
    assert caps.supports_truth_state is False  # declared False despite overriding methods


def test_duck_typed_engine_counts_callables():
    class Community:
        async def set_node_feedback_weights(self, weights):
            return {}

        async def set_edge_feedback_weights(self, weights):
            return {}

    caps = probe_graph_capabilities(Community())
    assert caps.supports_feedback_weights is True
    assert caps.supports_truth_state is False


def test_ladybug_adapter_reports_feedback_weights_and_truth_state():
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    engine = LadybugAdapter.__new__(LadybugAdapter)  # class-level probe, no database
    caps = probe_graph_capabilities(engine)
    assert caps.supports_feedback_weights is True
    assert caps.supports_truth_state is True
    assert caps.supports_incremental_chunk_updates is True


def test_neo4j_adapter_reports_feedback_weights_but_no_truth_state():
    pytest.importorskip("neo4j")
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    caps = probe_graph_capabilities(Neo4jAdapter.__new__(Neo4jAdapter))
    assert caps.supports_feedback_weights is True
    assert caps.supports_truth_state is False


def test_capabilities_are_frozen():
    caps = GraphCapabilities.assume_supported()
    with pytest.raises(Exception):
        caps.supports_truth_state = False  # type: ignore[misc]


@pytest.mark.asyncio
async def test_resolve_probes_inside_the_dataset_scope(monkeypatch):
    entered = []

    class Scope:
        def __init__(self, dataset_id, owner_id):
            entered.append(("enter", dataset_id, owner_id))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            entered.append(("exit",))
            return False

    ctx_mod = __import__("cognee.context_global_variables", fromlist=["x"])
    graph_mod = __import__("cognee.infrastructure.databases.graph", fromlist=["x"])
    monkeypatch.setattr(ctx_mod, "set_database_global_context_variables", Scope)
    monkeypatch.setattr(graph_mod, "get_graph_engine", AsyncMock(return_value=FullAdapter()))

    dataset_id, owner_id = uuid4(), uuid4()
    caps = await capabilities_mod.resolve_graph_capabilities(dataset_id, owner_id)

    assert caps.supports_truth_state is True
    assert entered == [("enter", dataset_id, owner_id), ("exit",)]


@pytest.mark.asyncio
async def test_resolve_fails_open_when_the_engine_cannot_be_created(monkeypatch):
    class Scope:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    ctx_mod = __import__("cognee.context_global_variables", fromlist=["x"])
    graph_mod = __import__("cognee.infrastructure.databases.graph", fromlist=["x"])
    monkeypatch.setattr(ctx_mod, "set_database_global_context_variables", Scope)
    monkeypatch.setattr(graph_mod, "get_graph_engine", AsyncMock(side_effect=RuntimeError("x")))

    caps = await capabilities_mod.resolve_graph_capabilities(uuid4(), None)

    assert caps.supports_feedback_weights is True
    assert caps.supports_truth_state is True
    assert caps.supports_incremental_chunk_updates is False
