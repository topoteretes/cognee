import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.tasks.memify.apply_feedback_weights import (
    apply_feedback_weights,
    normalize_feedback_score,
    stream_update_weight,
)
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)

apply_feedback_weights_module = sys.modules["cognee.tasks.memify.apply_feedback_weights"]


class InMemoryGraphWithWeights:
    def __init__(self, missing_edge: bool = False):
        self.node_weights = {"n1": 0.5}
        self.edge_weights = {"e1": 0.5}
        self.missing_edge = missing_edge

    async def get_node_feedback_weights(self, node_ids):
        return {
            node_id: self.node_weights[node_id]
            for node_id in node_ids
            if node_id in self.node_weights
        }

    async def set_node_feedback_weights(self, node_feedback_weights):
        result = {}
        for node_id, weight in node_feedback_weights.items():
            if node_id in self.node_weights:
                self.node_weights[node_id] = float(weight)
                result[node_id] = True
            else:
                result[node_id] = False
        return result

    async def get_edge_feedback_weights(self, edge_object_ids):
        if self.missing_edge:
            return {}
        return {
            edge_object_id: self.edge_weights[edge_object_id]
            for edge_object_id in edge_object_ids
            if edge_object_id in self.edge_weights
        }

    async def set_edge_feedback_weights(self, edge_feedback_weights):
        result = {}
        for edge_object_id, weight in edge_feedback_weights.items():
            if edge_object_id in self.edge_weights:
                self.edge_weights[edge_object_id] = float(weight)
                result[edge_object_id] = True
            else:
                result[edge_object_id] = False
        return result


class InMemoryGraphWithNestedEdgeProperties:
    def __init__(self):
        self.nodes = {"n1": {"id": "n1", "feedback_weight": 0.5}}
        self.edges = {
            "e1": {
                "from_id": "n1",
                "to_id": "n2",
                "relationship_name": "REL",
                "properties": {"edge_object_id": "e1", "feedback_weight": 0.5},
            }
        }

    async def get_node_feedback_weights(self, node_ids):
        return {
            node_id: float(self.nodes[node_id].get("feedback_weight", 0.5))
            for node_id in node_ids
            if node_id in self.nodes
        }

    async def set_node_feedback_weights(self, node_feedback_weights):
        result = {}
        for node_id, weight in node_feedback_weights.items():
            if node_id in self.nodes:
                self.nodes[node_id]["feedback_weight"] = float(weight)
                result[node_id] = True
            else:
                result[node_id] = False
        return result

    async def get_edge_feedback_weights(self, edge_object_ids):
        result = {}
        for edge_object_id in edge_object_ids:
            edge = self.edges.get(edge_object_id)
            if edge is not None:
                result[edge_object_id] = float(edge["properties"].get("feedback_weight", 0.5))
        return result

    async def set_edge_feedback_weights(self, edge_feedback_weights):
        result = {}
        for edge_object_id, weight in edge_feedback_weights.items():
            edge = self.edges.get(edge_object_id)
            if edge is None:
                result[edge_object_id] = False
            else:
                edge["properties"]["feedback_weight"] = float(weight)
                result[edge_object_id] = True
        return result


def _feedback_item(memify_metadata=None, used_graph_element_ids=None):
    return {
        "session_id": "s1",
        "qa_id": "q1",
        "feedback_score": 5,
        "used_graph_element_ids": used_graph_element_ids
        if used_graph_element_ids is not None
        else {"node_ids": ["n1"], "edge_ids": ["e1"]},
        "memify_metadata": memify_metadata if memify_metadata is not None else {},
    }


def _mock_user():
    user = MagicMock()
    user.id = "u1"
    return user


def test_normalize_feedback_score_mapping():
    assert normalize_feedback_score(1) == 0.0
    assert normalize_feedback_score(2) == 0.25
    assert normalize_feedback_score(3) == 0.5
    assert normalize_feedback_score(4) == 0.75
    assert normalize_feedback_score(5) == 1.0


def test_streaming_update_formula_and_bounds():
    assert stream_update_weight(0.5, 1.0, 0.1) == pytest.approx(0.55)
    assert stream_update_weight(0.5, 0.0, 0.1) == pytest.approx(0.45)
    assert stream_update_weight(2.0, 1.0, 0.5) == 1.0
    assert stream_update_weight(-1.0, 0.0, 0.5) == 0.0


@pytest.mark.asyncio
async def test_apply_feedback_weights_neo4j_success_marks_applied_true():
    graph = InMemoryGraphWithWeights()
    session_manager = MagicMock()
    session_manager.is_available = True
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_feedback_weights_module, "session_user") as mock_session_user,
        patch.object(apply_feedback_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_feedback_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_feedback_weights([_feedback_item()], alpha=0.1)

    assert result["processed"] == 1
    assert result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert graph.edge_weights["e1"] == pytest.approx(0.55)

    call_kwargs = session_manager.update_qa.call_args.kwargs
    assert call_kwargs["memify_metadata"][MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY] is True


@pytest.mark.asyncio
async def test_apply_feedback_weights_ladybug_success_marks_applied_true():
    graph = InMemoryGraphWithNestedEdgeProperties()
    session_manager = MagicMock()
    session_manager.is_available = True
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_feedback_weights_module, "session_user") as mock_session_user,
        patch.object(apply_feedback_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_feedback_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_feedback_weights([_feedback_item()], alpha=0.1)

    assert result["processed"] == 1
    assert result["applied"] == 1
    assert graph.nodes["n1"]["feedback_weight"] == pytest.approx(0.55)
    assert graph.edges["e1"]["properties"]["feedback_weight"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_apply_feedback_weights_skips_already_applied():
    graph = InMemoryGraphWithWeights()
    session_manager = MagicMock()
    session_manager.is_available = True
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_feedback_weights_module, "session_user") as mock_session_user,
        patch.object(apply_feedback_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_feedback_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_feedback_weights(
            [_feedback_item(memify_metadata={MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: True})],
            alpha=0.1,
        )

    assert result["processed"] == 0
    assert result["applied"] == 0
    session_manager.update_qa.assert_not_called()


@pytest.mark.asyncio
async def test_apply_feedback_weights_missing_mapping_sets_false():
    graph = InMemoryGraphWithWeights()
    session_manager = MagicMock()
    session_manager.is_available = True
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_feedback_weights_module, "session_user") as mock_session_user,
        patch.object(apply_feedback_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_feedback_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        await apply_feedback_weights(
            [_feedback_item(used_graph_element_ids={"node_ids": [], "edge_ids": []})],
            alpha=0.1,
        )

    call_kwargs = session_manager.update_qa.call_args.kwargs
    assert call_kwargs["memify_metadata"][MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY] is False


@pytest.mark.asyncio
async def test_apply_feedback_weights_partial_failure_keeps_false():
    graph = InMemoryGraphWithWeights(missing_edge=True)
    session_manager = MagicMock()
    session_manager.is_available = True
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_feedback_weights_module, "session_user") as mock_session_user,
        patch.object(apply_feedback_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_feedback_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_feedback_weights([_feedback_item()], alpha=0.1)

    assert result["processed"] == 1
    assert result["applied"] == 0
    call_kwargs = session_manager.update_qa.call_args.kwargs
    assert call_kwargs["memify_metadata"][MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY] is False
