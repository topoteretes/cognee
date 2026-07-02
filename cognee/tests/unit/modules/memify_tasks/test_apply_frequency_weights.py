import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.tasks.memify.apply_frequency_weights import apply_frequency_weights
from cognee.tasks.memify.frequency_weights_constants import (
    MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY,
)

apply_frequency_weights_module = sys.modules["cognee.tasks.memify.apply_frequency_weights"]


class InMemoryGraphWithFrequencyWeights:
    def __init__(self, missing_edge: bool = False):
        self.node_weights = {"n1": 0.0}
        self.edge_weights = {"e1": 0.0}
        self.missing_edge = missing_edge

    async def get_node_frequency_weights(self, node_ids):
        return {
            node_id: self.node_weights[node_id]
            for node_id in node_ids
            if node_id in self.node_weights
        }

    async def set_node_frequency_weights(self, node_frequency_weights):
        result = {}
        for node_id, weight in node_frequency_weights.items():
            if node_id in self.node_weights:
                self.node_weights[node_id] = float(weight)
                result[node_id] = True
            else:
                result[node_id] = False
        return result

    async def get_edge_frequency_weights(self, edge_object_ids):
        if self.missing_edge:
            return {}
        return {
            edge_object_id: self.edge_weights[edge_object_id]
            for edge_object_id in edge_object_ids
            if edge_object_id in self.edge_weights
        }

    async def set_edge_frequency_weights(self, edge_frequency_weights):
        result = {}
        for edge_object_id, weight in edge_frequency_weights.items():
            if edge_object_id in self.edge_weights:
                self.edge_weights[edge_object_id] = float(weight)
                result[edge_object_id] = True
            else:
                result[edge_object_id] = False
        return result


def _frequency_item(memify_metadata=None, used_graph_element_ids=None):
    return {
        "session_id": "s1",
        "qa_id": "q1",
        "used_graph_element_ids": used_graph_element_ids
        if used_graph_element_ids is not None
        else {"node_ids": ["n1"], "edge_ids": ["e1"]},
        "memify_metadata": memify_metadata if memify_metadata is not None else {},
    }


def _mock_user():
    user = MagicMock()
    user.id = "u1"
    return user


@pytest.mark.asyncio
async def test_apply_frequency_weights_success_increments_by_one_and_marks_true():
    graph = InMemoryGraphWithFrequencyWeights()
    session_manager = MagicMock()
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_frequency_weights_module, "session_user") as mock_session_user,
        patch.object(apply_frequency_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_frequency_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_frequency_weights([_frequency_item()])

    assert result["processed"] == 1
    assert result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(1.0)
    assert graph.edge_weights["e1"] == pytest.approx(1.0)

    call_kwargs = session_manager.update_qa.call_args.kwargs
    assert call_kwargs["memify_metadata"][MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY] is True


@pytest.mark.asyncio
async def test_apply_frequency_weights_skips_already_applied():
    graph = InMemoryGraphWithFrequencyWeights()
    session_manager = MagicMock()
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_frequency_weights_module, "session_user") as mock_session_user,
        patch.object(apply_frequency_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_frequency_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_frequency_weights(
            [_frequency_item(memify_metadata={MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY: True})]
        )

    assert result["processed"] == 0
    assert result["applied"] == 0
    assert result["skipped"] == 1
    session_manager.update_qa.assert_not_called()


@pytest.mark.asyncio
async def test_apply_frequency_weights_missing_mapping_sets_false():
    graph = InMemoryGraphWithFrequencyWeights()
    session_manager = MagicMock()
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_frequency_weights_module, "session_user") as mock_session_user,
        patch.object(apply_frequency_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_frequency_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_frequency_weights(
            [_frequency_item(used_graph_element_ids={"node_ids": [], "edge_ids": []})]
        )

    assert result["processed"] == 0
    assert result["applied"] == 0
    assert result["skipped"] == 1
    call_kwargs = session_manager.update_qa.call_args.kwargs
    assert call_kwargs["memify_metadata"][MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY] is False


@pytest.mark.asyncio
async def test_apply_frequency_weights_partial_failure_keeps_false():
    graph = InMemoryGraphWithFrequencyWeights(missing_edge=True)
    session_manager = MagicMock()
    session_manager.update_qa = AsyncMock(return_value=True)

    with (
        patch.object(apply_frequency_weights_module, "session_user") as mock_session_user,
        patch.object(apply_frequency_weights_module, "get_graph_engine", return_value=graph),
        patch.object(
            apply_frequency_weights_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = _mock_user()
        result = await apply_frequency_weights([_frequency_item()])

    assert result["processed"] == 1
    assert result["applied"] == 0
    assert graph.node_weights["n1"] == pytest.approx(1.0)
    call_kwargs = session_manager.update_qa.call_args.kwargs
    assert call_kwargs["memify_metadata"][MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY] is False
