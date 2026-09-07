import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.improve.constants import DEFAULT_FEEDBACK_ALPHA
from cognee.tasks.memify.apply_feedback_weights import (
    apply_feedback_weights,
    normalize_feedback_score,
    stream_update_weight,
    validate_feedback_alpha,
)
from cognee.tasks.memify.feedback_weights_constants import (
    FEEDBACK_SOURCE_IMPLICIT,
    FEEDBACK_WEIGHTS_MAX_ATTEMPTS,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_EDGE_IDS_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_NODE_IDS_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_SCORE_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_ATTEMPTS_KEY,
)

apply_feedback_weights_module = sys.modules["cognee.tasks.memify.apply_feedback_weights"]

APPLIED = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY
NODE_IDS = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_NODE_IDS_KEY
EDGE_IDS = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_EDGE_IDS_KEY
SCORE = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_SCORE_KEY
ATTEMPTS = MEMIFY_METADATA_FEEDBACK_WEIGHTS_ATTEMPTS_KEY


class InMemoryGraphWithWeights:
    """Flat weight maps; ids absent from a map count as deleted from the graph."""

    def __init__(self, missing_edge: bool = False, failing_edge_writes: set | None = None):
        self.node_weights = {"n1": 0.5}
        self.edge_weights = {"e1": 0.5}
        self.missing_edge = missing_edge
        self.failing_edge_writes = failing_edge_writes or set()
        self.node_write_log: list[dict] = []
        self.edge_write_log: list[dict] = []

    async def get_node_feedback_weights(self, node_ids):
        return {
            node_id: self.node_weights[node_id]
            for node_id in node_ids
            if node_id in self.node_weights
        }

    async def set_node_feedback_weights(self, node_feedback_weights):
        self.node_write_log.append(dict(node_feedback_weights))
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
        self.edge_write_log.append(dict(edge_feedback_weights))
        result = {}
        for edge_object_id, weight in edge_feedback_weights.items():
            if edge_object_id in self.failing_edge_writes:
                result[edge_object_id] = False
            elif edge_object_id in self.edge_weights:
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


class RecordingSessionManager:
    """Stores memify_metadata the way the cache adapters do: overlay incoming keys."""

    def __init__(self):
        self.is_available = True
        self.metadata: dict[str, dict] = {}
        self.update_qa = AsyncMock(side_effect=self._update_qa)

    async def _update_qa(self, *, user_id, session_id, qa_id, memify_metadata, **_):
        existing = self.metadata.get(qa_id, {})
        self.metadata[qa_id] = {**existing, **memify_metadata}
        return True


def _feedback_item(memify_metadata=None, used_graph_element_ids=None, **overrides):
    item = {
        "session_id": "s1",
        "qa_id": "q1",
        "feedback_score": 5,
        "used_graph_element_ids": used_graph_element_ids
        if used_graph_element_ids is not None
        else {"node_ids": ["n1"], "edge_ids": ["e1"]},
        "memify_metadata": memify_metadata if memify_metadata is not None else {},
    }
    item.update(overrides)
    return item


def _mock_user():
    user = MagicMock()
    user.id = "u1"
    return user


async def _run(graph, session_manager, items, alpha=0.1):
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
        return await apply_feedback_weights(items, alpha=alpha)


async def _run_from_store(graph, session_manager, item_factory, times: int, alpha=0.1):
    """Re-run the task ``times`` times, feeding back the metadata the store holds."""
    results = []
    for _ in range(times):
        stored = session_manager.metadata.get("q1", {})
        results.append(await _run(graph, session_manager, [item_factory(stored)], alpha=alpha))
    return results


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
    session_manager = RecordingSessionManager()

    result = await _run(graph, session_manager, [_feedback_item()])

    assert result["processed"] == 1
    assert result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert graph.edge_weights["e1"] == pytest.approx(0.55)

    written = session_manager.update_qa.call_args.kwargs["memify_metadata"]
    assert written[APPLIED] is True
    assert written[NODE_IDS] == ["n1"]
    assert written[EDGE_IDS] == ["e1"]
    assert written[SCORE] == 5
    assert written[ATTEMPTS] == 1


@pytest.mark.asyncio
async def test_apply_feedback_weights_ladybug_success_marks_applied_true():
    graph = InMemoryGraphWithNestedEdgeProperties()
    session_manager = RecordingSessionManager()

    result = await _run(graph, session_manager, [_feedback_item()])

    assert result["processed"] == 1
    assert result["applied"] == 1
    assert graph.nodes["n1"]["feedback_weight"] == pytest.approx(0.55)
    assert graph.edges["e1"]["properties"]["feedback_weight"] == pytest.approx(0.55)
    assert session_manager.metadata["q1"][APPLIED] is True


@pytest.mark.asyncio
async def test_apply_feedback_weights_skips_already_applied():
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()

    result = await _run(graph, session_manager, [_feedback_item(memify_metadata={APPLIED: True})])

    assert result["processed"] == 0
    assert result["applied"] == 0
    assert result["skipped"] == 1
    session_manager.update_qa.assert_not_called()
    assert graph.node_weights["n1"] == 0.5


@pytest.mark.asyncio
async def test_apply_feedback_weights_no_ids_marks_row_done_and_touches_no_weights():
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()

    result = await _run(
        graph,
        session_manager,
        [_feedback_item(used_graph_element_ids={"node_ids": [], "edge_ids": []})],
    )

    assert result == {"processed": 0, "applied": 0, "skipped": 1}
    assert session_manager.update_qa.call_args.kwargs["memify_metadata"] == {APPLIED: True}
    assert graph.node_weights["n1"] == 0.5
    assert graph.edge_weights["e1"] == 0.5


@pytest.mark.asyncio
async def test_apply_feedback_weights_deleted_ids_are_pruned_and_row_is_done():
    graph = InMemoryGraphWithWeights(missing_edge=True)
    session_manager = RecordingSessionManager()

    result = await _run(graph, session_manager, [_feedback_item()])

    assert result["processed"] == 1
    assert result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    written = session_manager.metadata["q1"]
    assert written[APPLIED] is True
    assert written[NODE_IDS] == ["n1"]
    assert written[EDGE_IDS] == []  # e1 is gone: dropped, not retried


@pytest.mark.asyncio
async def test_deleted_node_three_runs_moves_each_surviving_element_exactly_once():
    """Acceptance: a QA whose ids include one deleted node, run three times, moves each
    surviving element exactly once."""
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()

    def item(stored):
        return _feedback_item(
            memify_metadata=stored,
            used_graph_element_ids={"node_ids": ["n1", "n_deleted"], "edge_ids": ["e1"]},
        )

    results = await _run_from_store(graph, session_manager, item, times=3)

    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert graph.edge_weights["e1"] == pytest.approx(0.55)
    assert graph.node_write_log == [{"n1": pytest.approx(0.55)}]
    assert graph.edge_write_log == [{"e1": pytest.approx(0.55)}]
    assert [r["processed"] for r in results] == [1, 0, 0]
    assert [r["skipped"] for r in results] == [0, 1, 1]
    stored = session_manager.metadata["q1"]
    assert stored[APPLIED] is True
    assert stored[NODE_IDS] == ["n1"]
    assert stored[ATTEMPTS] == 1


@pytest.mark.asyncio
async def test_partial_write_failure_retries_only_unapplied_ids():
    graph = InMemoryGraphWithWeights(failing_edge_writes={"e1"})
    session_manager = RecordingSessionManager()

    def item(stored):
        return _feedback_item(memify_metadata=stored)

    first = (await _run_from_store(graph, session_manager, item, times=1))[0]
    assert first == {"processed": 1, "applied": 0, "skipped": 0}
    stored = session_manager.metadata["q1"]
    assert stored[APPLIED] is False
    assert stored[NODE_IDS] == ["n1"]
    assert stored[EDGE_IDS] == []
    assert stored[ATTEMPTS] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert graph.edge_weights["e1"] == 0.5

    graph.failing_edge_writes = set()
    second = (await _run_from_store(graph, session_manager, item, times=1))[0]
    assert second == {"processed": 1, "applied": 1, "skipped": 0}
    stored = session_manager.metadata["q1"]
    assert stored[APPLIED] is True
    assert stored[NODE_IDS] == ["n1"]
    assert stored[EDGE_IDS] == ["e1"]
    assert stored[ATTEMPTS] == 2
    # n1 moved exactly once across both runs; e1 moved once on the retry.
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert graph.edge_weights["e1"] == pytest.approx(0.55)
    assert graph.node_write_log == [{"n1": pytest.approx(0.55)}]


@pytest.mark.asyncio
async def test_attempts_are_capped_then_row_is_marked_done():
    graph = InMemoryGraphWithWeights(failing_edge_writes={"e1"})
    session_manager = RecordingSessionManager()

    def item(stored):
        return _feedback_item(memify_metadata=stored)

    results = await _run_from_store(
        graph, session_manager, item, times=FEEDBACK_WEIGHTS_MAX_ATTEMPTS + 1
    )

    assert [r["processed"] for r in results] == [1] * FEEDBACK_WEIGHTS_MAX_ATTEMPTS + [0]
    assert all(r["applied"] == 0 for r in results)
    stored = session_manager.metadata["q1"]
    assert stored[APPLIED] is True
    assert stored[ATTEMPTS] == FEEDBACK_WEIGHTS_MAX_ATTEMPTS
    assert stored[EDGE_IDS] == []
    assert graph.node_weights["n1"] == pytest.approx(0.55)
    assert len(graph.node_write_log) == 1
    assert len(graph.edge_write_log) == FEEDBACK_WEIGHTS_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_rerated_row_starts_over_with_the_new_score():
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()
    # add_feedback resets the done flag but leaves the bookkeeping; a different score
    # must move every id again, with the new rating.
    stored = {APPLIED: False, NODE_IDS: ["n1"], EDGE_IDS: ["e1"], SCORE: 5, ATTEMPTS: 1}

    result = await _run(
        graph, session_manager, [_feedback_item(memify_metadata=stored, feedback_score=1)]
    )

    assert result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.45)
    assert graph.edge_weights["e1"] == pytest.approx(0.45)
    written = session_manager.metadata["q1"]
    assert written[SCORE] == 1
    assert written[ATTEMPTS] == 1
    assert written[APPLIED] is True


@pytest.mark.asyncio
async def test_same_score_with_reset_flag_is_a_no_op_on_weights():
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()
    stored = {APPLIED: False, NODE_IDS: ["n1"], EDGE_IDS: ["e1"], SCORE: 5, ATTEMPTS: 1}

    result = await _run(graph, session_manager, [_feedback_item(memify_metadata=stored)])

    assert result["applied"] == 1
    assert graph.node_weights["n1"] == 0.5
    assert graph.edge_weights["e1"] == 0.5
    assert session_manager.metadata["q1"][APPLIED] is True


@pytest.mark.asyncio
async def test_implicit_feedback_uses_half_alpha():
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()

    result = await _run(
        graph,
        session_manager,
        [_feedback_item(feedback_source=FEEDBACK_SOURCE_IMPLICIT, feedback_text="thanks!")],
        alpha=0.1,
    )

    assert result["applied"] == 1
    assert graph.node_weights["n1"] == pytest.approx(0.525)
    assert graph.edge_weights["e1"] == pytest.approx(0.525)


@pytest.mark.asyncio
async def test_explicit_feedback_uses_full_alpha():
    graph = InMemoryGraphWithWeights()
    session_manager = RecordingSessionManager()

    await _run(graph, session_manager, [_feedback_item(feedback_source="explicit")], alpha=0.1)

    assert graph.node_weights["n1"] == pytest.approx(0.55)


class TestFeedbackAlpha:
    """The learning rate has one default and one range check, shared by task and pipeline."""

    def test_default_alpha_is_the_shared_constant(self):
        import inspect

        from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline

        assert DEFAULT_FEEDBACK_ALPHA == 0.1
        task_default = inspect.signature(apply_feedback_weights).parameters["alpha"].default
        pipeline_default = (
            inspect.signature(apply_feedback_weights_pipeline).parameters["alpha"].default
        )
        assert task_default is DEFAULT_FEEDBACK_ALPHA
        assert pipeline_default is DEFAULT_FEEDBACK_ALPHA

    @pytest.mark.parametrize("alpha", [0.0001, 0.1, 0.5, 1.0])
    def test_validate_feedback_alpha_accepts_the_half_open_unit_interval(self, alpha):
        assert validate_feedback_alpha(alpha) == alpha

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0001, 2.0])
    def test_validate_feedback_alpha_rejects_values_outside_range(self, alpha):
        with pytest.raises(CogneeValidationError, match=r"alpha must be in range \(0, 1\]"):
            validate_feedback_alpha(alpha)

    @pytest.mark.parametrize("alpha", [0.0, 1.5])
    def test_stream_update_weight_uses_the_shared_check(self, alpha):
        with pytest.raises(CogneeValidationError):
            stream_update_weight(0.5, 1.0, alpha)

    @pytest.mark.asyncio
    async def test_apply_feedback_weights_uses_the_shared_check(self):
        with pytest.raises(CogneeValidationError):
            await apply_feedback_weights([], alpha=0.0)
