"""Tests for cognee.skills.inspect — unit tests with mocked LLM + graph."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from cognee.skills.models.skill_inspection import InspectionResult


def _make_skill_node(skill_id="test-skill", name="Test Skill"):
    nid = uuid4()
    props = {
        "type": "Skill",
        "skill_id": skill_id,
        "name": name,
        "instructions": "Do the thing step by step.",
        "content_hash": "abc123",
    }
    return (nid, props)


def _make_failed_run(skill_id="test-skill", score=0.2, run_id=None, error_type="llm_error"):
    nid = uuid4()
    props = {
        "type": "SkillRun",
        "run_id": run_id or str(uuid4()),
        "selected_skill_id": skill_id,
        "success_score": score,
        "task_text": "Summarize this document",
        "error_type": error_type,
        "error_message": "Model failed to follow instructions",
        "result_summary": "Incomplete output",
    }
    return (nid, props)


MOCK_INSPECTION_RESULT = InspectionResult(
    failure_category="instruction_gap",
    root_cause="Instructions lack detail on edge cases",
    severity="high",
    improvement_hypothesis="Add examples for edge cases in the instructions",
    confidence=0.85,
)


class TestInspectSkill(unittest.TestCase):
    @patch("cognee.skills.inspect.add_data_points", new_callable=AsyncMock)
    @patch("cognee.skills.inspect.get_llm_config")
    @patch("cognee.skills.inspect.LLMGateway.acreate_structured_output", new_callable=AsyncMock)
    @patch("cognee.skills.inspect.get_graph_engine", new_callable=AsyncMock)
    def test_inspect_returns_inspection(self, mock_engine_fn, mock_llm, mock_config, mock_add_dp):
        skill_node = _make_skill_node()
        run1 = _make_failed_run(score=0.1)
        run2 = _make_failed_run(score=0.3)

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([skill_node, run1, run2], []))
        mock_engine_fn.return_value = engine
        mock_llm.return_value = MOCK_INSPECTION_RESULT
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini")

        from cognee.skills.inspect import inspect_skill

        result = asyncio.run(inspect_skill("test-skill", min_runs=1))

        assert result is not None
        assert result.skill_id == "test-skill"
        assert result.failure_category == "instruction_gap"
        assert result.severity == "high"
        assert result.analyzed_run_count == 2
        assert result.avg_success_score == 0.2
        assert result.inspection_confidence == 0.85
        mock_add_dp.assert_called_once()

    @patch("cognee.skills.inspect.get_graph_engine", new_callable=AsyncMock)
    def test_inspect_no_failed_runs(self, mock_engine_fn):
        skill_node = _make_skill_node()

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([skill_node], []))
        mock_engine_fn.return_value = engine

        from cognee.skills.inspect import inspect_skill

        result = asyncio.run(inspect_skill("test-skill", min_runs=1))

        assert result is None

    @patch("cognee.skills.inspect.add_data_points", new_callable=AsyncMock)
    @patch("cognee.skills.inspect.get_llm_config")
    @patch("cognee.skills.inspect.LLMGateway.acreate_structured_output", new_callable=AsyncMock)
    @patch("cognee.skills.inspect.get_graph_engine", new_callable=AsyncMock)
    def test_inspect_aggregates_multiple_runs(
        self, mock_engine_fn, mock_llm, mock_config, mock_add_dp
    ):
        skill_node = _make_skill_node()
        runs = [_make_failed_run(score=0.1 * i, run_id=f"run-{i}") for i in range(5)]

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([skill_node] + runs, []))
        mock_engine_fn.return_value = engine
        mock_llm.return_value = MOCK_INSPECTION_RESULT
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini")

        from cognee.skills.inspect import inspect_skill

        result = asyncio.run(inspect_skill("test-skill", min_runs=1))

        assert result is not None
        assert result.analyzed_run_count == 5
        assert len(result.analyzed_run_ids) == 5

    @patch("cognee.skills.inspect.get_graph_engine", new_callable=AsyncMock)
    def test_inspect_insufficient_runs(self, mock_engine_fn):
        skill_node = _make_skill_node()
        run1 = _make_failed_run(score=0.2)

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([skill_node, run1], []))
        mock_engine_fn.return_value = engine

        from cognee.skills.inspect import inspect_skill

        # Requires 3 runs but only 1 exists
        result = asyncio.run(inspect_skill("test-skill", min_runs=3))

        assert result is None


if __name__ == "__main__":
    unittest.main()
