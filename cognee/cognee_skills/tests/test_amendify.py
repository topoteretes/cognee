"""Tests for cognee.cognee_skills.amendify — unit tests with mocked graph engine."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock, call
from uuid import uuid4


def _make_amendment_node(amendment_id="amend-001", status="proposed"):
    nid = uuid4()
    props = {
        "type": "SkillAmendment",
        "nid": str(nid),
        "id": str(nid),
        "amendment_id": amendment_id,
        "skill_id": "test-skill",
        "skill_name": "Test Skill",
        "inspection_id": "insp-001",
        "original_instructions": "Do the thing step by step.",
        "amended_instructions": "Do the thing step by step.\n\nAlso handle edge cases.",
        "change_explanation": "Added edge case handling",
        "expected_improvement": "Fewer failures on edge cases",
        "status": status,
        "pre_amendment_avg_score": 0.2,
        "amendment_model": "",
        "amendment_confidence": 0.0,
        "applied_at_ms": 0,
        "post_amendment_avg_score": 0.0,
        "post_amendment_run_count": 0,
    }
    return (nid, props)


def _make_skill_node(skill_id="test-skill"):
    nid = uuid4()
    props = {
        "type": "Skill",
        "id": str(nid),
        "skill_id": skill_id,
        "name": "Test Skill",
        "description": "A test skill.",
        "instructions": "Do the thing step by step.",
        "instruction_summary": "Does the thing.",
        "content_hash": "abc123",
        "source_path": "",
        "tags": [],
        "complexity": "simple",
    }
    return (nid, props)


class TestAmendify(unittest.TestCase):
    @patch("cognee.cognee_skills.amendify.add_data_points", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.amendify._make_change_event")
    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.tasks.enrich_skills.enrich_skills", new_callable=AsyncMock)
    @patch(
        "cognee.cognee_skills.tasks.materialize_task_patterns.materialize_task_patterns",
        new_callable=AsyncMock,
    )
    def test_amendify_updates_skill(
        self, mock_mat, mock_enrich, mock_engine_fn, mock_event, mock_add_dp
    ):
        amendment_node = _make_amendment_node()
        skill_node = _make_skill_node()

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([amendment_node, skill_node], []))
        mock_engine_fn.return_value = engine

        mock_event.return_value = MagicMock()
        mock_enrich.return_value = []

        from cognee.cognee_skills.amendify import amendify

        result = asyncio.run(amendify("amend-001"))

        assert result["success"] is True
        assert result["status"] == "applied"
        assert result["skill_id"] == "test-skill"

        # Verify add_data_points was called (skill update + change event + amendment status)
        assert mock_add_dp.call_count >= 3

    @patch("cognee.cognee_skills.amendify.add_data_points", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.amendify._make_change_event")
    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.tasks.enrich_skills.enrich_skills", new_callable=AsyncMock)
    @patch(
        "cognee.cognee_skills.tasks.materialize_task_patterns.materialize_task_patterns",
        new_callable=AsyncMock,
    )
    def test_amendify_emits_change_event(
        self, mock_mat, mock_enrich, mock_engine_fn, mock_event, mock_add_dp
    ):
        amendment_node = _make_amendment_node()
        skill_node = _make_skill_node()

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([amendment_node, skill_node], []))
        mock_engine_fn.return_value = engine

        event_obj = MagicMock()
        mock_event.return_value = event_obj
        mock_enrich.return_value = []

        from cognee.cognee_skills.amendify import amendify

        asyncio.run(amendify("amend-001"))

        # Verify change event was created with change_type="amended"
        mock_event.assert_called_once()
        call_kwargs = mock_event.call_args
        assert call_kwargs[0][2] == "amended"  # change_type arg
        # Verify the event was persisted
        mock_add_dp.assert_called()

    @patch("cognee.cognee_skills.amendify.add_data_points", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.amendify._make_change_event")
    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.tasks.enrich_skills.enrich_skills", new_callable=AsyncMock)
    @patch(
        "cognee.cognee_skills.tasks.materialize_task_patterns.materialize_task_patterns",
        new_callable=AsyncMock,
    )
    def test_rollback_restores_original(
        self, mock_mat, mock_enrich, mock_engine_fn, mock_event, mock_add_dp
    ):
        amendment_node = _make_amendment_node(status="applied")
        skill_node = _make_skill_node()

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([amendment_node, skill_node], []))
        mock_engine_fn.return_value = engine

        mock_event.return_value = MagicMock()
        mock_enrich.return_value = []

        from cognee.cognee_skills.amendify import rollback_amendify

        result = asyncio.run(rollback_amendify("amend-001"))

        assert result is True

        # Verify change event was emitted with rolled_back type
        mock_event.assert_called_once()
        assert mock_event.call_args[0][2] == "rolled_back"

        # Verify add_data_points was called (skill restore + event + amendment status)
        assert mock_add_dp.call_count >= 3

    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    def test_rollback_fails_for_non_applied_amendment(self, mock_engine_fn):
        amendment_node = _make_amendment_node(status="proposed")

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([amendment_node], []))
        mock_engine_fn.return_value = engine

        from cognee.cognee_skills.amendify import rollback_amendify

        result = asyncio.run(rollback_amendify("amend-001"))

        assert result is False

    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    def test_amendify_not_found(self, mock_engine_fn):
        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([], []))
        mock_engine_fn.return_value = engine

        from cognee.cognee_skills.amendify import amendify

        result = asyncio.run(amendify("nonexistent"))

        assert result["success"] is False
        assert "not found" in result["error"]

    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    def test_amendify_rejects_already_applied(self, mock_engine_fn):
        """Cannot apply an amendment that's already been applied."""
        amendment_node = _make_amendment_node(status="applied")

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=([amendment_node], []))
        mock_engine_fn.return_value = engine

        from cognee.cognee_skills.amendify import amendify

        result = asyncio.run(amendify("amend-001"))

        assert result["success"] is False
        assert "applied" in result["error"]

    @patch("cognee.cognee_skills.amendify.add_data_points", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.amendify.get_graph_engine", new_callable=AsyncMock)
    def test_evaluate_amendify(self, mock_engine_fn, mock_add_dp):
        """evaluate_amendify computes post-amendment stats from runs after applied_at_ms."""
        amendment_node = _make_amendment_node(status="applied")
        # Add applied_at_ms to amendment
        amendment_node[1]["applied_at_ms"] = 1000

        # Runs: one before amendment (should be excluded), two after
        pre_run = (
            uuid4(),
            {
                "type": "SkillRun",
                "selected_skill_id": "test-skill",
                "success_score": 0.1,
                "started_at_ms": 500,
            },
        )
        post_run1 = (
            uuid4(),
            {
                "type": "SkillRun",
                "selected_skill_id": "test-skill",
                "success_score": 0.8,
                "started_at_ms": 2000,
            },
        )
        post_run2 = (
            uuid4(),
            {
                "type": "SkillRun",
                "selected_skill_id": "test-skill",
                "success_score": 0.9,
                "started_at_ms": 3000,
            },
        )

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(
            return_value=([amendment_node, pre_run, post_run1, post_run2], [])
        )
        mock_engine_fn.return_value = engine

        from cognee.cognee_skills.amendify import evaluate_amendify

        result = asyncio.run(evaluate_amendify("amend-001"))

        assert result["run_count"] == 2
        assert abs(result["post_avg"] - 0.85) < 0.01
        assert result["improvement"] > 0
        assert result["recommendation"] == "keep"

        # Verify amendment stats were persisted via add_data_points
        mock_add_dp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
