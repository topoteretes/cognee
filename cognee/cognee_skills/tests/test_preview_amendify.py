"""Tests for cognee.cognee_skills.preview_amendify — unit tests with mocked LLM."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from cognee.cognee_skills.models.skill_inspection import SkillInspection
from cognee.cognee_skills.models.skill_amendment import AmendmentProposal


SAMPLE_INSPECTION = SkillInspection(
    id=uuid4(),
    name="inspection: Test Skill",
    description="Test inspection",
    inspection_id="insp-001",
    skill_id="test-skill",
    skill_name="Test Skill",
    failure_category="instruction_gap",
    root_cause="Missing edge case handling",
    severity="high",
    improvement_hypothesis="Add examples for edge cases",
    analyzed_run_ids=["run-1", "run-2"],
    analyzed_run_count=2,
    avg_success_score=0.2,
    inspection_model="openai/gpt-4o-mini",
    inspection_confidence=0.85,
)

SAMPLE_SKILL = {
    "skill_id": "test-skill",
    "name": "Test Skill",
    "instructions": "Do the thing step by step.\n\n1. First step\n2. Second step",
    "instruction_summary": "Does the thing.",
    "description": "A test skill.",
    "tags": ["code"],
    "complexity": "simple",
    "source_path": "",
    "task_patterns": [],
}

MOCK_AMENDMENT_PROPOSAL = AmendmentProposal(
    amended_instructions=(
        "Do the thing step by step.\n\n1. First step\n2. Second step\n"
        "3. Handle edge case: if input is empty, return a helpful message"
    ),
    change_explanation="Added step 3 for edge case handling when input is empty",
    expected_improvement="Skill will no longer fail on empty inputs",
    confidence=0.80,
)


class TestPreviewAmendify(unittest.TestCase):
    @patch("cognee.cognee_skills.preview_amendify.add_data_points", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.preview_amendify.get_llm_config")
    @patch(
        "cognee.cognee_skills.preview_amendify.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
    )
    def test_preview_returns_amendment(self, mock_llm, mock_config, mock_add_dp):
        mock_llm.return_value = MOCK_AMENDMENT_PROPOSAL
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini")

        from cognee.cognee_skills.preview_amendify import preview_skill_amendify

        result = asyncio.run(
            preview_skill_amendify(inspection=SAMPLE_INSPECTION, skill=SAMPLE_SKILL)
        )

        assert result.status == "proposed"
        assert result.skill_id == "test-skill"
        assert result.inspection_id == "insp-001"
        assert "edge case" in result.change_explanation
        assert result.amendment_confidence == 0.80
        mock_add_dp.assert_called_once()

    @patch("cognee.cognee_skills.preview_amendify.add_data_points", new_callable=AsyncMock)
    @patch("cognee.cognee_skills.preview_amendify.get_llm_config")
    @patch(
        "cognee.cognee_skills.preview_amendify.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
    )
    def test_preserves_original_instructions(self, mock_llm, mock_config, mock_add_dp):
        mock_llm.return_value = MOCK_AMENDMENT_PROPOSAL
        mock_config.return_value = MagicMock(llm_model="openai/gpt-4o-mini")

        from cognee.cognee_skills.preview_amendify import preview_skill_amendify

        result = asyncio.run(
            preview_skill_amendify(inspection=SAMPLE_INSPECTION, skill=SAMPLE_SKILL)
        )

        assert result.original_instructions == SAMPLE_SKILL["instructions"]
        assert result.amended_instructions != result.original_instructions
        assert result.pre_amendment_avg_score == SAMPLE_INSPECTION.avg_success_score


if __name__ == "__main__":
    unittest.main()
