"""Unit tests for skills data models — no LLM or database required."""

from uuid import uuid4

from cognee.skills.models.skill import Skill, SkillResource
from cognee.skills.models.skill_run import SkillRun, ToolCall, CandidateSkill
from cognee.skills.models.task_pattern import TaskPattern


class TestSkillModel:
    def test_minimal_skill(self):
        s = Skill(
            id=uuid4(),
            skill_id="test-skill",
            name="Test Skill",
            description="A test skill.",
            instructions="# Test\n\nDo the thing.",
        )
        assert s.skill_id == "test-skill"
        assert s.name == "Test Skill"
        assert s.complexity == ""
        assert s.tags == []
        assert s.is_active is True

    def test_skill_with_enrichment(self):
        s = Skill(
            id=uuid4(),
            skill_id="enriched",
            name="Enriched Skill",
            description="With LLM fields.",
            instructions="# Enriched",
            instruction_summary="A summary from LLM.",
            tags=["code", "testing"],
            complexity="workflow",
            task_pattern_candidates=["code_review", "linting"],
        )
        assert s.instruction_summary == "A summary from LLM."
        assert len(s.tags) == 2
        assert len(s.task_pattern_candidates) == 2


class TestSkillResourceModel:
    def test_resource(self):
        r = SkillResource(
            id=uuid4(),
            name="prompts.md",
            path="references/prompts.md",
            resource_type="reference",
            content="# Prompts",
        )
        assert r.resource_type == "reference"
        assert r.content_hash == ""


class TestSkillRunModel:
    def test_minimal_run(self):
        run = SkillRun(
            id=uuid4(),
            run_id="session-1:task:0",
            session_id="session-1",
            task_text="summarize this",
            selected_skill_id="summarize",
            success_score=0.85,
        )
        assert run.success_score == 0.85
        assert run.error_type == ""
        assert run.candidate_skills == []
        assert run.tool_trace == []

    def test_run_with_candidates(self):
        c = CandidateSkill(
            id=uuid4(),
            skill_id="summarize",
            score=0.92,
            signals={"vector": 0.9, "prefers": 0.1},
        )
        run = SkillRun(
            id=uuid4(),
            run_id="session-1:task:1",
            session_id="session-1",
            task_text="compress context",
            selected_skill_id="summarize",
            success_score=0.9,
            candidate_skills=[c],
        )
        assert len(run.candidate_skills) == 1
        assert run.candidate_skills[0].skill_id == "summarize"


class TestToolCallModel:
    def test_tool_call(self):
        tc = ToolCall(
            id=uuid4(),
            tool_name="read_file",
            tool_input={"path": "/tmp/test.txt"},
            tool_output="file contents",
            success=True,
            duration_ms=150,
        )
        assert tc.tool_name == "read_file"
        assert tc.success is True


class TestTaskPatternModel:
    def test_task_pattern(self):
        tp = TaskPattern(
            id=uuid4(),
            pattern_id="context_compression",
            name="context_compression",
            pattern_key="context_compression",
            text="Compress or reduce context window usage",
            category="context-management",
        )
        assert tp.pattern_key == "context_compression"
        assert tp.category == "context-management"
