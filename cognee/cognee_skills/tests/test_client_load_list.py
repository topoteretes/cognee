"""Tests for Skills.load(), Skills.list(), and the happy-path execute loop.

Happy path: execute() succeeds → evaluate scores quality → observe records score →
auto_amendify is NOT triggered when quality is above threshold.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SKILL_ID = "summarize"
SKILL_NID = uuid4()
TP_NID = uuid4()
OTHER_NID = uuid4()

_SKILL_PROPS = {
    "type": "Skill",
    "skill_id": SKILL_ID,
    "name": "Summarize",
    "description": "Condense any text into bullet points.",
    "instructions": "Condense the input into 2-3 key bullet points.",
    "instruction_summary": "Summarizes text into bullet points.",
    "tags": ["context-management"],
    "complexity": "simple",
    "source_path": "/skills/summarize/SKILL.md",
}

_TP_PROPS = {
    "type": "TaskPattern",
    "pattern_key": "summarize:compress-text",
    "text": "Summarize this document",
    "category": "text-processing",
}

_OTHER_SKILL_PROPS = {
    "type": "Skill",
    "skill_id": "code-review",
    "name": "Code Review",
    "description": "Review code for bugs.",
    "instructions": "Check for bugs, style, and security issues.",
    "instruction_summary": "Reviews code for quality.",
    "tags": ["engineering"],
    "complexity": "workflow",
    "source_path": "/skills/code-review/SKILL.md",
}

MOCK_EVALUATION = {"score": 0.85, "reason": "Good summary"}


def _make_engine(nodes, edges=None):
    engine = AsyncMock()
    engine.get_nodeset_subgraph = AsyncMock(return_value=(nodes, edges or []))
    return engine


# ---------------------------------------------------------------------------
# Skills.load()
# ---------------------------------------------------------------------------


class TestSkillsLoad(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_load_returns_full_skill_dict(self):
        """load() returns all expected fields for an existing skill."""
        from cognee.cognee_skills.client import Skills

        nodes = [(SKILL_NID, _SKILL_PROPS)]
        engine = _make_engine(nodes)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().load(SKILL_ID)

        result = self._run(_go())

        assert result is not None
        assert result["skill_id"] == SKILL_ID
        assert result["name"] == "Summarize"
        assert result["instructions"] == _SKILL_PROPS["instructions"]
        assert result["instruction_summary"] == _SKILL_PROPS["instruction_summary"]
        assert result["description"] == _SKILL_PROPS["description"]
        assert result["tags"] == ["context-management"]
        assert result["complexity"] == "simple"
        assert result["source_path"] == _SKILL_PROPS["source_path"]
        assert result["task_patterns"] == []

    def test_load_returns_none_for_unknown_skill(self):
        """load() returns None when skill_id does not exist in the graph."""
        from cognee.cognee_skills.client import Skills

        nodes = [(SKILL_NID, _SKILL_PROPS)]
        engine = _make_engine(nodes)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().load("nonexistent-skill")

        result = self._run(_go())
        assert result is None

    def test_load_resolves_task_patterns_from_solves_edges(self):
        """load() walks 'solves' edges to attach TaskPattern dicts."""
        from cognee.cognee_skills.client import Skills

        nodes = [
            (SKILL_NID, _SKILL_PROPS),
            (TP_NID, _TP_PROPS),
        ]
        # Skill -solves-> TaskPattern
        edges = [(SKILL_NID, TP_NID, "solves", {})]
        engine = _make_engine(nodes, edges)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().load(SKILL_ID)

        result = self._run(_go())

        assert len(result["task_patterns"]) == 1
        tp = result["task_patterns"][0]
        assert tp["pattern_key"] == "summarize:compress-text"
        assert tp["text"] == "Summarize this document"
        assert tp["category"] == "text-processing"

    def test_load_ignores_unrelated_edges(self):
        """load() only follows 'solves' edges; other edge types are ignored."""
        from cognee.cognee_skills.client import Skills

        nodes = [
            (SKILL_NID, _SKILL_PROPS),
            (TP_NID, _TP_PROPS),
        ]
        # Wrong relationship name — should not be resolved
        edges = [(SKILL_NID, TP_NID, "related_to", {})]
        engine = _make_engine(nodes, edges)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().load(SKILL_ID)

        result = self._run(_go())
        assert result["task_patterns"] == []

    def test_load_ignores_solves_edges_from_other_skills(self):
        """load() only resolves patterns linked from the requested skill node."""
        from cognee.cognee_skills.client import Skills

        nodes = [
            (SKILL_NID, _SKILL_PROPS),
            (OTHER_NID, _OTHER_SKILL_PROPS),
            (TP_NID, _TP_PROPS),
        ]
        # Edge from OTHER skill, not from SKILL_NID
        edges = [(OTHER_NID, TP_NID, "solves", {})]
        engine = _make_engine(nodes, edges)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().load(SKILL_ID)

        result = self._run(_go())
        assert result["task_patterns"] == []

    def test_load_multiple_patterns(self):
        """load() returns all TaskPatterns linked via 'solves' edges."""
        from cognee.cognee_skills.client import Skills

        tp2_nid = uuid4()
        tp2_props = {
            "type": "TaskPattern",
            "pattern_key": "summarize:tldr",
            "text": "Give me a TL;DR",
            "category": "text-processing",
        }
        nodes = [
            (SKILL_NID, _SKILL_PROPS),
            (TP_NID, _TP_PROPS),
            (tp2_nid, tp2_props),
        ]
        edges = [
            (SKILL_NID, TP_NID, "solves", {}),
            (SKILL_NID, tp2_nid, "solves", {}),
        ]
        engine = _make_engine(nodes, edges)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().load(SKILL_ID)

        result = self._run(_go())
        assert len(result["task_patterns"]) == 2
        keys = {tp["pattern_key"] for tp in result["task_patterns"]}
        assert keys == {"summarize:compress-text", "summarize:tldr"}


# ---------------------------------------------------------------------------
# Skills.list()
# ---------------------------------------------------------------------------


class TestSkillsList(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_list_returns_all_skills(self):
        """list() returns one summary dict per Skill node."""
        from cognee.cognee_skills.client import Skills

        nodes = [
            (SKILL_NID, _SKILL_PROPS),
            (OTHER_NID, _OTHER_SKILL_PROPS),
        ]
        engine = _make_engine(nodes)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().list()

        result = self._run(_go())

        assert len(result) == 2
        ids = {s["skill_id"] for s in result}
        assert ids == {SKILL_ID, "code-review"}

    def test_list_summary_fields_only(self):
        """list() includes only summary fields — no full instructions."""
        from cognee.cognee_skills.client import Skills

        nodes = [(SKILL_NID, _SKILL_PROPS)]
        engine = _make_engine(nodes)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().list()

        result = self._run(_go())

        assert len(result) == 1
        skill = result[0]
        assert skill["skill_id"] == SKILL_ID
        assert skill["name"] == "Summarize"
        assert skill["instruction_summary"] == _SKILL_PROPS["instruction_summary"]
        assert skill["tags"] == ["context-management"]
        assert skill["complexity"] == "simple"
        # Full instructions must NOT be included in list output
        assert "instructions" not in skill

    def test_list_skips_non_skill_nodes(self):
        """list() ignores TaskPattern, SkillRun, and other node types."""
        from cognee.cognee_skills.client import Skills

        nodes = [
            (SKILL_NID, _SKILL_PROPS),
            (TP_NID, _TP_PROPS),  # TaskPattern — should be skipped
            (
                uuid4(),
                {
                    "type": "SkillRun",
                    "run_id": "r1",
                    "selected_skill_id": SKILL_ID,
                    "success_score": 0.9,
                },
            ),
        ]
        engine = _make_engine(nodes)

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().list()

        result = self._run(_go())
        assert len(result) == 1
        assert result[0]["skill_id"] == SKILL_ID

    def test_list_empty_when_no_skills(self):
        """list() returns [] when the graph contains no Skill nodes."""
        from cognee.cognee_skills.client import Skills

        engine = _make_engine([])

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().list()

        result = self._run(_go())
        assert result == []

    def test_list_returns_empty_on_graph_error(self):
        """list() swallows graph exceptions and returns []."""
        from cognee.cognee_skills.client import Skills

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(side_effect=RuntimeError("DB unavailable"))

        async def _go():
            with patch(
                "cognee.cognee_skills.client.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ):
                return await Skills().list()

        result = self._run(_go())
        assert result == []


# ---------------------------------------------------------------------------
# Happy-path execute: success → evaluate quality → observe → no auto_amendify
# ---------------------------------------------------------------------------


def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_EVAL_PATCH = patch(
    "cognee.cognee_skills.client.evaluate_output",
    new_callable=AsyncMock,
    return_value=MOCK_EVALUATION,
)


class TestHappyPathExecute(unittest.TestCase):
    """execute() success path: LLM produces output, quality is evaluated,
    run is observed with the quality score, amendify is never called."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_execute_success_returns_output_with_quality(self):
        """On success, execute() returns output and quality_score."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        async def _go():
            with (
                patch.object(client, "load", new_callable=AsyncMock, return_value={
                    "skill_id": SKILL_ID,
                    "name": "Summarize",
                    "instructions": "Condense the input into 2-3 key bullet points.",
                    "instruction_summary": "Summarizes text into bullet points.",
                    "description": "Condense any text.",
                    "tags": ["context-management"],
                    "complexity": "simple",
                    "source_path": "",
                    "task_patterns": [],
                }),
                patch.object(client, "_resolve_pattern", new_callable=AsyncMock, return_value=""),
                patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
                patch(
                    "cognee.cognee_skills.execute.litellm.acompletion",
                    new_callable=AsyncMock,
                    return_value=_make_llm_response("- Point 1\n- Point 2\n- Point 3"),
                ),
                _EVAL_PATCH,
                patch.object(client, "observe", new_callable=AsyncMock) as mock_observe,
            ):
                mock_cfg.return_value = MagicMock(
                    llm_model="openai/gpt-4o-mini", llm_api_key="test"
                )
                return await client.execute(SKILL_ID, "Summarize this document"), mock_observe

        result, mock_observe = self._run(_go())

        assert result["success"] is True
        assert result["output"] == "- Point 1\n- Point 2\n- Point 3"
        assert result["skill_id"] == SKILL_ID
        assert result["quality_score"] == 0.85
        assert result["error"] is None

    def test_execute_success_observes_quality_score(self):
        """On success, auto_observe records quality_score (not binary 1.0)."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        async def _go():
            with (
                patch.object(client, "load", new_callable=AsyncMock, return_value={
                    "skill_id": SKILL_ID,
                    "name": "Summarize",
                    "instructions": "Condense the input.",
                    "instruction_summary": "",
                    "description": "",
                    "tags": [],
                    "complexity": "simple",
                    "source_path": "",
                    "task_patterns": [],
                }),
                patch.object(
                    client, "_resolve_pattern",
                    new_callable=AsyncMock,
                    return_value="summarize:compress-text",
                ),
                patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
                patch(
                    "cognee.cognee_skills.execute.litellm.acompletion",
                    new_callable=AsyncMock,
                    return_value=_make_llm_response("Summary here"),
                ),
                _EVAL_PATCH,
                patch.object(client, "observe", new_callable=AsyncMock) as mock_observe,
            ):
                mock_cfg.return_value = MagicMock(
                    llm_model="openai/gpt-4o-mini", llm_api_key="test"
                )
                await client.execute(SKILL_ID, "Summarize this", session_id="sess-42")
                return mock_observe

        mock_observe = self._run(_go())

        mock_observe.assert_called_once()
        obs_call = mock_observe.call_args[0][0]
        assert obs_call["success_score"] == 0.85  # quality score, not binary
        assert obs_call["selected_skill_id"] == SKILL_ID
        assert obs_call["session_id"] == "sess-42"
        assert obs_call["task_pattern_id"] == "summarize:compress-text"

    def test_execute_success_does_not_trigger_auto_amendify(self):
        """On high quality, auto_amendify is never called even when auto_amendify=True."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        async def _go():
            with (
                patch.object(client, "load", new_callable=AsyncMock, return_value={
                    "skill_id": SKILL_ID,
                    "name": "Summarize",
                    "instructions": "Condense the input.",
                    "instruction_summary": "",
                    "description": "",
                    "tags": [],
                    "complexity": "simple",
                    "source_path": "",
                    "task_patterns": [],
                }),
                patch.object(client, "_resolve_pattern", new_callable=AsyncMock, return_value=""),
                patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
                patch(
                    "cognee.cognee_skills.execute.litellm.acompletion",
                    new_callable=AsyncMock,
                    return_value=_make_llm_response("Summary here"),
                ),
                _EVAL_PATCH,
                patch.object(client, "observe", new_callable=AsyncMock),
                patch.object(client, "auto_amendify", new_callable=AsyncMock) as mock_aa,
            ):
                mock_cfg.return_value = MagicMock(
                    llm_model="openai/gpt-4o-mini", llm_api_key="test"
                )
                result = await client.execute(
                    SKILL_ID, "Summarize this", auto_amendify=True
                )
                return result, mock_aa

        result, mock_aa = self._run(_go())

        assert result["success"] is True
        assert "amended" not in result
        mock_aa.assert_not_called()

    def test_execute_success_no_observe_when_disabled(self):
        """auto_observe=False skips observation even on success."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        async def _go():
            with (
                patch.object(client, "load", new_callable=AsyncMock, return_value={
                    "skill_id": SKILL_ID,
                    "name": "Summarize",
                    "instructions": "Condense the input.",
                    "instruction_summary": "",
                    "description": "",
                    "tags": [],
                    "complexity": "simple",
                    "source_path": "",
                    "task_patterns": [],
                }),
                patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
                patch(
                    "cognee.cognee_skills.execute.litellm.acompletion",
                    new_callable=AsyncMock,
                    return_value=_make_llm_response("Summary here"),
                ),
                _EVAL_PATCH,
                patch.object(client, "observe", new_callable=AsyncMock) as mock_observe,
            ):
                mock_cfg.return_value = MagicMock(
                    llm_model="openai/gpt-4o-mini", llm_api_key="test"
                )
                await client.execute(SKILL_ID, "Summarize this", auto_observe=False)
                return mock_observe

        mock_observe = self._run(_go())
        mock_observe.assert_not_called()

    def test_execute_observe_captures_output_summary(self):
        """The observe call includes a truncated result_summary from the LLM output."""
        from cognee.cognee_skills.client import Skills

        client = Skills()
        long_output = "x" * 1000

        async def _go():
            with (
                patch.object(client, "load", new_callable=AsyncMock, return_value={
                    "skill_id": SKILL_ID,
                    "name": "Summarize",
                    "instructions": "Condense.",
                    "instruction_summary": "",
                    "description": "",
                    "tags": [],
                    "complexity": "simple",
                    "source_path": "",
                    "task_patterns": [],
                }),
                patch.object(client, "_resolve_pattern", new_callable=AsyncMock, return_value=""),
                patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
                patch(
                    "cognee.cognee_skills.execute.litellm.acompletion",
                    new_callable=AsyncMock,
                    return_value=_make_llm_response(long_output),
                ),
                _EVAL_PATCH,
                patch.object(client, "observe", new_callable=AsyncMock) as mock_observe,
            ):
                mock_cfg.return_value = MagicMock(
                    llm_model="openai/gpt-4o-mini", llm_api_key="test"
                )
                await client.execute(SKILL_ID, "Summarize this")
                return mock_observe

        mock_observe = self._run(_go())

        obs_call = mock_observe.call_args[0][0]
        assert len(obs_call["result_summary"]) <= 500
        assert obs_call["result_summary"] == long_output[:500]

    def test_happy_path_full_loop_load_execute_observe(self):
        """Full happy-path loop: load → execute → evaluate → observe.

        Uses a mock graph engine seeded with a Skill node. load() reads
        from the graph; execute_skill() calls the (mocked) LLM; evaluate
        scores quality; observe() persists the SkillRun with the quality score.
        """
        from cognee.cognee_skills.client import Skills

        nodes = [(SKILL_NID, _SKILL_PROPS)]
        engine = _make_engine(nodes, [])

        client = Skills()
        observed_runs = []

        async def _fake_observe(run_dict):
            observed_runs.append(run_dict)
            return run_dict

        async def _go():
            with (
                patch(
                    "cognee.cognee_skills.client.get_graph_engine",
                    new_callable=AsyncMock,
                    return_value=engine,
                ),
                patch.object(
                    client, "_resolve_pattern",
                    new_callable=AsyncMock,
                    return_value="summarize:compress-text",
                ),
                patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
                patch(
                    "cognee.cognee_skills.execute.litellm.acompletion",
                    new_callable=AsyncMock,
                    return_value=_make_llm_response("- Bullet 1\n- Bullet 2"),
                ),
                _EVAL_PATCH,
                patch.object(client, "observe", side_effect=_fake_observe),
            ):
                mock_cfg.return_value = MagicMock(
                    llm_model="openai/gpt-4o-mini", llm_api_key="test"
                )
                return await client.execute(
                    SKILL_ID,
                    "Summarize this article",
                    session_id="happy-sess",
                )

        result = self._run(_go())

        assert result["success"] is True
        assert "Bullet 1" in result["output"]
        assert result["skill_id"] == SKILL_ID
        assert result["quality_score"] == 0.85

        assert len(observed_runs) == 1
        obs = observed_runs[0]
        assert obs["selected_skill_id"] == SKILL_ID
        assert obs["success_score"] == 0.85  # quality score, not binary
        assert obs["session_id"] == "happy-sess"
        assert obs["task_text"] == "Summarize this article"
        assert obs["task_pattern_id"] == "summarize:compress-text"


# ---------------------------------------------------------------------------
# Skills.run() — one-call integration
# ---------------------------------------------------------------------------


class TestRun(unittest.TestCase):
    """skills.run(task_text) = get_context + execute + observe + auto_amendify."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_run_selects_top_skill_and_executes(self):
        """run() picks the highest-scored skill and returns its output."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        recs = [
            {"skill_id": SKILL_ID, "name": "Summarize", "score": 0.95,
             "vector_score": 0.9, "prefers_score": 0.05,
             "instruction_summary": "", "task_pattern_id": "", "tags": []},
        ]
        exec_result = {
            "output": "- Bullet 1", "skill_id": SKILL_ID,
            "model": "openai/gpt-4o-mini", "latency_ms": 100,
            "success": True, "error": None,
            "quality_score": 0.85, "quality_reason": "Good",
        }

        async def _go():
            with (
                patch.object(client, "get_context", new_callable=AsyncMock, return_value=recs),
                patch.object(client, "execute", new_callable=AsyncMock, return_value=exec_result),
            ):
                return await client.run("Summarize this document")

        result = self._run(_go())

        assert result["success"] is True
        assert result["output"] == "- Bullet 1"
        assert result["skill_id"] == SKILL_ID
        assert result["name"] == "Summarize"
        assert result["score"] == 0.95

    def test_run_passes_auto_evaluate_and_amendify_to_execute(self):
        """run() forwards auto_evaluate, auto_amendify, and amendify_min_runs to execute()."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        recs = [
            {"skill_id": SKILL_ID, "name": "Summarize", "score": 0.9,
             "vector_score": 0.9, "prefers_score": 0.0,
             "instruction_summary": "", "task_pattern_id": "", "tags": []},
        ]

        async def _go():
            with (
                patch.object(client, "get_context", new_callable=AsyncMock, return_value=recs),
                patch.object(client, "execute", new_callable=AsyncMock, return_value={
                    "output": "", "skill_id": SKILL_ID, "model": "",
                    "latency_ms": 0, "success": False, "error": "LLM error",
                }) as mock_exec,
            ):
                await client.run(
                    "Summarize this",
                    auto_evaluate=True,
                    auto_amendify=True,
                    amendify_min_runs=5,
                    session_id="sess-run",
                )
                return mock_exec

        mock_exec = self._run(_go())

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["auto_evaluate"] is True
        assert call_kwargs["auto_amendify"] is True
        assert call_kwargs["amendify_min_runs"] == 5
        assert call_kwargs["session_id"] == "sess-run"
        assert call_kwargs["auto_observe"] is True

    def test_run_returns_error_when_no_skills(self):
        """run() returns success=False when get_context finds nothing."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        async def _go():
            with patch.object(client, "get_context", new_callable=AsyncMock, return_value=[]):
                return await client.run("Do something")

        result = self._run(_go())

        assert result["success"] is False
        assert "No skills found" in result["error"]
        assert result["skill_id"] == ""

    def test_run_passes_context_to_execute(self):
        """run() forwards the context parameter to execute()."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        recs = [
            {"skill_id": SKILL_ID, "name": "Summarize", "score": 0.9,
             "vector_score": 0.9, "prefers_score": 0.0,
             "instruction_summary": "", "task_pattern_id": "", "tags": []},
        ]

        async def _go():
            with (
                patch.object(client, "get_context", new_callable=AsyncMock, return_value=recs),
                patch.object(client, "execute", new_callable=AsyncMock, return_value={
                    "output": "Done", "skill_id": SKILL_ID, "model": "",
                    "latency_ms": 0, "success": True, "error": None,
                }) as mock_exec,
            ):
                await client.run("Summarize this", context="The article is about AI.")
                return mock_exec

        mock_exec = self._run(_go())
        assert mock_exec.call_args[1]["context"] == "The article is about AI."


if __name__ == "__main__":
    unittest.main()
