"""Smoke test: the unified skill ingestion path through cognee.remember.

Covers the full promise of the ``graphskills-on-agentic`` branch in one
run — no LLM calls, no real graph engine, just code wiring:

1. ``cognee.remember("<skills-dir>/")`` accepts a SKILL.md directory
   and persists a canonical ``cognee.modules.engine.models.Skill``.
2. Re-running the same call is idempotent — content-hash diff
   against the graph skips unchanged skills.
3. Editing a skill triggers an ``"updated"`` ``SkillChangeEvent``.
4. The memify skill-improvement task
   (``cognee.modules.memify.skill_improvement.improve_failing_skills``)
   imports cleanly.
5. ``SearchType.AGENTIC_COMPLETION`` — the PR #2676 agentic retriever
   entrypoint — still resolves.
6. ``cognee.skills`` top-level attribute is gone (reshape contract).

What this DOESN'T cover (requires LLM + real DB — integration tests):
    * ``enrich=True`` LLM enrichment round-trip.
    * ``improve=True`` running inspect/amendify end-to-end.
    * Agentic retriever routing + SkillRun recording.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


_SKILL_V1 = """\
---
name: summarize
description: Summarize text into bullet points.
allowed-tools: memory_search
---
# Instructions

Condense the input into 2-3 key bullet points.
"""

_SKILL_V2 = """\
---
name: summarize
description: Summarize text into bullet points.
allowed-tools: memory_search
---
# Instructions

Condense the input into 3-5 key bullet points, prioritising decisions over
descriptions.
"""


def _make_skills_dir(body: str, entry_name: str = "SKILL.md"):
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    skill_dir = tmp / "summarize"
    skill_dir.mkdir()
    (skill_dir / entry_name).write_text(body)
    return tmp


class TestSkillIngest(unittest.TestCase):
    """End-to-end wiring test of ``cognee.remember`` → Skill DataPoint."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_remember_detects_skill_source(self):
        from cognee.modules.tools import looks_like_skill_source

        folder = _make_skills_dir(_SKILL_V1)
        try:
            assert looks_like_skill_source(str(folder))
        finally:
            import shutil

            shutil.rmtree(folder)

    def test_remember_detects_mixed_case_skill_source(self):
        from cognee.modules.tools import looks_like_skill_source

        folder = _make_skills_dir(_SKILL_V1, entry_name="Skill.md")
        try:
            assert looks_like_skill_source(str(folder))
        finally:
            import shutil

            shutil.rmtree(folder)

    def test_add_skills_persists_canonical_skill(self):
        from cognee.modules.tools.ingest_skills import add_skills

        folder = _make_skills_dir(_SKILL_V1)
        try:

            async def _run():
                with (
                    patch(
                        "cognee.modules.tools.ingest_skills._diff_against_graph",
                        new_callable=AsyncMock,
                    ) as mock_diff,
                    patch(
                        "cognee.modules.tools.ingest_skills.add_data_points",
                        new_callable=AsyncMock,
                    ) as mock_add,
                ):

                    async def diff_side_effect(parsed, node_set, dataset_id=None):
                        return (parsed, [], [])

                    mock_diff.side_effect = diff_side_effect
                    persisted = await add_skills(str(folder), enrich=False)
                    return persisted, mock_add

            persisted, mock_add = self._run(_run())

            assert len(persisted) == 1
            skill = persisted[0]
            assert skill.name == "summarize"
            assert "Condense the input into 2-3 key bullet points" in skill.procedure
            assert skill.content_hash
            assert skill.source_path.endswith("summarize")

            from cognee.modules.engine.models import Skill as CanonicalSkill

            assert isinstance(skill, CanonicalSkill)
            assert mock_add.await_count == 1
        finally:
            import shutil

            shutil.rmtree(folder)

    def test_add_skills_scopes_persisted_skill_to_dataset(self):
        from cognee.modules.tools.ingest_skills import add_skills

        folder = _make_skills_dir(_SKILL_V1)
        dataset = SimpleNamespace(id=uuid4(), name="hackathon")
        user = SimpleNamespace(id=uuid4())
        try:

            async def _run():
                with (
                    patch(
                        "cognee.modules.tools.ingest_skills._diff_against_graph",
                        new_callable=AsyncMock,
                    ) as mock_diff,
                    patch(
                        "cognee.modules.tools.ingest_skills.add_data_points",
                        new_callable=AsyncMock,
                    ) as mock_add,
                ):

                    async def diff_side_effect(parsed, node_set, dataset_id=None):
                        assert dataset_id == dataset.id
                        return (parsed, [], [])

                    mock_diff.side_effect = diff_side_effect
                    persisted = await add_skills(
                        str(folder),
                        enrich=False,
                        user=user,
                        dataset=dataset,
                    )
                    return persisted, mock_add

            persisted, mock_add = self._run(_run())

            assert persisted[0].dataset_scope == [str(dataset.id)]
            assert mock_add.await_count == 1
            ctx = mock_add.await_args.kwargs["ctx"]
            assert ctx.user is user
            assert ctx.dataset is dataset
            assert ctx.data_item.id
        finally:
            import shutil

            shutil.rmtree(folder)

    def test_re_ingest_is_idempotent_when_content_unchanged(self):
        from cognee.modules.tools.ingest_skills import add_skills
        from cognee.modules.tools.skill_parser import parse_skills_folder

        folder = _make_skills_dir(_SKILL_V1)
        try:
            parsed = parse_skills_folder(folder)
            existing_hash = parsed[0].content_hash
            existing_nid = str(parsed[0].id)
            fake_node = (
                existing_nid,
                {
                    "type": "Skill",
                    "name": "summarize",
                    "content_hash": existing_hash,
                },
            )
            mock_engine = AsyncMock()
            mock_engine.get_nodeset_subgraph = AsyncMock(return_value=([fake_node], []))

            async def _run():
                with (
                    patch(
                        "cognee.infrastructure.databases.graph.get_graph_engine",
                        new_callable=AsyncMock,
                        return_value=mock_engine,
                    ),
                    patch(
                        "cognee.modules.tools.ingest_skills.add_data_points",
                        new_callable=AsyncMock,
                    ) as mock_add,
                ):
                    persisted = await add_skills(str(folder), enrich=False)
                    return persisted, mock_add

            persisted, mock_add = self._run(_run())
            assert persisted == []
            assert mock_add.await_count == 0
        finally:
            import shutil

            shutil.rmtree(folder)

    def test_edit_triggers_updated_change_event(self):
        from cognee.modules.tools.ingest_skills import add_skills
        from cognee.modules.tools.skill_parser import parse_skills_folder

        folder = _make_skills_dir(_SKILL_V1)
        try:
            parsed_v1 = parse_skills_folder(folder)
            old_hash = parsed_v1[0].content_hash
            old_nid = str(parsed_v1[0].id)

            (folder / "summarize" / "SKILL.md").write_text(_SKILL_V2)

            fake_node = (
                old_nid,
                {
                    "type": "Skill",
                    "name": "summarize",
                    "content_hash": old_hash,
                },
            )
            mock_engine = AsyncMock()
            mock_engine.get_nodeset_subgraph = AsyncMock(return_value=([fake_node], []))
            mock_engine.delete_nodes = AsyncMock()
            mock_vector_engine = MagicMock()
            mock_vector_engine.delete_data_points = AsyncMock()

            async def _run():
                with (
                    patch(
                        "cognee.infrastructure.databases.graph.get_graph_engine",
                        new_callable=AsyncMock,
                        return_value=mock_engine,
                    ),
                    patch(
                        "cognee.infrastructure.databases.vector.get_vector_engine",
                        return_value=mock_vector_engine,
                    ),
                    patch(
                        "cognee.modules.tools.ingest_skills.add_data_points",
                        new_callable=AsyncMock,
                    ) as mock_add,
                ):
                    persisted = await add_skills(str(folder), enrich=False)
                    return persisted, mock_add, mock_engine

            persisted, mock_add, engine = self._run(_run())

            assert len(persisted) == 1
            assert persisted[0].content_hash != old_hash
            assert engine.delete_nodes.await_count == 1
            assert old_nid in engine.delete_nodes.await_args.args[0]
            assert mock_add.await_count >= 2

            flat = [item for call in mock_add.await_args_list for item in call.args[0]]
            change_events = [x for x in flat if type(x).__name__ == "SkillChangeEvent"]
            assert any(e.change_type == "updated" for e in change_events)
        finally:
            import shutil

            shutil.rmtree(folder)

    def test_top_level_skills_attr_is_gone(self):
        """The reshape deletes cognee.skills — only cognee.remember / memify remain."""
        import cognee

        assert not hasattr(cognee, "skills"), (
            "cognee.skills should have been removed by the graphskills reshape"
        )

    def test_agentic_completion_still_present(self):
        from cognee.api.v1.search import SearchType

        assert any("AGENTIC" in t.name for t in SearchType)

    def test_memify_improvement_task_importable(self):
        from cognee.modules.memify.skill_improvement import improve_failing_skills

        assert callable(improve_failing_skills)

    def test_agentic_skill_runs_use_neutral_success_score(self):
        from cognee.modules.engine.models import Skill
        from cognee.modules.engine.models.SkillRun import UNSCORED_SKILL_RUN_SCORE
        from cognee.modules.retrieval.agentic_retriever import AgenticRetriever

        skill = Skill(name="summarize", description="Summarize text.")

        async def _run():
            with patch(
                "cognee.tasks.storage.add_data_points.add_data_points",
                new_callable=AsyncMock,
            ) as mock_add:
                await AgenticRetriever._record_skill_runs([skill], "summarize this", "done")
                return mock_add.await_args.args[0][0]

        run = self._run(_run())

        assert run.success_score == UNSCORED_SKILL_RUN_SCORE

    def test_skill_run_entry_validates_score_ranges(self):
        from pydantic import ValidationError

        from cognee.memory import SkillRunEntry

        with self.assertRaises(ValidationError):
            SkillRunEntry(selected_skill_id="code-review", success_score=1.1)

        with self.assertRaises(ValidationError):
            SkillRunEntry(selected_skill_id="code-review", feedback=-1.1)

    def test_remember_skill_run_entry_persists_with_dataset_context(self):
        from cognee.memory import SkillRunEntry
        from cognee.modules.engine.models import Skill
        from cognee.modules.tools.skill_runs import remember_skill_run_entry

        dataset = SimpleNamespace(id=uuid4(), name="hackathon")
        user = SimpleNamespace(id=uuid4())
        entry = SkillRunEntry(
            run_id="review-run-1",
            selected_skill_id="code-review",
            task_text="Review the permission changes",
            result_summary="Missing dataset check.",
            success_score=0.2,
            feedback=-0.6,
            error_type="permission_gap",
            error_message="The review missed dataset ownership.",
        )

        async def _run():
            with (
                patch("cognee.modules.tools.skill_runs.setup", new_callable=AsyncMock),
                patch(
                    "cognee.modules.tools.skill_runs.resolve_authorized_user_datasets",
                    new_callable=AsyncMock,
                    return_value=(user, [dataset]),
                ),
                patch(
                    "cognee.modules.tools.skill_runs.resolve_skills",
                    new_callable=AsyncMock,
                    return_value=[Skill(name="code-review", description="Review code changes.")],
                ),
                patch(
                    "cognee.modules.tools.skill_runs.add_data_points",
                    new_callable=AsyncMock,
                ) as mock_add,
            ):
                run, resolved_dataset, applied = await remember_skill_run_entry(
                    entry,
                    dataset_name="hackathon",
                    session_id="agent-review",
                    user=user,
                )
                return run, resolved_dataset, applied, mock_add

        run, resolved_dataset, applied, mock_add = self._run(_run())

        assert resolved_dataset is dataset
        assert applied == []
        assert run.run_id == "review-run-1"
        assert run.selected_skill_id == "code-review"
        assert run.success_score == 0.2
        assert run.feedback == -0.6
        assert run.error_type == "permission_gap"
        assert run.session_id == "agent-review"
        assert run.belongs_to_set[0].name == "skills"
        assert mock_add.await_count == 1
        ctx = mock_add.await_args.kwargs["ctx"]
        assert ctx.user is user
        assert ctx.dataset is dataset

    def test_resolve_skills_skips_explicit_skill_outside_dataset_scope(self):
        from cognee.modules.engine.models import Skill
        from cognee.modules.tools.resolve_skills import resolve_skills

        current_dataset_id = uuid4()
        other_dataset_id = uuid4()
        skill = Skill(
            name="scoped-summarize",
            description="Summarize in a scoped dataset.",
            dataset_scope=[str(other_dataset_id)],
        )

        resolved = self._run(resolve_skills([skill], dataset_id=current_dataset_id))

        assert resolved == []

    def test_graph_tools_are_hidden_without_matching_dataset_scope(self):
        from cognee.modules.engine.models import Tool
        from cognee.modules.tools.registry import _query_tool_nodes

        dataset_id = uuid4()
        scoped_tool = Tool(
            name="dataset_tool",
            description="Dataset-scoped tool.",
            handler_ref="module:handler",
            dataset_id=dataset_id,
        )
        global_tool = Tool(
            name="global_tool",
            description="Global tool.",
            handler_ref="module:handler",
        )
        mock_engine = AsyncMock()
        mock_engine.get_nodes_by_type = AsyncMock(
            return_value=[scoped_tool.model_dump(), global_tool.model_dump()]
        )

        async def _run(dataset_scope):
            with patch(
                "cognee.infrastructure.databases.graph.get_graph_engine",
                new_callable=AsyncMock,
                return_value=mock_engine,
            ):
                return await _query_tool_nodes(dataset_id=dataset_scope)

        unscoped = self._run(_run(None))
        scoped = self._run(_run(dataset_id))

        assert [tool.name for tool in unscoped] == ["global_tool"]
        assert {tool.name for tool in scoped} == {"global_tool", "dataset_tool"}

    def test_memory_search_rejects_non_positive_top_k(self):
        from cognee.modules.tools.builtin.memory_search import handler
        from cognee.modules.tools.errors import ToolInvocationError

        with self.assertRaisesRegex(ToolInvocationError, "greater than 0"):
            self._run(handler({"query": "summaries", "top_k": 0}))
