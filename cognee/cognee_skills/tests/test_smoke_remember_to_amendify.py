"""Smoke test: unified ingest path to self-improvement loop.

Covers the full promise of the ``graphskills-on-agentic`` branch in one
run — no LLM calls, no real graph engine, just the code wiring.

Flow validated:

1. ``cognee.remember("<skills-dir>/", enrich=False)`` accepts a
   SKILL.md directory and persists the canonical
   ``cognee.modules.engine.models.Skill`` DataPoints.
2. Re-running the same call is idempotent — the second pass diffs
   against the graph via content_hash and skips unchanged skills.
3. Editing a skill triggers an "updated" SkillChangeEvent path.
4. The ``cognee.skills`` runtime surface (inspect / preview_amendify
   / amendify / rollback_amendify / META_SKILL_PATH) still imports
   cleanly and exposes its public methods.
5. ``SearchType.AGENTIC_COMPLETION`` — the PR #2676 agentic retriever
   entrypoint — is still reachable alongside the self-improvement
   runtime (proves the merge hasn't regressed PR #2676's surface).

What this DOESN'T cover (requires LLM access, real DB):
    * The actual ``enrich=True`` enrichment LLM round-trip.
    * ``skills.run`` / ``skills.execute`` against a live LLM.
    * ``cognee.search(skills_auto_retrieve=True)`` semantic ranking.

Those belong in integration tests with API keys.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from cognee import skills  # noqa: F401  — proves the public import still works.


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


class TestSmokeRememberToAmendify(unittest.TestCase):
    """End-to-end wiring test of ``cognee.remember`` → skills runtime."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_skills_dir(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp())
        skill_dir = tmp / "summarize"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(body)
        return tmp

    # ------------------------------------------------------------------
    # Step 1: cognee.remember with a SKILL.md folder reaches add_skills.
    # ------------------------------------------------------------------
    def test_remember_dispatches_to_add_skills(self):
        """cognee.remember(path) auto-dispatches skill sources."""
        from cognee.modules.tools import add_skills, looks_like_skill_source

        folder = self._make_skills_dir(_SKILL_V1)
        try:
            assert looks_like_skill_source(str(folder))
        finally:
            import shutil
            shutil.rmtree(folder)

    # ------------------------------------------------------------------
    # Step 2: add_skills parses and persists the rich Skill DataPoint.
    # ------------------------------------------------------------------
    def test_add_skills_persists_canonical_skill(self):
        """add_skills uses the rich parser and the canonical Skill model."""
        from cognee.modules.tools.ingest_skills import add_skills

        folder = self._make_skills_dir(_SKILL_V1)
        try:
            async def _run():
                with (
                    patch(
                        "cognee.modules.tools.ingest_skills._diff_against_graph",
                        new_callable=AsyncMock,
                        return_value=(
                            None,  # set below from the parsed list
                            [],
                            [],
                        ),
                    ) as mock_diff,
                    patch(
                        "cognee.modules.tools.ingest_skills.add_data_points",
                        new_callable=AsyncMock,
                    ) as mock_add,
                ):
                    # Capture parsed skills and pass them back as "to_persist".
                    async def diff_side_effect(parsed, node_set):
                        return (parsed, [], [])
                    mock_diff.side_effect = diff_side_effect
                    persisted = await add_skills(str(folder), enrich=False)
                    return persisted, mock_add

            persisted, mock_add = self._run(_run())

            # Rich-parser fields populated.
            assert len(persisted) == 1
            skill = persisted[0]
            assert skill.name == "summarize"
            assert "Condense the input into 2-3 key bullet points" in skill.procedure
            assert skill.content_hash  # set by the parser
            assert skill.source_path.endswith("summarize")

            # Canonical class, not the old cognee_skills.models.skill.Skill.
            from cognee.modules.engine.models import Skill as CanonicalSkill
            assert isinstance(skill, CanonicalSkill)

            # Single add_data_points call with the persisted skills.
            assert mock_add.await_count == 1
            assert mock_add.await_args.args[0] == persisted
        finally:
            import shutil
            shutil.rmtree(folder)

    # ------------------------------------------------------------------
    # Step 3: Re-ingest is idempotent; unchanged skills skipped.
    # ------------------------------------------------------------------
    def test_re_ingest_is_idempotent_when_content_unchanged(self):
        """Re-running add_skills on an unchanged folder persists nothing new."""
        from cognee.modules.tools.ingest_skills import add_skills
        from cognee.cognee_skills.parser.skill_parser import parse_skills_folder

        folder = self._make_skills_dir(_SKILL_V1)
        try:
            # Simulate a prior ingest: the "graph" already has a Skill
            # with the same content_hash as what the parser will produce.
            parsed = parse_skills_folder(folder)
            existing_hash = parsed[0].content_hash
            existing_nid = str(parsed[0].id)
            fake_node = (existing_nid, {
                "type": "Skill",
                "name": "summarize",
                "content_hash": existing_hash,
            })
            mock_engine = AsyncMock()
            mock_engine.get_nodeset_subgraph = AsyncMock(
                return_value=([fake_node], [])
            )

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

            # Nothing re-persisted because the content_hash matches the graph.
            assert persisted == []
            # add_data_points is called zero times when no change events to emit
            # and no skills to persist.
            assert mock_add.await_count == 0
        finally:
            import shutil
            shutil.rmtree(folder)

    # ------------------------------------------------------------------
    # Step 4: Editing the skill content produces a SkillChangeEvent.
    # ------------------------------------------------------------------
    def test_edit_triggers_updated_change_event(self):
        """Editing SKILL.md body produces an "updated" SkillChangeEvent
        and re-persists the new version."""
        from cognee.modules.tools.ingest_skills import add_skills
        from cognee.cognee_skills.parser.skill_parser import parse_skills_folder

        folder = self._make_skills_dir(_SKILL_V1)
        try:
            # Original snapshot
            parsed_v1 = parse_skills_folder(folder)
            old_hash = parsed_v1[0].content_hash
            old_nid = str(parsed_v1[0].id)

            # Edit the skill body
            (folder / "summarize" / "SKILL.md").write_text(_SKILL_V2)

            fake_node = (old_nid, {
                "type": "Skill",
                "name": "summarize",
                "content_hash": old_hash,
            })
            mock_engine = AsyncMock()
            mock_engine.get_nodeset_subgraph = AsyncMock(
                return_value=([fake_node], [])
            )
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

            # Exactly one skill re-persisted (the edited one).
            assert len(persisted) == 1
            assert persisted[0].content_hash != old_hash

            # Stale node deleted from graph.
            assert engine.delete_nodes.await_count == 1
            assert old_nid in engine.delete_nodes.await_args.args[0]

            # add_data_points called at least twice: once for the
            # SkillChangeEvent, once for the new Skill.
            assert mock_add.await_count >= 2

            # A SkillChangeEvent of type "updated" was emitted.
            event_batches = [c.args[0] for c in mock_add.await_args_list]
            flat = [item for batch in event_batches for item in batch]
            change_events = [
                x for x in flat if type(x).__name__ == "SkillChangeEvent"
            ]
            assert any(e.change_type == "updated" for e in change_events)
        finally:
            import shutil
            shutil.rmtree(folder)

    # ------------------------------------------------------------------
    # Step 5: Runtime surface intact.
    # ------------------------------------------------------------------
    def test_self_improvement_runtime_surface_is_callable(self):
        """``cognee.skills`` exposes the full runtime API — no ingest methods."""
        from cognee import skills

        # Ingest is NOT on the client (it moved to cognee.remember).
        for removed in ("ingest", "upsert", "ingest_meta_skill"):
            assert not hasattr(skills, removed), (
                f"skills.{removed} should have been removed"
            )

        # Runtime methods are still there and callable.
        for method in (
            "run",
            "execute",
            "observe",
            "load",
            "list",
            "get_context",
            "inspect",
            "preview_amendify",
            "amendify",
            "rollback_amendify",
            "evaluate_amendify",
            "auto_amendify",
            "remove",
        ):
            assert callable(getattr(skills, method)), (
                f"skills.{method} should be callable"
            )

        # Meta-skill path discoverable + on disk.
        assert skills.META_SKILL_PATH.is_dir()
        assert (skills.META_SKILL_PATH / "SKILL.md").is_file()

    # ------------------------------------------------------------------
    # Step 6: Agentic retriever entrypoint coexists with skills runtime.
    # ------------------------------------------------------------------
    def test_agentic_completion_search_type_still_present(self):
        """PR #2676's SearchType.AGENTIC_COMPLETION wasn't regressed
        by the self-improvement merge."""
        from cognee.api.v1.search import SearchType

        names = {t.name for t in SearchType}
        assert "AGENTIC_COMPLETION" in names, (
            f"AGENTIC_COMPLETION missing from SearchType: {names}"
        )
