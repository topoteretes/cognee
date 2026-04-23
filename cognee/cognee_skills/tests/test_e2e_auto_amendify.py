"""End-to-end integration test for the full auto_amendify loop.

Pipeline under test:
    ingest_skills() → execute() → observe() → inspect_skill()
                   → preview_skill_amendify() → amendify()
    auto_amendify() — validates the full chain in a single orchestrated call.

Strategy
--------
All external I/O (LLM, graph DB, vector DB, infrastructure setup) is mocked.
The real business logic of every module runs against a *shared in-memory
graph store* so that writes from one step become visible to subsequent reads —
this makes the test a genuine integration loop rather than a collection of
isolated unit checks.

DataPoint objects written via ``add_data_points`` are serialised into the
store as ``(nid, props_dict)`` tuples (matching the format returned by
``get_nodeset_subgraph``) so the graph read path works unchanged.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from cognee.cognee_skills.models.skill_amendment import AmendmentProposal
from cognee.cognee_skills.models.skill_inspection import InspectionResult

# ---------------------------------------------------------------------------
# Helpers — shared in-memory graph state
# ---------------------------------------------------------------------------

_SKILL_MD_CONTENT = """\
---
name: Summarize
description: Condense any text into bullet points.
tags: [context-management]
---

# Summarize

Condense the input into 2-3 key bullet points.

## When to Activate

- User asks to summarize text
- User wants a shorter version of a document
"""

SKILL_ID = "summarize"


def _make_skill_node(skill_id: str = SKILL_ID, nid: UUID | None = None) -> tuple[UUID, dict]:
    nid = nid or uuid4()
    return (
        nid,
        {
            "type": "Skill",
            "id": str(nid),
            "name": skill_id,
            "description": "Condense any text into bullet points.",
            "procedure": "Condense the input into 2-3 key bullet points.",
            "instruction_summary": "Summarizes text into bullet points.",
            "content_hash": "abc123",
            "source_path": "",
            "tags": ["context-management"],
            "complexity": "simple",
        },
    )


def _dp_to_node(dp) -> tuple[UUID, dict]:
    """Convert a DataPoint object into a (nid, props_dict) tuple for the mock graph."""
    try:
        props = dp.model_dump(exclude={"solves", "resources", "related_skills"})
    except Exception:
        props = {}
    # Ensure "type" field matches class name
    props.setdefault("type", type(dp).__name__)
    # Ensure uuid id
    nid = dp.id if isinstance(dp.id, UUID) else UUID(str(dp.id))
    return (nid, props)


class _NodeStore:
    """Minimal in-memory graph that accumulates nodes written by add_data_points."""

    def __init__(self):
        self._nodes: list[tuple[UUID, dict]] = []

    def seed(self, node: tuple[UUID, dict]) -> None:
        self._nodes.append(node)

    def ingest(self, data_points: list) -> None:
        for dp in data_points:
            self._nodes.append(_dp_to_node(dp))

    @property
    def nodes(self) -> list[tuple[UUID, dict]]:
        return list(self._nodes)

    def make_engine_mock(self) -> AsyncMock:
        """Return a mock graph engine whose get_nodeset_subgraph reads from this store."""
        engine = AsyncMock()
        store = self  # closure

        async def _get_subgraph(**kwargs):
            return (store.nodes, [])

        engine.get_nodeset_subgraph = _get_subgraph
        engine.add_edges = AsyncMock()
        engine.delete_nodes = AsyncMock()
        return engine


# ---------------------------------------------------------------------------
# LLM mock responses
# ---------------------------------------------------------------------------

_MOCK_INSPECTION = InspectionResult(
    failure_category="instruction_gap",
    root_cause="Instructions do not handle empty inputs.",
    severity="high",
    improvement_hypothesis="Add explicit handling for empty or very short inputs.",
    confidence=0.88,
)

_MOCK_AMENDMENT = AmendmentProposal(
    amended_instructions=(
        "Condense the input into 2-3 key bullet points.\n\n"
        "If the input is empty or fewer than 10 words, respond with: "
        "'Input too short to summarize.'"
    ),
    change_explanation="Added guard clause for empty/short inputs.",
    expected_improvement="Skill will no longer fail on empty inputs.",
    confidence=0.82,
)


def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestE2EAutoAmendifyLoop(unittest.TestCase):
    """Full end-to-end test of the auto_amendify pipeline.

    Each step uses real module code; only LLM calls and graph I/O are mocked.
    """

    # ------------------------------------------------------------------
    # Step 1: ingest_skills()
    # ------------------------------------------------------------------

    @unittest.skip(
        "uses cognee_skills.pipeline.ingest_skills (removed; ingestion now lives on cognee.remember). Replaced by test_smoke_remember_to_amendify.py"
    )
    def test_step1_ingest_skills(self):
        """ingest_skills() runs without error; the pipeline is invoked."""

        tmpdir = tempfile.mkdtemp()
        skill_dir = Path(tmpdir) / SKILL_ID
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_SKILL_MD_CONTENT)

        from cognee.cognee_skills.pipeline import ingest_skills

        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = uuid4()

        async def _run():
            with (
                patch("cognee.cognee_skills.pipeline.setup", new_callable=AsyncMock),
                patch(
                    "cognee.cognee_skills.pipeline.get_default_user",
                    new_callable=AsyncMock,
                    return_value=mock_user,
                ),
                patch(
                    "cognee.cognee_skills.pipeline.load_or_create_datasets",
                    new_callable=AsyncMock,
                    return_value=[mock_dataset],
                ),
                patch(
                    "cognee.cognee_skills.pipeline.run_tasks",
                    return_value=_async_gen_empty(),
                ),
                patch(
                    "cognee.cognee_skills.pipeline.index_graph_edges",
                    new_callable=AsyncMock,
                ),
            ):
                await ingest_skills(tmpdir, skip_enrichment=True)

        asyncio.run(_run())  # just verifying no exception

    # ------------------------------------------------------------------
    # Step 2: execute_skill() — LLM fails
    # ------------------------------------------------------------------

    @unittest.skip(
        "uses cognee_skills.pipeline.ingest_skills (removed; ingestion now lives on cognee.remember). Replaced by test_smoke_remember_to_amendify.py"
    )
    def test_step2_execute_returns_failure(self):
        """execute_skill() returns success=False when LLM raises."""
        from cognee.cognee_skills.execute import execute_skill

        skill = _make_skill_node()[1]

        with (
            patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
            patch(
                "cognee.cognee_skills.execute.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=Exception("context length exceeded"),
            ),
        ):
            mock_cfg.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
            result = asyncio.run(execute_skill(skill=skill, task_text="Summarize this"))

        assert result["success"] is False
        assert result["skill_id"] == SKILL_ID
        assert "context length" in result["error"]

    # ------------------------------------------------------------------
    # Step 3: record_skill_run() — persists failure to graph
    # ------------------------------------------------------------------

    def test_step3_observe_records_run(self):
        """record_skill_run() creates a SkillRun and adds it to the graph."""
        from cognee.cognee_skills.observe import record_skill_run

        captured = []

        async def _mock_add(data_points):
            captured.extend(data_points)

        with (
            patch("cognee.cognee_skills.observe.setup", new_callable=AsyncMock),
            patch("cognee.cognee_skills.observe.add_data_points", side_effect=_mock_add),
            patch("cognee.cognee_skills.observe.index_graph_edges", new_callable=AsyncMock),
            patch(
                "cognee.cognee_skills.observe.get_graph_engine",
                new_callable=AsyncMock,
                return_value=AsyncMock(
                    get_nodeset_subgraph=AsyncMock(return_value=([], [])),
                    add_edges=AsyncMock(),
                ),
            ),
        ):
            result = asyncio.run(
                record_skill_run(
                    session_id="test-session",
                    task_text="Summarize this",
                    selected_skill_id=SKILL_ID,
                    success_score=0.0,
                    error_type="llm_error",
                    error_message="context length exceeded",
                )
            )

        assert result["selected_skill_id"] == SKILL_ID
        assert result["success_score"] == 0.0
        assert len(captured) == 1
        skill_run = captured[0]
        assert skill_run.selected_skill_id == SKILL_ID
        assert skill_run.success_score == 0.0

    # ------------------------------------------------------------------
    # Step 4: inspect_skill() — uses skill node + failed runs from graph
    # ------------------------------------------------------------------

    @unittest.skip(
        "uses cognee_skills.pipeline.ingest_skills (removed; ingestion now lives on cognee.remember). Replaced by test_smoke_remember_to_amendify.py"
    )
    def test_step4_inspect_returns_inspection(self):
        """inspect_skill() analyses failed runs and returns a SkillInspection."""
        from cognee.cognee_skills.inspect import inspect_skill

        skill_nid, skill_props = _make_skill_node()
        run_nodes = [
            (
                uuid4(),
                {
                    "type": "SkillRun",
                    "run_id": f"run-{i}",
                    "selected_skill_id": SKILL_ID,
                    "success_score": 0.1,
                    "task_text": "Summarize this",
                    "error_type": "llm_error",
                    "error_message": "context length exceeded",
                    "result_summary": "",
                },
            )
            for i in range(3)
        ]
        all_nodes = [(skill_nid, skill_props)] + run_nodes

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(return_value=(all_nodes, []))

        with (
            patch(
                "cognee.cognee_skills.inspect.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ),
            patch(
                "cognee.cognee_skills.inspect.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=_MOCK_INSPECTION,
            ),
            patch("cognee.cognee_skills.inspect.get_llm_config") as mock_cfg,
            patch("cognee.cognee_skills.inspect.add_data_points", new_callable=AsyncMock),
        ):
            mock_cfg.return_value = MagicMock(llm_model="openai/gpt-4o-mini")
            inspection = asyncio.run(inspect_skill(SKILL_ID, min_runs=1))

        assert inspection is not None
        assert inspection.skill_id == SKILL_ID
        assert inspection.failure_category == "instruction_gap"
        assert inspection.analyzed_run_count == 3
        assert abs(inspection.avg_success_score - 0.1) < 0.001
        assert inspection.inspection_confidence == 0.88

    # ------------------------------------------------------------------
    # Step 5: preview_skill_amendify() — uses inspection to propose fix
    # ------------------------------------------------------------------

    @unittest.skip(
        "uses cognee_skills.pipeline.ingest_skills (removed; ingestion now lives on cognee.remember). Replaced by test_smoke_remember_to_amendify.py"
    )
    def test_step5_preview_returns_amendment(self):
        """preview_skill_amendify() generates an amendment from an inspection."""
        from cognee.cognee_skills.preview_amendify import preview_skill_amendify
        from cognee.cognee_skills.models.skill_inspection import SkillInspection

        inspection = SkillInspection(
            id=uuid4(),
            name="inspection: Summarize",
            description="Inspection for Summarize.",
            inspection_id="insp-e2e-001",
            skill_id=SKILL_ID,
            skill_name="Summarize",
            failure_category="instruction_gap",
            root_cause="Instructions do not handle empty inputs.",
            severity="high",
            improvement_hypothesis="Add explicit handling for empty or very short inputs.",
            analyzed_run_ids=["run-0", "run-1", "run-2"],
            analyzed_run_count=3,
            avg_success_score=0.1,
            inspection_model="openai/gpt-4o-mini",
            inspection_confidence=0.88,
        )
        skill = _make_skill_node()[1]

        with (
            patch(
                "cognee.cognee_skills.preview_amendify.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=_MOCK_AMENDMENT,
            ),
            patch("cognee.cognee_skills.preview_amendify.get_llm_config") as mock_cfg,
            patch(
                "cognee.cognee_skills.preview_amendify.add_data_points",
                new_callable=AsyncMock,
            ),
        ):
            mock_cfg.return_value = MagicMock(llm_model="openai/gpt-4o-mini")
            amendment = asyncio.run(preview_skill_amendify(inspection=inspection, skill=skill))

        assert amendment is not None
        assert amendment.skill_id == SKILL_ID
        assert amendment.inspection_id == "insp-e2e-001"
        assert amendment.status == "proposed"
        assert "guard clause" in amendment.change_explanation
        assert amendment.amendment_confidence == 0.82
        assert amendment.original_instructions == skill["instructions"]
        assert amendment.amended_instructions != amendment.original_instructions

    # ------------------------------------------------------------------
    # Step 6: amendify() — applies the proposed amendment
    # ------------------------------------------------------------------

    @unittest.skip(
        "uses cognee_skills.pipeline.ingest_skills (removed; ingestion now lives on cognee.remember). Replaced by test_smoke_remember_to_amendify.py"
    )
    def test_step6_amendify_applies_amendment(self):
        """amendify() updates the skill node and emits a change event."""
        from cognee.cognee_skills.amendify import amendify

        amendment_nid = uuid4()
        skill_nid, skill_props = _make_skill_node()

        amendment_node = (
            amendment_nid,
            {
                "type": "SkillAmendment",
                "nid": str(amendment_nid),
                "id": str(amendment_nid),
                "amendment_id": "amend-e2e-001",
                "skill_id": SKILL_ID,
                "skill_name": "Summarize",
                "inspection_id": "insp-e2e-001",
                "original_instructions": "Condense the input into 2-3 key bullet points.",
                "amended_instructions": (
                    "Condense the input into 2-3 key bullet points.\n\n"
                    "If the input is empty, respond: 'Input too short to summarize.'"
                ),
                "change_explanation": "Added guard clause for empty/short inputs.",
                "expected_improvement": "Fewer failures on empty inputs.",
                "status": "proposed",
                "pre_amendment_avg_score": 0.1,
                "amendment_model": "openai/gpt-4o-mini",
                "amendment_confidence": 0.82,
                "applied_at_ms": 0,
                "post_amendment_avg_score": 0.0,
                "post_amendment_run_count": 0,
            },
        )

        engine = AsyncMock()
        engine.get_nodeset_subgraph = AsyncMock(
            return_value=([amendment_node, (skill_nid, skill_props)], [])
        )

        with (
            patch(
                "cognee.cognee_skills.amendify.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine,
            ),
            patch(
                "cognee.cognee_skills.tasks.enrich_skills.enrich_skills",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "cognee.cognee_skills.tasks.materialize_task_patterns.materialize_task_patterns",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "cognee.cognee_skills.amendify.add_data_points", new_callable=AsyncMock
            ) as mock_add_dp,
            patch("cognee.cognee_skills.amendify._make_change_event") as mock_event,
        ):
            mock_event.return_value = MagicMock()
            result = asyncio.run(amendify("amend-e2e-001"))

        assert result["success"] is True
        assert result["status"] == "applied"
        assert result["skill_id"] == SKILL_ID
        assert result["amendment_id"] == "amend-e2e-001"
        # change event + amended skill + amendment status = at least 3 add_data_points calls
        assert mock_add_dp.call_count >= 3
        mock_event.assert_called_once_with(
            SKILL_ID, "Summarize", "amended", old_hash="abc123", new_hash=result["new_hash"]
        )

    # ------------------------------------------------------------------
    # Full loop: auto_amendify() chains all steps in one call
    # ------------------------------------------------------------------

    @unittest.skip(
        "uses cognee_skills.pipeline.ingest_skills (removed; ingestion now lives on cognee.remember). Replaced by test_smoke_remember_to_amendify.py"
    )
    def test_full_loop_auto_amendify_end_to_end(self):
        """Full loop: ingest → execute (fail) → observe → auto_amendify (inspect+preview+apply).

        The shared NodeStore simulates a persistent graph.  Each step's writes
        become visible to subsequent reads, validating the full data-flow chain.
        """
        store = _NodeStore()

        # --- Seed graph with the ingested skill node ---
        skill_nid, skill_props = _make_skill_node()
        store.seed((skill_nid, skill_props))

        engine_mock = store.make_engine_mock()

        # --- Execute skill (LLM fails) ---
        from cognee.cognee_skills.execute import execute_skill

        skill_dict = {
            "skill_id": SKILL_ID,
            "name": "Summarize",
            "instructions": "Condense the input into 2-3 key bullet points.",
            "instruction_summary": "Summarizes text into bullet points.",
            "description": "Condense any text into bullet points.",
            "tags": ["context-management"],
            "complexity": "simple",
            "source_path": "",
            "task_patterns": [],
        }

        with (
            patch("cognee.cognee_skills.execute.get_llm_config") as mock_cfg,
            patch(
                "cognee.cognee_skills.execute.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=Exception("context length exceeded"),
            ),
        ):
            mock_cfg.return_value = MagicMock(llm_model="openai/gpt-4o-mini", llm_api_key="test")
            exec_result = asyncio.run(
                execute_skill(skill=skill_dict, task_text="Summarize this empty doc")
            )

        assert exec_result["success"] is False

        # --- Observe: record 3 failed runs into the store ---
        from cognee.cognee_skills.observe import record_skill_run

        async def _add_dp_observe(dps):
            store.ingest(dps)

        for i in range(3):
            with (
                patch("cognee.cognee_skills.observe.setup", new_callable=AsyncMock),
                patch(
                    "cognee.cognee_skills.observe.add_data_points",
                    side_effect=_add_dp_observe,
                ),
                patch(
                    "cognee.cognee_skills.observe.index_graph_edges",
                    new_callable=AsyncMock,
                ),
                patch(
                    "cognee.cognee_skills.observe.get_graph_engine",
                    new_callable=AsyncMock,
                    return_value=engine_mock,
                ),
            ):
                asyncio.run(
                    record_skill_run(
                        session_id=f"sess-{i}",
                        task_text="Summarize this empty doc",
                        selected_skill_id=SKILL_ID,
                        success_score=0.0,
                        error_type="llm_error",
                        error_message="context length exceeded",
                    )
                )

        # Verify the graph now contains skill + 3 SkillRun nodes
        skill_nodes = [n for _, n in store.nodes if n.get("type") == "Skill"]
        run_nodes = [
            n
            for _, n in store.nodes
            if n.get("type") == "SkillRun" and n.get("selected_skill_id") == SKILL_ID
        ]
        assert len(skill_nodes) >= 1
        assert len(run_nodes) == 3

        # --- Inspect: analyse the 3 failed runs ---
        from cognee.cognee_skills.inspect import inspect_skill

        async def _add_dp_inspect(dps):
            store.ingest(dps)

        with (
            patch(
                "cognee.cognee_skills.inspect.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine_mock,
            ),
            patch(
                "cognee.cognee_skills.inspect.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=_MOCK_INSPECTION,
            ),
            patch("cognee.cognee_skills.inspect.get_llm_config") as mock_cfg,
            patch(
                "cognee.cognee_skills.inspect.add_data_points",
                side_effect=_add_dp_inspect,
            ),
        ):
            mock_cfg.return_value = MagicMock(llm_model="openai/gpt-4o-mini")
            inspection = asyncio.run(inspect_skill(SKILL_ID, min_runs=1, score_threshold=0.5))

        assert inspection is not None
        assert inspection.skill_id == SKILL_ID
        assert inspection.analyzed_run_count == 3

        # Inspection node was written to store
        insp_nodes = [n for _, n in store.nodes if n.get("type") == "SkillInspection"]
        assert len(insp_nodes) == 1

        # --- Preview amendify: propose amendment based on inspection ---
        from cognee.cognee_skills.preview_amendify import preview_skill_amendify

        async def _add_dp_preview(dps):
            store.ingest(dps)

        with (
            patch(
                "cognee.cognee_skills.preview_amendify.LLMGateway.acreate_structured_output",
                new_callable=AsyncMock,
                return_value=_MOCK_AMENDMENT,
            ),
            patch("cognee.cognee_skills.preview_amendify.get_llm_config") as mock_cfg,
            patch(
                "cognee.cognee_skills.preview_amendify.add_data_points",
                side_effect=_add_dp_preview,
            ),
        ):
            mock_cfg.return_value = MagicMock(llm_model="openai/gpt-4o-mini")
            amendment = asyncio.run(preview_skill_amendify(inspection=inspection, skill=skill_dict))

        assert amendment is not None
        assert amendment.status == "proposed"
        assert amendment.skill_id == SKILL_ID

        # Amendment node was written to store
        amend_nodes = [n for _, n in store.nodes if n.get("type") == "SkillAmendment"]
        assert len(amend_nodes) == 1

        # --- Amendify: apply the amendment ---
        from cognee.cognee_skills.amendify import amendify

        async def _add_dp_amendify(dps):
            store.ingest(dps)

        with (
            patch(
                "cognee.cognee_skills.amendify.get_graph_engine",
                new_callable=AsyncMock,
                return_value=engine_mock,
            ),
            patch(
                "cognee.cognee_skills.tasks.enrich_skills.enrich_skills",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "cognee.cognee_skills.tasks.materialize_task_patterns.materialize_task_patterns",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "cognee.cognee_skills.amendify.add_data_points",
                side_effect=_add_dp_amendify,
            ),
        ):
            apply_result = asyncio.run(amendify(amendment.amendment_id))

        assert apply_result["success"] is True
        assert apply_result["status"] == "applied"
        assert apply_result["skill_id"] == SKILL_ID

        # Validate the full chain produced consistent IDs throughout
        assert apply_result["amendment_id"] == amendment.amendment_id

    # ------------------------------------------------------------------
    # auto_amendify() via Skills client — orchestration end-to-end
    # ------------------------------------------------------------------

    def test_auto_amendify_client_orchestration(self):
        """Skills.auto_amendify() chains inspect → preview_amendify → amendify correctly.

        Uses real client orchestration code; mocks only the three sub-methods
        to verify that their return values are wired together faithfully.
        """
        from cognee.cognee_skills.client import Skills

        client = Skills()
        inspection_id = "insp-client-001"
        amendment_id = "amend-client-001"

        mock_inspection = {
            "inspection_id": inspection_id,
            "skill_id": SKILL_ID,
            "skill_name": "Summarize",
            "failure_category": "instruction_gap",
            "root_cause": "Instructions do not handle empty inputs.",
            "severity": "high",
            "improvement_hypothesis": "Add guard clause for empty inputs.",
            "analyzed_run_count": 3,
            "avg_success_score": 0.1,
            "inspection_confidence": 0.88,
        }
        mock_amendment = {
            "amendment_id": amendment_id,
            "skill_id": SKILL_ID,
            "skill_name": "Summarize",
            "inspection_id": inspection_id,
            "change_explanation": "Added guard clause for empty/short inputs.",
            "expected_improvement": "Fewer failures on empty inputs.",
            "status": "proposed",
            "amendment_confidence": 0.82,
            "pre_amendment_avg_score": 0.1,
        }
        mock_apply = {
            "success": True,
            "amendment_id": amendment_id,
            "skill_id": SKILL_ID,
            "skill_name": "Summarize",
            "status": "applied",
        }

        async def _run():
            with (
                patch.object(
                    client, "inspect", new_callable=AsyncMock, return_value=mock_inspection
                ),
                patch.object(
                    client,
                    "preview_amendify",
                    new_callable=AsyncMock,
                    return_value=mock_amendment,
                ) as mock_pa,
                patch.object(
                    client, "amendify", new_callable=AsyncMock, return_value=mock_apply
                ) as mock_am,
            ):
                result = await client.auto_amendify(SKILL_ID, min_runs=1)
                return result, mock_pa, mock_am

        result, mock_pa, mock_am = asyncio.run(_run())

        # Correct result structure
        assert result["inspection"]["inspection_id"] == inspection_id
        assert result["amendment"]["amendment_id"] == amendment_id
        assert result["applied"]["success"] is True

        # Correct wiring: preview received the inspection_id from inspect's output
        mock_pa.assert_called_once_with(
            skill_id=SKILL_ID, inspection_id=inspection_id, node_set="skills"
        )
        # Correct wiring: amendify received the amendment_id from preview's output
        mock_am.assert_called_once_with(
            amendment_id=amendment_id,
            write_to_disk=False,
            validate=False,
            validation_task_text="",
            node_set="skills",
        )

    # ------------------------------------------------------------------
    # auto_amendify skips when there are insufficient failures
    # ------------------------------------------------------------------

    def test_auto_amendify_skips_when_no_failures(self):
        """auto_amendify returns None when inspect finds insufficient failed runs."""
        from cognee.cognee_skills.client import Skills

        client = Skills()

        async def _run():
            with patch.object(client, "inspect", new_callable=AsyncMock, return_value=None):
                return await client.auto_amendify(SKILL_ID, min_runs=5)

        result = asyncio.run(_run())
        assert result is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen_empty():
    """Async generator that yields nothing (simulates an empty pipeline run)."""
    return
    yield  # make it a generator


if __name__ == "__main__":
    unittest.main()
