"""Promote SkillRuns from short-term cache into the long-term graph.

All runs are promoted — both successes and failures — because failures
are valuable negative signal (they pull prefers weights down and appear
as prior_runs in retrieval).

Flow:
  1. Load QA entries from SessionManager cache for skill_runs:{session_id}.
  2. Parse each entry, build SkillRun DataPoints.
  3. Persist via add_data_points().
  4. Update TaskPattern.prefers edge weights via the graph adapter (MERGE/update).
  5. Delete promoted entries from cache.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import uuid5, UUID

from cognee.low_level import setup
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.storage import add_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges

from cognee_skills.models.skill_run import SkillRun, ToolCall, CandidateSkill
from cognee_skills.observe import CACHE_USER_ID, _session_key

logger = logging.getLogger(__name__)

NAMESPACE = UUID("c3d4e5f6-a7b8-9012-cdef-123456789012")


def _parse_entry(entry: dict) -> Optional[dict]:
    """Parse a cache QA entry into a run-data dict."""
    context_raw = entry.get("context", "")
    answer_raw = entry.get("answer", "")

    try:
        run_data = json.loads(context_raw) if context_raw else {}
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        answer_data = json.loads(answer_raw) if answer_raw else {}
    except (json.JSONDecodeError, TypeError):
        answer_data = {}

    if answer_data.get("success_score") is not None:
        run_data.setdefault("success_score", answer_data["success_score"])
    if answer_data.get("result_summary"):
        run_data.setdefault("result_summary", answer_data["result_summary"])

    run_data["cache_qa_id"] = entry.get("qa_id", "")
    return run_data


def _build_skill_run(run_data: dict) -> SkillRun:
    """Convert a run-data dict into a SkillRun DataPoint."""
    run_id = f"{run_data.get('session_id', '')}:{run_data.get('task_text', '')}:{run_data.get('started_at_ms', 0)}"

    candidates = []
    for cs in run_data.get("candidate_skills", []):
        candidates.append(
            CandidateSkill(
                id=uuid5(NAMESPACE, f"{run_id}:candidate:{cs.get('skill_id', '')}"),
                skill_id=cs.get("skill_id", ""),
                score=cs.get("score", 0.0),
                signals=cs.get("signals"),
            )
        )

    tool_calls = []
    for i, tc in enumerate(run_data.get("tool_trace", [])):
        tool_calls.append(
            ToolCall(
                id=uuid5(NAMESPACE, f"{run_id}:{tc.get('tool_name', '')}:{i}"),
                tool_name=tc.get("tool_name", ""),
                tool_input=tc.get("tool_input"),
                tool_output=tc.get("tool_output"),
                success=tc.get("success", True),
                duration_ms=tc.get("duration_ms", 0),
            )
        )

    return SkillRun(
        id=uuid5(NAMESPACE, run_id),
        run_id=run_id,
        session_id=run_data.get("session_id", ""),
        cognee_session_id=run_data.get("cognee_session_id", ""),
        task_text=run_data.get("task_text", ""),
        result_summary=run_data.get("result_summary", ""),
        success_score=run_data.get("success_score", 0.0),
        candidate_skills=candidates,
        selected_skill_id=run_data.get("selected_skill_id", ""),
        task_pattern_id=run_data.get("task_pattern_id", ""),
        router_version=run_data.get("router_version", ""),
        tool_trace=tool_calls,
        error_type=run_data.get("error_type", ""),
        error_message=run_data.get("error_message", ""),
        started_at_ms=run_data.get("started_at_ms", 0),
        latency_ms=run_data.get("latency_ms", 0),
        feedback=run_data.get("feedback", 0.0),
        cache_qa_id=run_data.get("cache_qa_id", ""),
    )


async def _update_prefers_weights(
    promoted_runs: List[SkillRun],
) -> int:
    """Update TaskPattern → Skill 'prefers' edge weights via graph adapter.

    Uses incremental averaging: reads prior weight_sum and run_count from
    existing edges, adds the new batch, and writes back the updated values.

    Returns the number of edges updated.
    """
    batch_scores: Dict[tuple, List[float]] = defaultdict(list)
    for run in promoted_runs:
        if run.task_pattern_id and run.selected_skill_id:
            batch_scores[(run.task_pattern_id, run.selected_skill_id)].append(run.success_score)

    if not batch_scores:
        return 0

    engine = await get_graph_engine()
    raw_nodes, raw_edges = await engine.get_graph_data()

    node_id_by_key: Dict[str, str] = {}
    for nid, props in raw_nodes:
        ntype = props.get("type", "")
        if ntype == "TaskPattern":
            node_id_by_key[f"tp:{props.get('pattern_key', '')}"] = str(nid)
        elif ntype == "Skill":
            node_id_by_key[f"sk:{props.get('skill_id', '')}"] = str(nid)

    existing_prefers: Dict[tuple, dict] = {}
    for src_id, tgt_id, rel_name, edge_props in raw_edges:
        if rel_name == "prefers":
            existing_prefers[(str(src_id), str(tgt_id))] = edge_props or {}

    edges_to_update = []
    for (tp_key, skill_id), new_scores in batch_scores.items():
        tp_nid = node_id_by_key.get(f"tp:{tp_key}")
        sk_nid = node_id_by_key.get(f"sk:{skill_id}")
        if not tp_nid or not sk_nid:
            logger.debug("Skipping prefers edge: tp=%s sk=%s (node not found)", tp_key, skill_id)
            continue

        prior = existing_prefers.get((tp_nid, sk_nid), {})
        prior_sum = float(prior.get("weight_sum", 0.0))
        prior_count = int(prior.get("run_count", 0))

        new_sum = prior_sum + sum(new_scores)
        new_count = prior_count + len(new_scores)
        new_weight = new_sum / new_count

        edges_to_update.append(
            (
                tp_nid,
                sk_nid,
                "prefers",
                {
                    "weight": round(new_weight, 4),
                    "weight_sum": round(new_sum, 4),
                    "run_count": new_count,
                },
            )
        )

    if edges_to_update:
        await engine.add_edges(edges_to_update)
        logger.info("Updated %d prefers edges", len(edges_to_update))

    return len(edges_to_update)


async def promote_skill_runs(
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote all runs from cache into the long-term graph.

    Both successful and failed runs are promoted — failures are valuable
    negative signal that pull prefers weights down for unreliable
    pattern+skill pairs and appear as prior_runs in retrieval.

    Args:
        session_id: Specific session to promote. If None, uses "default".

    Returns:
        Dict with promoted/errors counts.
    """
    await setup()

    effective_session = session_id or "default"
    cache_session = _session_key(effective_session)

    sm = get_session_manager()
    entries = await sm.get_session(
        user_id=CACHE_USER_ID,
        session_id=cache_session,
        formatted=False,
    )

    if not entries:
        logger.info("No cached runs found for session %s", effective_session)
        return {"promoted": 0, "errors": 0, "edges_updated": 0}

    runs: List[SkillRun] = []
    errors = 0
    qa_ids_to_delete: List[str] = []

    for entry in entries:
        run_data = _parse_entry(entry)
        if run_data is None:
            errors += 1
            continue

        skill_run = _build_skill_run(run_data)
        runs.append(skill_run)
        if run_data.get("cache_qa_id"):
            qa_ids_to_delete.append(run_data["cache_qa_id"])

    if runs:
        await add_data_points(runs)
        await index_graph_edges()
        logger.info("Promoted %d SkillRuns to graph", len(runs))

    edges_updated = await _update_prefers_weights(runs)

    for qa_id in qa_ids_to_delete:
        try:
            await sm.delete_qa(
                user_id=CACHE_USER_ID,
                session_id=cache_session,
                qa_id=qa_id,
            )
        except Exception as exc:
            logger.warning("Failed to delete cache entry %s: %s", qa_id, exc)

    result = {
        "promoted": len(runs),
        "errors": errors,
        "edges_updated": edges_updated,
    }
    logger.info("Promotion complete: %s", result)
    return result
