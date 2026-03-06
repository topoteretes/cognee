"""End-to-end demo: closed feedback loop.

Ingest skills -> recommend -> pick top -> simulate execution -> record ->
promote -> recommend again (prefers weights shift).

Usage:
    python cognee_skills/example.py                           # uses built-in example_skills/
    python cognee_skills/example.py /path/to/skills           # uses external folder
    python cognee_skills/example.py /path/to/skills my-repo   # with source_repo label
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import cognee

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee_skills.pipeline import ingest_skills
from cognee_skills.observe import record_skill_run
from cognee_skills.promote import promote_skill_runs
from cognee_skills.retrieve import recommend_skills

COGNEE_SYSTEM_DIR = Path(__file__).parent / ".cognee_system"
DEFAULT_SKILLS_DIR = Path(__file__).parent / "example_skills"

SESSION_ID = "demo-session-001"

SIMULATED_EXECUTIONS: Dict[str, Dict[str, Any]] = {
    "Compress my conversation history to fit in 8k tokens": {
        "result_summary": "Compressed 32k tokens to 7.5k using anchored iterative summarization.",
        "success_score": 0.92,
        "feedback": 0.8,
        "latency_ms": 3520,
    },
    "Reduce my prompt to under 4k tokens": {
        "result_summary": "Compressed 16k tokens to 3.8k using hierarchical summarization.",
        "success_score": 0.85,
        "feedback": 0.6,
        "latency_ms": 2100,
    },
}


async def _resolve_task_pattern(task_text: str, top_rec: Dict[str, Any]) -> str:
    """Pick the best task_pattern_id for a query.

    First tries a vector search on TaskPattern_text (same collection that
    retrieval uses) so the promoted prefers edge lands on a pattern that
    retrieval's matched-patterns-only logic will find.  Falls back to the
    first TaskPattern on the selected skill, or "".
    """
    try:
        vector_engine = get_vector_engine()
        hits = await vector_engine.search(
            collection_name="TaskPattern_text",
            query_text=task_text,
            limit=1,
            include_payload=True,
        )
        if hits:
            hit_id = str(hits[0].id)
            engine = await get_graph_engine()
            raw_nodes, _ = await engine.get_graph_data()
            for nid, props in raw_nodes:
                if str(nid) == hit_id and props.get("type") == "TaskPattern":
                    pk = props.get("pattern_key", "")
                    if pk:
                        return pk
    except Exception:
        pass

    patterns = top_rec.get("task_patterns", [])
    return patterns[0]["pattern_key"] if patterns else ""


def _build_candidate_list(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert recommendation list to the candidate_skills format for observe."""
    return [
        {
            "skill_id": r["skill_id"],
            "score": r["score"],
            "signals": {"vector": r["vector_score"], "prefers": r["prefers_score"]},
        }
        for r in recs
    ]


def _print_recs(recs: List[Dict[str, Any]]) -> None:
    for i, rec in enumerate(recs):
        print(
            f"    {i + 1}. {rec['name']}  "
            f"vector={rec['vector_score']}  prefers={rec['prefers_score']}  "
            f"final={rec['score']}"
        )
        if rec["prior_runs"]:
            best = max(r["success_score"] for r in rec["prior_runs"])
            print(f"       prior_runs={len(rec['prior_runs'])}  best_score={best}")


async def _recommend_record_promote(
    task_text: str,
    step_label: str,
) -> List[Dict[str, Any]]:
    """Full closed-loop iteration: recommend -> pick top -> record -> promote."""
    print(f"\n{'=' * 60}")
    print(f"{step_label}")
    print("=" * 60)

    # Recommend
    recs = await recommend_skills(task_text, top_k=3, node_set="skills")
    print(f"  Query: {task_text}")
    print("  Recommendations:")
    _print_recs(recs)

    if not recs:
        print("  (no skills found, skipping record/promote)")
        return recs

    top = recs[0]

    # Resolve pattern via vector search
    pattern_id = await _resolve_task_pattern(task_text, top)
    print(f"  Selected: {top['name']}  |  pattern: {pattern_id}")

    # Simulate execution
    sim = SIMULATED_EXECUTIONS.get(
        task_text,
        {
            "result_summary": f"Executed {top['name']} successfully.",
            "success_score": 0.80,
            "feedback": 0.5,
            "latency_ms": 1500,
        },
    )

    # Record to cache
    await record_skill_run(
        session_id=SESSION_ID,
        task_text=task_text,
        selected_skill_id=top["skill_id"],
        task_pattern_id=pattern_id,
        result_summary=sim["result_summary"],
        success_score=sim["success_score"],
        candidate_skills=_build_candidate_list(recs),
        router_version="v1.0-closed-loop",
        feedback=sim["feedback"],
        latency_ms=sim["latency_ms"],
    )
    print(
        f"  Recorded run: skill={top['skill_id']}  "
        f"score={sim['success_score']}  pattern={pattern_id}"
    )

    # Promote
    result = await promote_skill_runs(session_id=SESSION_ID)
    print(f"  Promoted: {result['promoted']}  Edges updated: {result['edges_updated']}")

    return recs


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    skills_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SKILLS_DIR
    source_repo = sys.argv[2] if len(sys.argv) > 2 else ""

    if not skills_folder.is_dir():
        print(f"Error: {skills_folder} is not a directory.")
        sys.exit(1)

    skill_count = sum(
        1 for d in skills_folder.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
    )
    print(f"Found {skill_count} skills in {skills_folder}")

    cognee.config.system_root_directory(str(COGNEE_SYSTEM_DIR))
    await cognee.prune.prune_system(metadata=True)

    # ── Step 1: Ingest skills ──
    print("\n" + "=" * 60)
    print("STEP 1: Ingesting skills")
    print("=" * 60)
    await ingest_skills(
        skills_folder=skills_folder,
        dataset_name="skills",
        source_repo=source_repo,
        node_set="skills",
    )

    # ── Step 2: First recommend + record + promote (prefers all zero) ──
    query1 = "Compress my conversation history to fit in 8k tokens"
    baseline_recs = await _recommend_record_promote(
        query1,
        "STEP 2: First query (no prefers data yet)",
    )

    # ── Step 3: Second task + recommend + record + promote ──
    query2 = "Reduce my prompt to under 4k tokens"
    await _recommend_record_promote(
        query2,
        "STEP 3: Second query (reinforces prefers stats)",
    )

    # ── Step 4: Re-run first query — show prefers boost ──
    print(f"\n{'=' * 60}")
    print("STEP 4: Re-run first query (should show prefers boost)")
    print("=" * 60)
    final_recs = await recommend_skills(query1, top_k=3, node_set="skills")
    print(f"  Query: {query1}")
    print("  Recommendations:")
    _print_recs(final_recs)

    # ── Before/after comparison ──
    if baseline_recs and final_recs:
        b = baseline_recs[0]
        f = final_recs[0]
        print(f"\n  Before/After for top skill ({f['name']}):")
        print(f"    vector_score:  {b['vector_score']}  ->  {f['vector_score']}")
        print(f"    prefers_score: {b['prefers_score']}  ->  {f['prefers_score']}")
        print(f"    final_score:   {b['score']}  ->  {f['score']}")

    print("\nDone. To visualize the graph:")
    print("  PYTHONPATH=. .venv/bin/python cognee_skills/inspect_graph.py")


if __name__ == "__main__":
    asyncio.run(main())
