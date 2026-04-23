"""Retrieval layer: recommend skills for a given task."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.models.node_set import NodeSet

logger = logging.getLogger(__name__)


async def recommend_skills(
    task_text: str,
    top_k: int = 5,
    node_set: str = "skills",
    prefers_boost: float = 0.2,
) -> List[Dict[str, Any]]:
    """Given a task description, return the best-matching skills with evidence.

    Stages:
      1. Vector search over Skill_instruction_summary (semantic candidates).
      2. Vector search over TaskPattern_text (query-relevant patterns).
      3. Collect prefers edges from graph, compute query-weighted prefers_score.
      4. Blend vector_score + prefers_score, rank, enrich with metadata.

    Args:
        task_text: Natural language task description.
        top_k: Number of recommendations to return.
        node_set: Scope vector search to this belongs_to_set.
        prefers_boost: Weight for prefers signal in blending (0-1).
    """
    vector_engine = get_vector_engine()

    # Stage 1: semantic candidate skills
    scored_results = await vector_engine.search(
        collection_name="Skill_instruction_summary",
        query_text=task_text,
        limit=top_k * 4,
        include_payload=True,
        node_name=[node_set] if node_set else None,
    )

    if not scored_results:
        return []

    hit_ids = {str(r.id) for r in scored_results}

    # Stage 2: match TaskPatterns for query-relevant prefers boosting
    matched_pattern_sims: Dict[str, float] = {}
    try:
        pattern_results = await vector_engine.search(
            collection_name="TaskPattern_text",
            query_text=task_text,
            limit=top_k * 2,
            include_payload=True,
        )
        for pr in pattern_results or []:
            matched_pattern_sims[str(pr.id)] = max(0.0, 1.0 - pr.score)
    except Exception:
        logger.debug("TaskPattern_text collection not available, skipping pattern matching")

    # Load skills subgraph (scoped to the "skills" nodeset)
    engine = await get_graph_engine()
    raw_nodes, raw_edges = await engine.get_nodeset_subgraph(
        node_type=NodeSet, node_name=[node_set]
    )

    nodes_by_id: Dict[str, Dict] = {}
    skill_nodes: Dict[str, Dict] = {}
    pattern_nodes: Dict[str, Dict] = {}
    run_nodes: List[Dict] = []

    for nid, props in raw_nodes:
        nid_str = str(nid)
        nodes_by_id[nid_str] = props
        node_type = props.get("type", "")
        if node_type == "Skill" and nid_str in hit_ids:
            skill_nodes[nid_str] = props
        elif node_type == "TaskPattern":
            pattern_nodes[nid_str] = props
        elif node_type == "SkillRun":
            run_nodes.append(props)

    # Build skill_id -> node_id reverse map. The "skill_id" key is the
    # Skill's canonical ``name`` — stored on disk as props["name"].
    nid_to_skill_id: Dict[str, str] = {}
    for nid_str, props in skill_nodes.items():
        sid = props.get("name", "")
        if sid:
            nid_to_skill_id[nid_str] = sid

    # Collect solves and prefers edges
    skill_to_patterns: Dict[str, List[Dict]] = {}
    prefers_edges: Dict[tuple, float] = {}

    for src_id, tgt_id, rel_name, edge_props in raw_edges:
        src_str, tgt_str = str(src_id), str(tgt_id)
        if rel_name == "solves" and tgt_str in pattern_nodes and src_str in skill_nodes:
            tp = pattern_nodes[tgt_str]
            sid = skill_nodes[src_str].get("name", "")
            if sid:
                skill_to_patterns.setdefault(sid, []).append(
                    {
                        "pattern_key": tp.get("pattern_key", ""),
                        "text": tp.get("text", ""),
                        "category": tp.get("category", ""),
                    }
                )
        elif rel_name == "prefers":
            w = float((edge_props or {}).get("weight", 0))
            prefers_edges[(src_str, tgt_str)] = w

    # Compute query-weighted prefers_score per skill
    # prefers_score(skill) = max over matched patterns p of: pattern_sim[p] * weight[p, skill]
    skill_id_to_prefers: Dict[str, float] = {}
    for (pattern_nid, skill_nid), weight in prefers_edges.items():
        pattern_sim = matched_pattern_sims.get(pattern_nid, 0.0)
        if pattern_sim <= 0:
            continue
        sid = nid_to_skill_id.get(skill_nid, "")
        if not sid:
            continue
        score = pattern_sim * weight
        if score > skill_id_to_prefers.get(sid, 0.0):
            skill_id_to_prefers[sid] = score

    # Collect prior runs
    skill_to_runs: Dict[str, List[Dict]] = {}
    for run in run_nodes:
        sid = run.get("selected_skill_id", "")
        if sid:
            skill_to_runs.setdefault(sid, []).append(
                {
                    "task_text": run.get("task_text", ""),
                    "success_score": run.get("success_score", 0),
                    "result_summary": run.get("result_summary", "")[:200],
                }
            )

    # Score, blend, rank
    candidates = []
    for result in scored_results:
        nid_str = str(result.id)
        props = skill_nodes.get(nid_str, nodes_by_id.get(nid_str, {}))
        skill_id = props.get("name", "")  # Skill.name is the canonical id
        vector_score = max(0.0, 1.0 - result.score)
        prefers_score = skill_id_to_prefers.get(skill_id, 0.0)
        final_score = (1 - prefers_boost) * vector_score + prefers_boost * prefers_score
        candidates.append((final_score, vector_score, prefers_score, nid_str, props, skill_id))

    candidates.sort(key=lambda c: c[0], reverse=True)

    recommendations = []
    for final_score, vector_score, prefers_score, nid_str, props, skill_id in candidates[:top_k]:
        recommendations.append(
            {
                "skill_id": skill_id,
                "name": props.get("name", ""),
                "instruction_summary": props.get("instruction_summary", ""),
                "tags": props.get("tags", []),
                "complexity": props.get("complexity", ""),
                "score": round(final_score, 4),
                "vector_score": round(vector_score, 4),
                "prefers_score": round(prefers_score, 4),
                "task_patterns": skill_to_patterns.get(skill_id, []),
                "prior_runs": skill_to_runs.get(skill_id, []),
            }
        )

    logger.info("Recommended %d skills for: %s", len(recommendations), task_text[:80])
    return recommendations
