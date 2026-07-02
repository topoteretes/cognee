"""
CONSOLIDATE memory task — merge semantically near-duplicate Entity nodes.

Algorithm
---------
1. Fetch all Entity nodes from the graph.
2. Embed their names/descriptions via the vector engine.
3. Cluster pairs above ``similarity_threshold`` by cosine similarity.
4. LLM confirmation: only merge pairs the LLM judges as the same entity.
5. Graph surgery: for every (canonical, duplicate) pair —
     a. fetch all edges of the duplicate,
     b. re-add them from/to the canonical node,
     c. merge descriptions via summarize_text,
     d. tag canonical node with merged_from provenance,
     e. delete the duplicate node.
6. dry_run=True reports proposed merges without writing.

Plugs into memify() as an enrichment_task (see consolidate_duplicate_entities_pipeline).
Template: memify_pipelines/consolidate_entity_descriptions.py
Issue: #3393
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MergeDecision(BaseModel):
    should_merge: bool
    reason: str


class MergeCandidate(BaseModel):
    canonical_id: str
    duplicate_id: str
    canonical_name: str
    duplicate_name: str
    similarity: float


class MergeReport(BaseModel):
    proposed: List[MergeCandidate]
    executed: List[MergeCandidate]
    dry_run: bool


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

async def _embed_text(text: str) -> List[float]:
    """Embed a single string using cognee's vector engine."""
    from cognee.infrastructure.databases.vector import get_vector_engine
    vector_engine = get_vector_engine()
    embeddings = await vector_engine.embed_data([text])
    return embeddings[0]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

async def _get_all_entity_nodes(graph_engine) -> List[Tuple[str, Dict[str, Any]]]:
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    return nodes


async def _find_candidate_pairs(
    nodes: List[Tuple[str, Dict[str, Any]]],
    similarity_threshold: float,
) -> List[Tuple[str, str, float, Dict, Dict]]:
    """Return (id_a, id_b, similarity, props_a, props_b) pairs above threshold."""
    if not nodes:
        return []

    texts = [
        f"{props.get('name', '')} {props.get('description', '')}".strip()
        for _, props in nodes
    ]

    embeddings = await asyncio.gather(*[_embed_text(t) for t in texts])

    candidates = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                node_id_a, props_a = nodes[i]
                node_id_b, props_b = nodes[j]
                candidates.append((node_id_a, node_id_b, sim, props_a, props_b))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# LLM confirmation
# ---------------------------------------------------------------------------

_MERGE_SYSTEM_PROMPT = """You are a knowledge graph deduplication assistant.
Given two entity descriptions, decide whether they refer to the same real-world entity.
Be conservative — only say should_merge=true when you are highly confident.
Respond with a JSON object with fields: should_merge (bool) and reason (str)."""


async def _llm_confirm_merge(
    name_a: str, desc_a: str, name_b: str, desc_b: str
) -> MergeDecision:
    text_input = (
        f"Entity A: name='{name_a}', description='{desc_a}'\n"
        f"Entity B: name='{name_b}', description='{desc_b}'"
    )
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=_MERGE_SYSTEM_PROMPT,
        response_model=MergeDecision,
    )


# ---------------------------------------------------------------------------
# Graph surgery
# ---------------------------------------------------------------------------

async def _move_edges(
    graph_engine,
    duplicate_id: str,
    canonical_id: str,
) -> None:
    """Re-point all edges of duplicate_id onto canonical_id."""
    edges = await graph_engine.get_edges(duplicate_id)

    new_edges = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 3:
            continue

        # Normalise across adapter formats
        if len(edge) >= 4:
            src, tgt, rel, props = edge[0], edge[1], edge[2], edge[3] or {}
        elif isinstance(edge[2], dict) and "relationship_name" in edge[2]:
            src, tgt, rel, props = edge[0], edge[1], edge[2]["relationship_name"], edge[2]
        else:
            src, tgt, rel, props = edge[0], edge[1], str(edge[2]), {}

        # Skip self-loops that would be created by re-pointing
        new_src = canonical_id if str(src) == duplicate_id else str(src)
        new_tgt = canonical_id if str(tgt) == duplicate_id else str(tgt)

        if new_src == new_tgt:
            continue

        new_edges.append((new_src, new_tgt, rel, {**props, "moved_from": duplicate_id}))

    if new_edges:
        await graph_engine.add_edges(new_edges)


async def _merge_descriptions(desc_a: str, desc_b: str) -> str:
    """Ask the LLM to write a single merged description."""
    from cognee.infrastructure.llm.prompts import render_prompt
    try:
        from cognee.memify_pipelines.consolidate_entity_descriptions import (
            NodeDescription,
            query_LLM,
        )
        text = (
            f"Merge these two descriptions into one concise sentence.\n"
            f"Description 1: {desc_a}\nDescription 2: {desc_b}"
        )
        result = await query_LLM(text, "Write a single merged description.")
        return result.description
    except Exception:
        return f"{desc_a} | {desc_b}"


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

async def consolidate_duplicate_entities(
    args,
    similarity_threshold: float = 0.92,
    protect_node_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> MergeReport:
    """
    Detect and merge near-duplicate Entity nodes in the graph.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity above which two entities are merge candidates (default 0.92).
    protect_node_types : list[str], optional
        Node types that must never be merged (e.g. ["DocumentChunk"]).
    dry_run : bool
        If True, report proposed merges but write nothing to the graph.
    """
    protect_node_types = protect_node_types or []
    graph_engine = await get_graph_engine()

    nodes = await _get_all_entity_nodes(graph_engine)

    # Filter protected types
    nodes = [
        (nid, props) for nid, props in nodes
        if props.get("type") not in protect_node_types
    ]

    if not nodes:
        logger.info("CONSOLIDATE: no entity nodes found.")
        return MergeReport(proposed=[], executed=[], dry_run=dry_run)

    logger.info("CONSOLIDATE: scanning %d entity nodes for duplicates.", len(nodes))

    candidates = await _find_candidate_pairs(nodes, similarity_threshold)
    logger.info("CONSOLIDATE: %d candidate pairs above threshold %.2f.", len(candidates), similarity_threshold)

    proposed: List[MergeCandidate] = []
    executed: List[MergeCandidate] = []
    merged_away: set[str] = set()

    for node_id_a, node_id_b, sim, props_a, props_b in candidates:
        # Skip if either node was already merged away in this run
        if node_id_a in merged_away or node_id_b in merged_away:
            continue

        name_a = props_a.get("name", node_id_a)
        name_b = props_b.get("name", node_id_b)
        desc_a = props_a.get("description", "")
        desc_b = props_b.get("description", "")

        decision = await _llm_confirm_merge(name_a, desc_a, name_b, desc_b)

        if not decision.should_merge:
            logger.debug("CONSOLIDATE: LLM rejected merge of '%s' + '%s': %s", name_a, name_b, decision.reason)
            continue

        # Canonical = whichever has more properties (richer node wins)
        canonical_id, duplicate_id = (
            (node_id_a, node_id_b)
            if len(props_a) >= len(props_b)
            else (node_id_b, node_id_a)
        )
        canonical_props = props_a if canonical_id == node_id_a else props_b
        duplicate_props = props_b if canonical_id == node_id_a else props_a

        candidate = MergeCandidate(
            canonical_id=canonical_id,
            duplicate_id=duplicate_id,
            canonical_name=canonical_props.get("name", canonical_id),
            duplicate_name=duplicate_props.get("name", duplicate_id),
            similarity=sim,
        )
        proposed.append(candidate)

        if dry_run:
            logger.info(
                "CONSOLIDATE [dry_run]: would merge '%s' (%s) into '%s' (%s) [sim=%.3f] — %s",
                candidate.duplicate_name, duplicate_id,
                candidate.canonical_name, canonical_id,
                sim, decision.reason,
            )
            continue

        # --- Execute merge ---
        try:
            await _move_edges(graph_engine, duplicate_id, canonical_id)

            merged_desc = await _merge_descriptions(
                canonical_props.get("description", ""),
                duplicate_props.get("description", ""),
            )

            # Update canonical node with merged description + provenance
            merged_node_props = {
                **canonical_props,
                "description": merged_desc,
                "merged_from": duplicate_id,
            }
            await graph_engine.add_node(canonical_id, merged_node_props)

            await graph_engine.delete_nodes([duplicate_id])
            merged_away.add(duplicate_id)
            executed.append(candidate)

            logger.info(
                "CONSOLIDATE: merged '%s' (%s) into '%s' (%s) [sim=%.3f]",
                candidate.duplicate_name, duplicate_id,
                candidate.canonical_name, canonical_id,
                sim,
            )
        except Exception as exc:
            logger.error(
                "CONSOLIDATE: failed to merge %s -> %s: %s",
                duplicate_id, canonical_id, exc,
            )

    logger.info(
        "CONSOLIDATE: done. proposed=%d executed=%d dry_run=%s",
        len(proposed), len(executed), dry_run,
    )
    return MergeReport(proposed=proposed, executed=executed, dry_run=dry_run)