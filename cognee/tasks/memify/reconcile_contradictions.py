import json
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger
from uuid import uuid4

logger = get_logger("reconcile_contradictions")


class ReconciliationDecision(BaseModel):
    is_contradiction: bool
    superseding_target_id: Optional[str] = None
    superseded_target_id: Optional[str] = None
    explanation: Optional[str] = None


async def reconcile_contradictions(data: Any, limit: int = 20) -> Dict[str, int]:
    """
    Search for potentially conflicting relationships, query the LLM to determine
    if they contradict, create a 'supersedes' edge from the new claim to the old one,
    and demote the weights of the stale edge/node.
    """
    graph_engine = await get_graph_engine()

    query = f"""
    MATCH (n:Node)-[r1:EDGE]->(target1:Node)
    MATCH (n)-[r2:EDGE]->(target2:Node)
    WHERE n.type = 'Entity' 
      AND r1.relationship_name = r2.relationship_name 
      AND target1.id < target2.id
    RETURN n.id AS source_id, n.name AS source_name, n.description AS source_desc,
           r1.relationship_name AS relationship_name,
           target1.id AS t1_id, target1.name AS t1_name, target1.description AS t1_desc, r1.properties AS r1_props,
           target2.id AS t2_id, target2.name AS t2_name, target2.description AS t2_desc, r2.properties AS r2_props
    LIMIT {limit}
    """

    try:
        rows = await graph_engine.query(query)
    except Exception as e:
        logger.warning("Reconcile contradictions: failed to query graph data: %s", e)
        return {"contradictions_resolved": 0}

    if not rows:
        return {"contradictions_resolved": 0}

    # Format rows to dicts
    rows_dicts = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 12:
            rows_dicts.append(
                {
                    "source_id": row[0],
                    "source_name": row[1],
                    "source_desc": row[2],
                    "relationship_name": row[3],
                    "t1_id": row[4],
                    "t1_name": row[5],
                    "t1_desc": row[6],
                    "r1_props": row[7],
                    "t2_id": row[8],
                    "t2_name": row[9],
                    "t2_desc": row[10],
                    "r2_props": row[11],
                }
            )
        elif isinstance(row, dict):
            rows_dicts.append(row)

    new_edges = []
    edge_weight_updates = {}
    node_weight_updates = {}

    async def evaluate_contradiction(row_dict):
        source_name = row_dict.get("source_name", "")
        relationship_name = row_dict.get("relationship_name", "")
        t1_id = str(row_dict.get("t1_id"))
        t1_name = row_dict.get("t1_name", "")
        t1_desc = row_dict.get("t1_desc", "")
        _t2_id = str(row_dict.get("t2_id"))
        t2_name = row_dict.get("t2_name", "")
        t2_desc = row_dict.get("t2_desc", "")

        if not t1_name or not t2_name:
            return

        try:
            system_prompt = (
                "You are an AI memory manager resolving contradictions in a knowledge graph.\n"
                f"Entity '{source_name}' has two relationships of type '{relationship_name}' pointing to different targets:\n\n"
                f"Claim 1 Target: '{t1_name}'\n"
                f"Claim 1 Description: {t1_desc}\n\n"
                f"Claim 2 Target: '{t2_name}'\n"
                f"Claim 2 Description: {t2_desc}\n\n"
                "Analyze these claims. Determine if they are contradictory (e.g. conflicting dates, locations, or states). "
                "If they contradict, set is_contradiction to true. Identify which target represents the newer or more correct version "
                "and set its ID to superseding_target_id. Set the older/stale target ID to superseded_target_id. "
                "If they are not contradictory (e.g. valid multiple values like multiple team members), set is_contradiction to false."
            )

            text_input = "Analyze these potential contradictions and return a JSON output."

            res = await LLMGateway.acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=ReconciliationDecision,
            )

            if (
                res
                and res.is_contradiction
                and res.superseding_target_id
                and res.superseded_target_id
            ):
                # 1. Create a supersedes relationship edge between targets
                edge_props = {
                    "edge_object_id": str(uuid4()),
                    "edge_text": res.explanation or "supersedes",
                    "feedback_weight": 0.5,
                    "frequency_weight": 0.0,
                }
                new_edges.append(
                    (res.superseding_target_id, res.superseded_target_id, "supersedes", edge_props)
                )

                # 2. Extract edge_object_id of the stale edge to demote it
                stale_edge_props = (
                    row_dict.get("r1_props")
                    if res.superseded_target_id == t1_id
                    else row_dict.get("r2_props")
                )
                if stale_edge_props:
                    if isinstance(stale_edge_props, str):
                        try:
                            stale_edge_props = json.loads(stale_edge_props)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(stale_edge_props, dict):
                        stale_edge_id = stale_edge_props.get("edge_object_id")
                        if stale_edge_id:
                            edge_weight_updates[stale_edge_id] = 0.1

                # 3. Demote the superseded node weight
                node_weight_updates[res.superseded_target_id] = 0.1
        except Exception as e:
            logger.warning("Reconcile contradictions: error evaluating contradiction: %s", e)

    # Evaluate all candidates in parallel
    await asyncio.gather(*(evaluate_contradiction(r) for r in rows_dicts))

    # Apply changes
    applied_count = 0
    if new_edges:
        try:
            await graph_engine.add_edges(new_edges)
            applied_count = len(new_edges)
        except Exception as e:
            logger.warning("Reconcile contradictions: failed to add supersedes edges: %s", e)

    try:
        if edge_weight_updates:
            await graph_engine.set_edge_feedback_weights(edge_weight_updates)
        if node_weight_updates:
            await graph_engine.set_node_feedback_weights(node_weight_updates)
    except NotImplementedError:
        pass
    except Exception as e:
        logger.warning("Reconcile contradictions: failed to demote weights: %s", e)

    return {"contradictions_resolved": applied_count}
