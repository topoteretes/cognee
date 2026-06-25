import json
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger
from uuid import uuid4

logger = get_logger("cross_connect")


class LinkPredictionResult(BaseModel):
    is_related: bool
    relationship_name: Optional[str] = None
    edge_text: Optional[str] = None


def extract_node_info(row: Dict[str, Any], prefix: str) -> Dict[str, str]:
    node_id = row.get(f"{prefix}_id")
    name = row.get(f"{prefix}_name")
    description = row.get(f"{prefix}_desc")

    props_raw = row.get(f"{prefix}_props")
    props = {}
    if props_raw:
        if isinstance(props_raw, dict):
            props = props_raw
        elif isinstance(props_raw, str):
            try:
                props = json.loads(props_raw)
            except json.JSONDecodeError:
                pass

    if not name:
        name = props.get("name", "")
    if not description:
        description = props.get("description", "")

    return {
        "id": str(node_id) if node_id else "",
        "name": str(name) if name else "",
        "description": str(description) if description else "",
    }


async def cross_connect(data: Any, limit: int = 20) -> Dict[str, int]:
    """
    Find unconnected Entity nodes with a path length of 2 (sharing a neighbor),
    use LLM to predict missing edges (relationships), and connect them.
    """
    graph_engine = await get_graph_engine()

    query = f"""
    MATCH (a:Node)-[r1:EDGE]-(b:Node)-[r2:EDGE]-(c:Node)
    WHERE a.id < c.id AND NOT (a)-[]-(c) AND a.type = 'Entity' AND c.type = 'Entity'
    RETURN a.id AS a_id, a.name AS a_name, a.description AS a_desc, a.properties AS a_props,
           c.id AS c_id, c.name AS c_name, c.description AS c_desc, c.properties AS c_props,
           b.id AS b_id, b.name AS b_name, b.description AS b_desc, b.properties AS b_props,
           r1.relationship_name AS r1_name, r2.relationship_name AS r2_name
    LIMIT {limit}
    """

    try:
        rows = await graph_engine.query(query)
    except Exception as e:
        logger.warning("Cross connect: failed to query graph data: %s", e)
        return {"connections_created": 0}

    if not rows:
        return {"connections_created": 0}

    # Format rows to dicts
    rows_dicts = []
    # Kuzu returns a list of lists or tuples
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 14:
            rows_dicts.append(
                {
                    "a_id": row[0],
                    "a_name": row[1],
                    "a_desc": row[2],
                    "a_props": row[3],
                    "c_id": row[4],
                    "c_name": row[5],
                    "c_desc": row[6],
                    "c_props": row[7],
                    "b_id": row[8],
                    "b_name": row[9],
                    "b_desc": row[10],
                    "b_props": row[11],
                    "r1_name": row[12],
                    "r2_name": row[13],
                }
            )
        elif isinstance(row, dict):
            rows_dicts.append(row)

    new_edges = []

    async def evaluate_pair(row_dict):
        entity_a = extract_node_info(row_dict, "a")
        entity_c = extract_node_info(row_dict, "c")
        entity_b = extract_node_info(row_dict, "b")
        r1_name = row_dict.get("r1_name", "connected_to")
        r2_name = row_dict.get("r2_name", "connected_to")

        if not entity_a["name"] or not entity_c["name"]:
            return

        try:
            system_prompt = (
                "You are an AI memory manager analyzing a knowledge graph. Your task is to perform link prediction.\n"
                "Given Entity A and Entity C, which are not currently linked but share a mutual connection Entity B, "
                "determine if Entity A and Entity C have a direct relationship.\n\n"
                "Entity A:\n"
                f"Name: {entity_a['name']}\n"
                f"Description: {entity_a['description']}\n\n"
                "Entity C:\n"
                f"Name: {entity_c['name']}\n"
                f"Description: {entity_c['description']}\n\n"
                "Mutual Connection Entity B:\n"
                f"Name: {entity_b['name']}\n"
                f"Description: {entity_b['description']}\n"
                f"Entity A connects to B via: {r1_name}\n"
                f"Entity B connects to C via: {r2_name}\n"
            )

            text_input = (
                "Analyze these entities and output a JSON response. Set is_related to true if they are directly related. "
                "Provide relationship_name (concise, lowercase, e.g. works_at, located_in, part_of) and a brief edge_text "
                "summarizing their relationship. If not related, set is_related to false."
            )

            res = await LLMGateway.acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=LinkPredictionResult,
            )

            if res and res.is_related and res.relationship_name:
                edge_props = {
                    "edge_object_id": str(uuid4()),
                    "edge_text": res.edge_text
                    or f"{entity_a['name']} {res.relationship_name} {entity_c['name']}",
                    "feedback_weight": 0.5,
                    "frequency_weight": 0.0,
                }
                new_edges.append(
                    (entity_a["id"], entity_c["id"], res.relationship_name, edge_props)
                )
        except Exception as e:
            logger.warning("Cross connect: error evaluating pair: %s", e)

    # Evaluate pairs in parallel
    await asyncio.gather(*(evaluate_pair(r) for r in rows_dicts))

    if new_edges:
        try:
            await graph_engine.add_edges(new_edges)
            logger.info("Cross connect: created %d new semantic relationships", len(new_edges))
        except Exception as e:
            logger.warning("Cross connect: failed to add edges to graph: %s", e)
            return {"connections_created": 0}

    return {"connections_created": len(new_edges)}
