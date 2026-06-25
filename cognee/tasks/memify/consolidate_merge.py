import json
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from difflib import SequenceMatcher
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger
from cognee.modules.storage.utils import JSONEncoder

logger = get_logger("consolidate_merge")


class MergeDecision(BaseModel):
    is_duplicate: bool
    primary_entity_id: Optional[str] = None
    consolidated_name: Optional[str] = None
    consolidated_description: Optional[str] = None


def get_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


async def update_node_metadata(graph_engine, node_id: str, name: str, description: str):
    """Update node name, description, and properties dict in a database-agnostic way."""
    try:
        nodes = await graph_engine.get_nodes([node_id])
    except Exception:
        nodes = []

    if not nodes:
        return

    node = nodes[0]
    node_properties = {}

    props_raw = node.get("properties")
    if props_raw:
        if isinstance(props_raw, dict):
            node_properties = props_raw
        elif isinstance(props_raw, str):
            try:
                node_properties = json.loads(props_raw)
            except json.JSONDecodeError:
                pass
    else:
        node_properties = {
            k: v for k, v in node.items() if k not in {"id", "type", "created_at", "updated_at"}
        }

    node_properties["name"] = name
    node_properties["description"] = description

    # Kuzu / Ladybug SET properties
    try:
        query_ladybug = """
        MATCH (n:Node)
        WHERE n.id = $id
        SET n.name = $name, n.description = $description, n.properties = $properties, n.updated_at = timestamp()
        """
        await graph_engine.query(
            query_ladybug,
            {
                "id": node_id,
                "name": name,
                "description": description,
                "properties": json.dumps(node_properties, cls=JSONEncoder),
            },
        )
    except Exception:
        # Neo4j fallback
        query_neo4j = """
        MATCH (n:`__Node__` {id: $id})
        SET n.name = $name, n.description = $description, n.updated_at = timestamp()
        """
        await graph_engine.query(
            query_neo4j, {"id": node_id, "name": name, "description": description}
        )


async def get_node_edges(graph_engine, node_id: str):
    """Retrieve outgoing and incoming edges for a node."""
    out_query = """
    MATCH (n:Node)-[r:EDGE]->(to:Node)
    WHERE n.id = $id
    RETURN to.id AS target_id, r.relationship_name AS relationship_name, r.properties AS properties
    """
    in_query = """
    MATCH (from:Node)-[r:EDGE]->(n:Node)
    WHERE n.id = $id
    RETURN from.id AS source_id, r.relationship_name AS relationship_name, r.properties AS properties
    """

    # Try querying (falls back to generic match if label differs)
    try:
        out_rows = await graph_engine.query(out_query, {"id": node_id})
        in_rows = await graph_engine.query(in_query, {"id": node_id})
    except Exception:
        # Neo4j / alternative label fallback
        out_query_alt = """
        MATCH (n {id: $id})-[r]->(to)
        RETURN to.id AS target_id, type(r) AS relationship_name, properties(r) AS properties
        """
        in_query_alt = """
        MATCH (from)-[r]->(n {id: $id})
        RETURN from.id AS source_id, type(r) AS relationship_name, properties(r) AS properties
        """
        out_rows = await graph_engine.query(out_query_alt, {"id": node_id})
        in_rows = await graph_engine.query(in_query_alt, {"id": node_id})

    # Format edges
    outgoing = []
    incoming = []

    for row in out_rows or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            outgoing.append((row[0], row[1], row[2]))
        elif isinstance(row, dict):
            outgoing.append(
                (row.get("target_id"), row.get("relationship_name"), row.get("properties"))
            )

    for row in in_rows or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            incoming.append((row[0], row[1], row[2]))
        elif isinstance(row, dict):
            incoming.append(
                (row.get("source_id"), row.get("relationship_name"), row.get("properties"))
            )

    return outgoing, incoming


async def consolidate_merge(data: Any, similarity_threshold: float = 0.85) -> Dict[str, int]:
    """
    Search for near-duplicate Entity nodes, evaluate them via LLM,
    and merge duplicates by transferring edges and updating the primary node.
    """
    graph_engine = await get_graph_engine()

    try:
        nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    except Exception as e:
        logger.warning("Consolidate merge: failed to fetch entity nodes: %s", e)
        return {"nodes_merged": 0}

    if len(nodes) < 2:
        return {"nodes_merged": 0}

    # Find candidate duplicate pairs based on name similarity
    candidates = []
    for i in range(len(nodes)):
        id_a, props_a = nodes[i]
        name_a = props_a.get("name", "")
        desc_a = props_a.get("description", "")
        if not name_a:
            continue

        for j in range(i + 1, len(nodes)):
            id_b, props_b = nodes[j]
            name_b = props_b.get("name", "")
            desc_b = props_b.get("description", "")
            if not name_b:
                continue

            if get_similarity(name_a, name_b) >= similarity_threshold:
                candidates.append(
                    {
                        "a": {"id": id_a, "name": name_a, "description": desc_a},
                        "b": {"id": id_b, "name": name_b, "description": desc_b},
                    }
                )

    merged_count = 0

    for candidate in candidates:
        ent_a = candidate["a"]
        ent_b = candidate["b"]

        try:
            system_prompt = (
                "You are an AI memory manager consolidating duplicate entities in a knowledge graph.\n"
                "Determine if the following two entities represent the same concept or real-world entity:\n\n"
                "Entity A:\n"
                f"ID: {ent_a['id']}\n"
                f"Name: {ent_a['name']}\n"
                f"Description: {ent_a['description']}\n\n"
                "Entity B:\n"
                f"ID: {ent_b['id']}\n"
                f"Name: {ent_b['name']}\n"
                f"Description: {ent_b['description']}\n"
            )

            text_input = (
                "Respond in JSON. Set is_duplicate to true if they are near-duplicates. "
                "Specify primary_entity_id (which must be exactly the ID of the entity with more details or the better name). "
                "Provide consolidated_name and consolidated_description summarizing their combined details."
            )

            res = await LLMGateway.acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=MergeDecision,
            )

            if res and res.is_duplicate and res.primary_entity_id:
                primary_id = res.primary_entity_id
                duplicate_id = ent_b["id"] if primary_id == ent_a["id"] else ent_a["id"]

                # 1. Fetch duplicate node's edges
                outgoing_edges, incoming_edges = await get_node_edges(graph_engine, duplicate_id)

                # 2. Redirect outgoing and incoming edges to primary node
                new_edges = []
                for target_id, rel_name, props in outgoing_edges:
                    # Prevent self-loops
                    if target_id != primary_id:
                        edge_props = {}
                        if isinstance(props, str):
                            try:
                                edge_props = json.loads(props)
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(props, dict):
                            edge_props = props
                        new_edges.append((primary_id, target_id, rel_name, edge_props))

                for source_id, rel_name, props in incoming_edges:
                    if source_id != primary_id:
                        edge_props = {}
                        if isinstance(props, str):
                            try:
                                edge_props = json.loads(props)
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(props, dict):
                            edge_props = props
                        new_edges.append((source_id, primary_id, rel_name, edge_props))

                if new_edges:
                    await graph_engine.add_edges(new_edges)

                # 3. Delete duplicate node (automatically prunes old connected edges)
                await graph_engine.delete_nodes([duplicate_id])

                # 4. Update primary node metadata
                await update_node_metadata(
                    graph_engine,
                    primary_id,
                    res.consolidated_name or ent_a["name"],
                    res.consolidated_description or ent_a["description"],
                )

                logger.info(
                    "Consolidate merge: merged duplicate node %s into primary node %s",
                    duplicate_id,
                    primary_id,
                )
                merged_count += 1
        except Exception as e:
            logger.warning("Consolidate merge: error merging candidate duplicate pair: %s", e)

    return {"nodes_merged": merged_count}
