import asyncio
import json
from typing import Any, Dict, List
from uuid import UUID
from typing import cast

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity
from cognee.shared.logging_utils import get_logger

from .constants import REASONING_HEADROOM_TOKENS
from .models import NodeDescription

logger = get_logger("consolidate_entity_descriptions")

prompt_name = "consolidate_entity_details.txt"
MAX_CONCURRENT_ENTITY_LLM_CALLS = 10
MAX_NEIGHBORS_IN_PROMPT = 20
MAX_NEIGHBOR_TEXT_CHARS = 500
# The response is one short paragraph - this call never needs more than the
# model deciding to ramble, and MAX_NEIGHBOR_TEXT_CHARS (~500 chars, ~125
# tokens) is already the target length once this description gets reused
# elsewhere as a compact card. REASONING_HEADROOM_TOKENS covers the rest:
# reasoning models spend hidden reasoning tokens out of this same budget
# (see constants.py), so a little slack above the content target isn't
# enough on its own - confirmed empirically against cognee's own default
# model, where 250 with no headroom intermittently returned empty content.
PARAGRAPH_MAX_COMPLETION_TOKENS = REASONING_HEADROOM_TOKENS + 250


def load_metadata_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"index_fields": ["name"]}
    if value is None:
        return {"index_fields": ["name"]}
    return value


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_node_neighborhood_prompt(
    node,
    max_neighbors: int = MAX_NEIGHBORS_IN_PROMPT,
    max_neighbor_text_chars: int = MAX_NEIGHBOR_TEXT_CHARS,
):
    props = node["properties"]
    neighbors = node["neighbors"]

    text = (
        "This node's description is the following: "
        + props["name"]
        + " - "
        + props["description"]
        + ". It is connected to it's neighbors in the following way:"
    )

    dropped_count = len(neighbors) - max_neighbors
    if dropped_count > 0:
        logger.warning(
            "build_node_neighborhood_prompt: dropping %d of %d neighbors for entity %r (cap is %d)",
            dropped_count,
            len(neighbors),
            props.get("name"),
            max_neighbors,
        )

    for neighbor in neighbors[:max_neighbors]:
        # A neighbor can be linked by more than one distinct edge (e.g.
        # "works_at" and "visited" both connecting the same pair) - emit one
        # line per edge rather than only the last one found.
        edge_infos = node.get("edges", {}).get(neighbor.get("id"), []) or [{}]
        neighbor_name = neighbor.get("name", "")
        neighbor_desc = neighbor.get("description", "")

        for edge_info in edge_infos:
            relationship_name = edge_info.get("relationship_name", "related to")
            edge_text = edge_info.get("edge_text")
            chunk_text = neighbor.get("text", "")

            if neighbor_desc:
                text += (
                    f"\n- {relationship_name}: {neighbor_name} - "
                    f"{_truncate(neighbor_desc, max_neighbor_text_chars)}"
                )
                if edge_text:
                    text += (
                        f" (relationship detail: {_truncate(edge_text, max_neighbor_text_chars)})"
                    )
            elif neighbor.get("type") == "DocumentChunk" and chunk_text:
                # Use the chunk's source text, not contains edge_text ("Document chunk
                # mentions …") - that meta label makes the LLM echo provenance instead
                # of the underlying facts.
                text += (
                    f"\n- {relationship_name} - {_truncate(chunk_text, max_neighbor_text_chars)}"
                )
            elif edge_text:
                text += f"\n- {relationship_name} - {_truncate(edge_text, max_neighbor_text_chars)}"
            else:
                text += (
                    f"\n- {relationship_name} - {_truncate(chunk_text, max_neighbor_text_chars)}"
                )

    return text


async def query_LLM(
    text_input, system_prompt, max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS
):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,  # no format()
        response_model=NodeDescription,
        max_completion_tokens=max_completion_tokens,
    )


def build_entity_type(entity_type_node):
    entity_type_id, entity_type_props = entity_type_node["id"], entity_type_node
    entity_type_props = {
        **entity_type_props,
        "id": entity_type_id,
        "metadata": load_metadata_to_dict(entity_type_props.get("metadata")),
    }
    entity_type = EntityType(**entity_type_props)
    return entity_type


def build_entity(props: Dict[str, Any], entity_types: List[EntityType], description: str) -> Entity:
    """Rebuild an Entity from its full stored properties and (possibly empty) list of EntityType nodes.

    Rebuilt from the full stored props - not a hand-picked subset - because
    add_data_points()'s upsert replaces a node's whole property blob instead
    of merging into it: any field missing from the rebuilt Entity is gone
    from the graph afterward, not left as it was. is_a/relations are always
    recomputed from entity_types rather than read off props, since they are
    graph edges, not node properties.

    The first type found always goes on is_a - same convention already used
    by rdf_ingest.py for ontology individuals with more than one rdf:type.
    is_a must never be left empty for an entity that has a type: code outside
    this pipeline (e.g. record_provenance._entity_type_name) reads only is_a
    and silently falls back to a generic label if it's None. Any additional
    types go on relations as equally-weighted is_a-tagged edges - not
    "lesser" than the one on is_a, just not the one exposed on the scalar
    field code elsewhere already expects to be able to read.
    """
    is_a = entity_types[0] if entity_types else None
    relations = [(Edge(relationship_type="is_a"), entity_type) for entity_type in entity_types[1:]]
    entity_props = {
        **props,
        "description": description,
        "metadata": load_metadata_to_dict(props.get("metadata")),
    }
    entity_props.pop("is_a", None)
    entity_props.pop("relations", None)
    entity = cast(Entity, Entity.from_dict(entity_props))
    entity_id = props["id"]
    entity.id = entity_id if isinstance(entity_id, UUID) else UUID(str(entity_id))
    entity.is_a = is_a
    entity.relations = relations
    return entity


async def generate_consolidated_entity(
    node,
    system_prompt,
    max_neighbors: int = MAX_NEIGHBORS_IN_PROMPT,
    max_neighbor_text_chars: int = MAX_NEIGHBOR_TEXT_CHARS,
    max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS,
) -> Entity:
    props = node["properties"]
    text = build_node_neighborhood_prompt(node, max_neighbors, max_neighbor_text_chars)
    result = await query_LLM(text, system_prompt, max_completion_tokens)
    entity_types = [build_entity_type(entity_type) for entity_type in node["entity_types"]]
    entity = build_entity(props, entity_types, result.description)
    return entity


async def generate_consolidated_entities(
    nodes,
    max_concurrent_calls: int = MAX_CONCURRENT_ENTITY_LLM_CALLS,
    max_neighbors: int = MAX_NEIGHBORS_IN_PROMPT,
    max_neighbor_text_chars: int = MAX_NEIGHBOR_TEXT_CHARS,
    max_completion_tokens: int = PARAGRAPH_MAX_COMPLETION_TOKENS,
) -> List[DataPoint]:
    system_prompt = render_prompt(prompt_name, {})
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def generate_with_limit(node):
        async with semaphore:
            return await generate_consolidated_entity(
                node, system_prompt, max_neighbors, max_neighbor_text_chars, max_completion_tokens
            )

    consolidate_entity_descriptions_tasks = (generate_with_limit(node) for node in nodes)

    return await asyncio.gather(*consolidate_entity_descriptions_tasks)
