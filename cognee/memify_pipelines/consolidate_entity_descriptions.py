import asyncio
import logging
from uuid import UUID

import cognee
from typing import Any, Dict, List, Optional, Set, Union

import json
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.engine.models.Entity import Entity
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models import EntityType

prompt_name = "consolidate_entity_details.txt"
type_prompt_name = "consolidate_entity_type_details.txt"
type_merge_prompt_name = "consolidate_entity_type_merge.txt"
is_a_only_prompt_name = "consolidate_entity_is_a_only.txt"
MAX_CONCURRENT_ENTITY_LLM_CALLS = 10
MAX_CONCURRENT_TYPE_LLM_CALLS = 10
MAX_NAMED_MEMBERS = 5
MAX_MEMBERS_PER_TYPE_PROMPT = 50


class NodeDescription(BaseModel):
    description: str


class MemberIsAText(BaseModel):
    member_name: str
    is_a_text: str


class EntityTypeDescription(BaseModel):
    description: str
    is_a_texts: List[MemberIsAText] = []


class EntityIsATexts(BaseModel):
    is_a_texts: List[MemberIsAText] = []


# region get_entities_with_neighborhood helper functions
async def get_all_entity_nodes(graph_engine):
    """Retrieve all nodes of type Entity from the graph."""
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    return nodes


async def get_entity_neighborhood(
    node_id: str, props: Dict[str, Any], graph_engine
) -> Dict[str, Any]:
    """Fetch and format data for a single entity node."""
    connections = await graph_engine.get_connections(node_id)

    entity_types, edges, filtered_neighbors = format_connections(node_id, connections)
    entity_props = get_entity_properties(props)
    if "id" not in entity_props:
        entity_props["id"] = str(node_id)
    return {
        "properties": entity_props,
        "edges": edges,
        "neighbors": filtered_neighbors,
        "entity_types": entity_types,
    }


def get_entity_properties(
    props: Dict[str, Any], properties: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Keep only relevant entity properties."""
    if properties is None:
        properties = {"id", "description", "name"}
    return {k: v for k, v in props.items() if k in properties}


def format_connections(
    node_id: str,
    connections: List[Any],
    node_fields: Optional[Set[str]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Optional[str]]], List[Dict[str, Any]]]:
    """Split get_connections() triples into EntityType neighbors, edge info, and other neighbors.

    get_connections(node_id) returns (source, edge, target) triples where node_id
    can be on either side of the edge, so the neighbor is whichever side does not
    match node_id. Unlike get_edges() (which never carries edge properties on any
    backend), the edge dict here includes edge_text when the edge has one.

    An entity can have more than one EntityType neighbor - e.g. classified
    differently across separate ingestions of the same (name-deduped) entity -
    so entity_types is a list, not a single value that would silently drop all
    but the last one found.
    """
    if node_fields is None:
        node_fields = {"id", "name", "description", "text", "type"}

    entity_types: List[Dict[str, Any]] = []
    edges: Dict[str, Dict[str, Optional[str]]] = {}
    filtered_neighbors: List[Dict[str, Any]] = []

    for connection in connections:
        if not isinstance(connection, (list, tuple)) or len(connection) != 3:
            continue

        source, edge_info, target = connection
        neighbor = target if str(source.get("id")) == str(node_id) else source
        neighbor_id = str(neighbor.get("id", ""))

        edges[neighbor_id] = {
            "relationship_name": str(edge_info.get("relationship_name") or "related to"),
            "edge_text": str(edge_info["edge_text"]) if edge_info.get("edge_text") else None,
        }

        if neighbor.get("type") == "EntityType":
            entity_types.append(neighbor)

        filtered_neighbor = {k: v for k, v in neighbor.items() if k in node_fields}
        if len(filtered_neighbor) > 1:
            filtered_neighbors.append(filtered_neighbor)

    return entity_types, edges, filtered_neighbors


# endregion


async def get_entities_with_neighborhood(args) -> List[Dict[str, Any]]:
    """Iterate through all Entity nodes and fetch their edges and neighbor nodes."""
    graph_engine = await get_graph_engine()
    entity_nodes = await get_all_entity_nodes(graph_engine)

    get_entity_neighborhood_tasks = (
        get_entity_neighborhood(node_id, props, graph_engine) for node_id, props in entity_nodes
    )

    return await asyncio.gather(*get_entity_neighborhood_tasks)


# region consolidate_entity_descriptions helper functions
def load_metadata_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"index_fields": ["name"]}
    if value is None:
        return {"index_fields": ["name"]}
    return value


def build_node_neighborhood_prompt(node):
    props = node["properties"]

    text = (
        "This node's description is the following: "
        + props["name"]
        + " - "
        + props["description"]
        + ". It is connected to it's neighbors in the following way:"
    )
    for neighbor in node["neighbors"]:
        edge_info = node.get("edges", {}).get(neighbor.get("id"), {})
        relationship_name = edge_info.get("relationship_name", "related to")
        edge_text = edge_info.get("edge_text")
        neighbor_name = neighbor.get("name", "")
        neighbor_desc = neighbor.get("description", "")
        if neighbor_desc:
            text += f"\n- {relationship_name}: {neighbor_name} - {neighbor_desc}"
        else:
            text += f"\n- {relationship_name} - {neighbor.get('text', '')}"
        if edge_text:
            text += f" (relationship detail: {edge_text})"

    return text


async def query_LLM(text_input, system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,  # no format()
        response_model=NodeDescription,
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


def build_entity(id, name, entity_types: List[EntityType], description):
    """Build an Entity from its (possibly empty) list of EntityType nodes.

    A single type still goes on is_a, unchanged from before. With more than
    one type, none is more "correct" than another - is_a can only hold one
    value, so all of them go on relations as equally-weighted is_a-tagged
    edges instead of picking one arbitrarily and silently dropping the rest.
    """
    is_a = entity_types[0] if len(entity_types) == 1 else None
    relations = (
        [(Edge(relationship_type="is_a"), entity_type) for entity_type in entity_types]
        if len(entity_types) > 1
        else []
    )
    return Entity(
        id=UUID(id),
        name=name,
        is_a=is_a,
        description=description,
        relations=relations,
    )


async def generate_consolidated_entity(node, system_prompt) -> Entity:
    props = node["properties"]
    text = build_node_neighborhood_prompt(node)
    result = await query_LLM(text, system_prompt)
    entity_types = [build_entity_type(entity_type) for entity_type in node["entity_types"]]
    entity = build_entity(props["id"], props["name"], entity_types, result.description)
    return entity


# endregion


# region generate_type_descriptions helper functions
def _entity_type_of(is_a: Optional[Union[EntityType, tuple]]) -> Optional[EntityType]:
    """Unwrap is_a to its EntityType, whether it's bare or (Edge, EntityType)."""
    if isinstance(is_a, tuple):
        return is_a[1]
    return is_a


def _is_a_relation_type(relation: Any) -> Optional[EntityType]:
    """Return the EntityType of an is_a-tagged (Edge, EntityType) tuple in relations, else None."""
    if (
        isinstance(relation, tuple)
        and len(relation) == 2
        and isinstance(relation[0], Edge)
        and relation[0].relationship_type == "is_a"
        and isinstance(relation[1], EntityType)
    ):
        return relation[1]
    return None


def all_entity_types(entity: Entity) -> List[EntityType]:
    """Every type this entity belongs to - the single one on is_a (the common
    case), or all of them from relations for an entity with more than one type
    (is_a is None in that case; see build_entity)."""
    primary = _entity_type_of(entity.is_a)
    if primary is not None:
        return [primary]
    return [
        entity_type
        for relation in entity.relations
        if (entity_type := _is_a_relation_type(relation)) is not None
    ]


def group_entities_by_type(entities: List[Entity]) -> Dict[str, Dict[str, Any]]:
    """Group rewritten entities by their EntityType id.

    An entity with multiple types is registered as a member of every one of
    its type groups, not just one - see all_entity_types(). Entities with no
    type at all are left out of the result.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        for entity_type in all_entity_types(entity):
            type_id = str(entity_type.id)
            group = groups.setdefault(type_id, {"entity_type": entity_type, "members": []})
            group["members"].append(entity)
    return groups


def build_naming_instruction(total_member_count: int, max_named_members: int) -> str:
    """Decide, in code, whether members should be named - never ask the LLM to
    compare total_member_count against the threshold itself. An LLM asked to
    judge "is 5 at or below 5" is unreliable exactly at that boundary; a plain
    Python `<=` never is.
    """
    if total_member_count <= max_named_members:
        return (
            "You MUST name every member shown below individually, pairing each "
            "name with the fact that best distinguishes it from the others."
        )
    return "You MUST NOT name any individual member below - do not mention their names at all."


def build_entity_type_prompt(
    entity_type_name: str,
    current_description: str,
    members: List[Entity],
    total_member_count: int,
    max_named_members: int = MAX_NAMED_MEMBERS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Current description: {current_description or '(none)'}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {member.description}")
    return "\n".join(lines)


async def query_type_LLM(text_input, system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,
        response_model=EntityTypeDescription,
    )


def batch_members(members: List[Entity], batch_size: int) -> List[List[Entity]]:
    return [members[i : i + batch_size] for i in range(0, len(members), batch_size)]


def build_type_merge_prompt(
    entity_type_name: str,
    total_member_count: int,
    partial_descriptions: List[str],
    max_named_members: int = MAX_NAMED_MEMBERS,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Total member count: {total_member_count}",
        build_naming_instruction(total_member_count, max_named_members),
        "You are given partial summaries, each covering a different subset of the members. "
        "Synthesize them into a single final summary following the same rules.",
        "Partial summaries:",
    ]
    for index, partial in enumerate(partial_descriptions, start=1):
        lines.append(f"{index}. {partial}")
    return "\n".join(lines)


async def query_type_merge_LLM(text_input, merge_system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=merge_system_prompt,
        response_model=EntityTypeDescription,
    )


def build_is_a_only_prompt(
    entity_type_name: str,
    final_type_description: str,
    members: List[Entity],
    total_member_count: int,
) -> str:
    lines = [
        f"Entity type: {entity_type_name}",
        f"Final type summary: {final_type_description}",
        f"Total member count: {total_member_count}",
        f"Member cards shown below ({len(members)} of {total_member_count}):",
    ]
    for member in members:
        lines.append(f"- {member.name}: {member.description}")
    return "\n".join(lines)


async def query_is_a_only_LLM(text_input, is_a_system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=is_a_system_prompt,
        response_model=EntityIsATexts,
    )


async def generate_type_description(
    entity_type: EntityType,
    members: List[Entity],
    system_prompt: str,
    merge_system_prompt: str,
    is_a_system_prompt: str,
) -> EntityTypeDescription:
    """Summarize a type's members, batching and merging when there are too many
    for a single prompt. Callers always pass the type's full member list here -
    batching is an internal detail, not something the caller decides.

    Below the batching threshold, one call produces both the description and
    every member's is_a line together, since that single call already sees
    every member. Above the threshold, is_a lines are NOT produced alongside
    the per-batch partial descriptions - a partial only sees a fraction of the
    members, so a comparative claim ("handles the most packages") could be
    true for that batch and false once every member is considered. Instead,
    is_a lines are generated in a second round, after the merge, once a single
    final description exists for every batch to be judged against.
    """
    total_member_count = len(members)

    if total_member_count <= MAX_MEMBERS_PER_TYPE_PROMPT:
        text = build_entity_type_prompt(
            entity_type.name, entity_type.description, members, total_member_count
        )
        return await query_type_LLM(text, system_prompt)

    batches = batch_members(members, MAX_MEMBERS_PER_TYPE_PROMPT)
    partial_results = await asyncio.gather(
        *(
            query_type_LLM(
                build_entity_type_prompt(
                    entity_type.name, entity_type.description, batch, total_member_count
                ),
                system_prompt,
            )
            for batch in batches
        )
    )
    partial_descriptions = [result.description for result in partial_results]
    merge_text = build_type_merge_prompt(entity_type.name, total_member_count, partial_descriptions)
    merged = await query_type_merge_LLM(merge_text, merge_system_prompt)

    is_a_results = await asyncio.gather(
        *(
            query_is_a_only_LLM(
                build_is_a_only_prompt(
                    entity_type.name, merged.description, batch, total_member_count
                ),
                is_a_system_prompt,
            )
            for batch in batches
        )
    )
    is_a_texts = [text for result in is_a_results for text in result.is_a_texts]

    return EntityTypeDescription(description=merged.description, is_a_texts=is_a_texts)


def apply_type_description(
    entity_type: EntityType,
    members: List[Entity],
    new_description: str,
    is_a_texts: Optional[List[MemberIsAText]] = None,
) -> EntityType:
    """Build one updated EntityType (same id, all other fields preserved) and
    point every member's is_a at that same shared instance.

    A single shared instance per type - not a copy per entity - is required:
    get_graph_from_model dedupes nested DataPoints by id and keeps only the
    first one it encounters, so independent per-entity copies of the "same"
    EntityType would silently lose all but one member's write.

    When a member has a matching is_a_text, is_a becomes the
    (Edge(relationship_type="is_a", edge_text=...), updated_entity_type) tuple
    so the text is searchable on the edge. A member with no matching text
    (name mismatch, or none produced) falls back to the bare EntityType rather
    than raising - the entity is still rewritten, just without edge_text.

    A member with more than one type appears here once per type it belongs to
    (once per call to this function, across different groups - see
    group_entities_by_type). Each call must only touch THIS type's slot -
    is_a if this is the member's sole type, or the matching tuple inside
    relations otherwise - and leave the member's other types exactly as they
    were, since a later call for another of its types still needs them intact.
    """
    updated_entity_type = entity_type.model_copy(update={"description": new_description})
    is_a_text_by_name = {item.member_name: item.is_a_text for item in (is_a_texts or [])}

    for member in members:
        is_a_text = is_a_text_by_name.get(member.name)

        primary_type = _entity_type_of(member.is_a)
        if primary_type is not None and primary_type.id == entity_type.id:
            # is_a is a scalar field: get_graph_from_model derives the "is_a"
            # relationship name from the field name itself when there's no
            # Edge wrapper, so a bare EntityType here still persists correctly.
            member.is_a = (
                (Edge(relationship_type="is_a", edge_text=is_a_text), updated_entity_type)
                if is_a_text
                else updated_entity_type
            )
            continue

        for index, relation in enumerate(member.relations):
            if _is_a_relation_type(relation) is not None and relation[1].id == entity_type.id:
                # relations is a list field: without an explicit Edge wrapper,
                # get_graph_from_model would label this edge "relations"
                # instead of "is_a" (it falls back to the field name). Always
                # wrap here, even with edge_text=None, to keep the "is_a"
                # label - unlike the is_a slot above, there is no bare form.
                member.relations[index] = (
                    Edge(relationship_type="is_a", edge_text=is_a_text),
                    updated_entity_type,
                )
                break

    return updated_entity_type


# endregion


async def generate_type_descriptions(entities: List[Entity]) -> List[Entity]:
    """Group rewritten entities by type, generate one type description per group,
    and point every member's is_a at the shared updated EntityType.

    Entities with no type pass through untouched. Returns the same list of
    entities (now with is_a updated where applicable), ready for add_data_points.
    """
    groups = group_entities_by_type(entities)
    system_prompt = render_prompt(type_prompt_name, {})
    merge_system_prompt = render_prompt(type_merge_prompt_name, {})
    is_a_system_prompt = render_prompt(is_a_only_prompt_name, {})
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TYPE_LLM_CALLS)

    async def process_group(group: Dict[str, Any]) -> None:
        async with semaphore:
            entity_type = group["entity_type"]
            members = group["members"]
            result = await generate_type_description(
                entity_type, members, system_prompt, merge_system_prompt, is_a_system_prompt
            )
            apply_type_description(entity_type, members, result.description, result.is_a_texts)

    await asyncio.gather(*(process_group(group) for group in groups.values()))

    return entities


async def generate_consolidated_entities(nodes) -> List[DataPoint]:
    system_prompt = render_prompt(prompt_name, {})
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ENTITY_LLM_CALLS)

    async def generate_with_limit(node):
        async with semaphore:
            return await generate_consolidated_entity(node, system_prompt)

    consolidate_entity_descriptions_tasks = (generate_with_limit(node) for node in nodes)

    return await asyncio.gather(*consolidate_entity_descriptions_tasks)


async def consolidate_entity_descriptions_pipeline():
    extraction_tasks = [Task(get_entities_with_neighborhood)]

    enrichment_tasks = [
        Task(generate_consolidated_entities),
        Task(generate_type_descriptions),
        Task(add_data_points),
    ]

    await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # A placeholder to prevent fetching the entire graph
    )
