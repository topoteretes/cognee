from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

from cognee.low_level import DataPoint
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm import LLMGateway
from cognee.shared.logging_utils import get_logger
from cognee.modules.engine.models import NodeSet
from cognee.tasks.storage import add_data_points, index_graph_edges
from typing import Optional, List, Any
from pydantic import Field

logger = get_logger("coding_rule_association")


class Rule(DataPoint):
    """A single developer rule extracted from text."""

    text: str = Field(..., description="The coding rule associated with the conversation")
    belongs_to_set: Optional[NodeSet] = None
    metadata: dict = {"index_fields": ["rule"]}


class RuleSet(DataPoint):
    """Collection of parsed rules."""

    rules: List[Rule] = Field(
        ...,
        description="List of developer rules extracted from the input text. Each rule represents a coding best practice or guideline.",
    )


async def get_existing_rules(rules_nodeset_name: str) -> List[str]:
    graph_engine = await get_graph_engine()
    nodes_data, _ = await graph_engine.get_nodeset_subgraph(
        node_type=NodeSet, node_name=[rules_nodeset_name]
    )

    existing_rules = [
        item[1]["text"]
        for item in nodes_data
        if isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[1], dict)
        and "text" in item[1]
    ]

    return existing_rules


async def get_origin_edges(data: str, rules: List[Rule]) -> list[Any]:
    vector_engine = get_vector_engine()

    origin_chunk = await vector_engine.search("DocumentChunk_text", data, limit=1)

    try:
        origin_id = origin_chunk[0].id
    except (AttributeError, KeyError, TypeError, IndexError):
        origin_id = None

    relationships = []

    if origin_id and isinstance(rules, (list, tuple)) and len(rules) > 0:
        for rule in rules:
            try:
                rule_id = getattr(rule, "id", None)
                if rule_id is not None:
                    rel_name = "rule_associated_from"
                    relationships.append(
                        (
                            rule_id,
                            origin_id,
                            rel_name,
                            {
                                "relationship_name": rel_name,
                                "source_node_id": rule_id,
                                "target_node_id": origin_id,
                                "ontology_valid": False,
                            },
                        )
                    )
            except Exception as e:
                logger.info(f"Warning: Skipping invalid rule due to error: {e}")
    else:
        logger.info("No valid origin_id or rules provided.")

    return relationships


async def add_rule_associations(
    data: str,
    rules_nodeset_name: str,
    user_prompt_location: str = "coding_rule_association_agent_user.txt",
    system_prompt_location: str = "coding_rule_association_agent_system.txt",
):
    if isinstance(data, list):
        # If data is a list of strings join all strings in list
        data = " ".join(data)

    graph_engine = await get_graph_engine()
    existing_rules = await get_existing_rules(rules_nodeset_name=rules_nodeset_name)
    existing_rules = "\n".join(f"- {rule}" for rule in existing_rules)

    user_context = {"chat": data, "rules": existing_rules}

    user_prompt = render_prompt(user_prompt_location, context=user_context)
    system_prompt = render_prompt(system_prompt_location, context={})

    rule_list = await LLMGateway.acreate_structured_output(
        text_input=user_prompt, system_prompt=system_prompt, response_model=RuleSet
    )

    rules_nodeset = NodeSet(
        id=uuid5(NAMESPACE_OID, name=rules_nodeset_name), name=rules_nodeset_name
    )
    for rule in rule_list.rules:
        rule.belongs_to_set = rules_nodeset

    edges_to_save = await get_origin_edges(data=data, rules=rule_list.rules)

    await add_data_points(data_points=rule_list.rules)

    if len(edges_to_save) > 0:
        await graph_engine.add_edges(edges_to_save)
        await index_graph_edges(edges_to_save)
