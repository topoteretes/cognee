"""Link technical schema nodes to the business concepts they realize.

Relational ingestion (``migrate_relational_database``, the DLT schema graph) writes one
node per source table. When an ontology is configured, this module looks each table name
up among the ontology's classes by exact key, the way recall grounds a question (schema
prefix stripped, last word also tried singular), and emits

* an ``EntityType`` node for the matched class (and its ``is_a`` parents), under the
  same deterministic id cognify would give it, so the two ingestion paths meet on one
  node,
* a ``realizes`` edge from the table node to that class, stamped with whether the
  table is the *authoritative* source for the concept and who owns it
  (``ontology_config["authoritative_sources"]`` / ``ONTOLOGY_AUTHORITATIVE_SOURCES``).

That is the business-to-technical mapping: "table ``crm.customers`` (authoritative,
owned by sales-ops) realizes concept ``Customer``". No LLM is involved; a table the
ontology does not know is left alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from cognee.modules.engine.models import EntityType
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    ontology_data_point,
)
from cognee.modules.ontology.models import ONTOLOGY_CLASS_CATEGORY, AttachedOntologyNode
from cognee.modules.ontology.term_matching import compact_key_index, lookup_words, tokens_of
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_schema_alignment")

REALIZES_RELATIONSHIP = "realizes"


def authority_for(
    table_name: str, authoritative_sources: Mapping[str, str | None] | None
) -> tuple[bool, str | None]:
    """``(authoritative, owner)`` of a table, looked up by qualified or bare name.

    A listed table is authoritative, and its value is the owner.
    """
    lowered = {key.lower(): owner for key, owner in (authoritative_sources or {}).items()}
    for name in (table_name.lower(), table_name.rsplit(".", 1)[-1].lower()):
        if name in lowered:
            return True, lowered[name]
    return False, None


def _class_subgraph(
    nodes: list[AttachedOntologyNode],
    edges: list[tuple[str, str, str]],
    root: AttachedOntologyNode,
) -> tuple[list[EntityType], list[tuple[UUID, UUID, str]]]:
    """The subgraph cognify writes for a matched class, so both paths meet on one node.

    The classes of a ``get_subgraph`` result (the class, its ``is_a`` chain, the classes
    its object properties point to) as ``EntityType`` nodes, and the ontology's own
    edges among them.
    """
    class_by_key = {
        generate_edge_name(node.name): node
        for node in nodes
        if node.category == ONTOLOGY_CLASS_CATEGORY
    }
    class_by_key.setdefault(generate_edge_name(root.name), root)

    class_edges = []
    for source_name, relationship_name, target_name in edges:
        source = class_by_key.get(generate_edge_name(source_name))
        target = class_by_key.get(generate_edge_name(target_name))
        if source is not None and target is not None:
            class_edges.append(
                (
                    EntityType.id_for(source.name),
                    EntityType.id_for(target.name),
                    generate_edge_name(relationship_name),
                )
            )
    return [ontology_data_point(EntityType, node) for node in class_by_key.values()], class_edges


def align_tables_with_ontology(
    table_names: Mapping[UUID, str],
    resolver: BaseOntologyResolver | None,
    authoritative_sources: Mapping[str, str | None] | None = None,
) -> tuple[list[EntityType], list[tuple[UUID, UUID, str, dict[str, Any]]]]:
    """Link each table to the ontology class its name is the exact key of.

    ``table_names`` maps a table node's id to the table's name, bare or qualified by
    its schema or database (``crm.customers``). Returns the ``EntityType`` nodes of the classes reached and the
    edges: ``table -realizes-> class``, with ``authoritative`` / ``owner`` from
    ``authoritative_sources``, and the ontology's own edges among those classes. Each
    node and edge appears once, although many tables reach one ``is_a`` chain. With no
    resolver nothing is returned.
    """
    if resolver is None or not table_names:
        return [], []

    class_index = compact_key_index(resolver, ONTOLOGY_CLASS_CATEGORY)
    entity_types: dict[UUID, EntityType] = {}
    edge_attributes: dict[tuple[UUID, UUID, str], dict[str, Any]] = {}
    roots_by_key: dict[str, AttachedOntologyNode] = {}
    realized: list[str] = []
    for table_id, table_name in table_names.items():
        class_key = lookup_words(tokens_of(table_name.rsplit(".", 1)[-1]), class_index)
        if class_key is None:
            continue
        root = roots_by_key.get(class_key)
        if root is None:
            nodes, edges, root = resolver.get_subgraph(
                node_name=class_key, node_type=ONTOLOGY_CLASS_CATEGORY
            )
            roots_by_key[class_key] = root
            class_types, class_edges = _class_subgraph(nodes, edges, root)
            for entity_type in class_types:
                entity_types.setdefault(entity_type.id, entity_type)
            for edge_key in class_edges:
                edge_attributes.setdefault(edge_key, {})
        authoritative, owner = authority_for(table_name, authoritative_sources)
        edge_attributes[(table_id, EntityType.id_for(root.name), REALIZES_RELATIONSHIP)] = {
            "authoritative": authoritative,
            "owner": owner,
            "ontology_valid": True,
        }
        realized.append(f"{table_name} -> {root.name}")

    if realized:
        logger.info(
            "Ontology alignment: %d of %d table(s) realize an ontology class (%s).",
            len(realized),
            len(table_names),
            ", ".join(realized),
        )
    return list(entity_types.values()), [
        (
            source_id,
            target_id,
            relationship_name,
            {
                "source_node_id": str(source_id),
                "target_node_id": str(target_id),
                "relationship_name": relationship_name,
                **attributes,
            },
        )
        for (source_id, target_id, relationship_name), attributes in edge_attributes.items()
    ]
