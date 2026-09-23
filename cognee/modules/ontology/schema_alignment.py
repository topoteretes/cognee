"""Link technical schema nodes to the business concepts they realize.

Relational ingestion (``migrate_relational_database``, the DLT schema graph) writes one
node per source table. When an ontology is configured, this module resolves each table
name against the ontology's classes — the same fuzzy lookup cognify uses for extracted
entity types — and emits

* an ``EntityType`` node for the matched class (and its ``is_a`` parents), under the
  same deterministic id cognify would give it, so the two ingestion paths meet on one
  node, and
* a ``realizes`` edge from the table node to that class.

That is the first version of the business-to-technical mapping: "table ``customers``
realizes concept ``Customer``". No LLM is involved; a table whose name the ontology
does not know is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from cognee.modules.engine.models import EntityType
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_schema_alignment")

REALIZES_RELATIONSHIP = "realizes"
_CLASS_CATEGORY = "classes"

EdgeTuple = tuple[UUID, UUID, str, dict[str, Any]]


@dataclass
class SchemaOntologyAlignment:
    """Nodes and edges that tie schema tables to ontology classes."""

    entity_types: dict[UUID, EntityType] = field(default_factory=dict)
    edges: list[EdgeTuple] = field(default_factory=list)
    realized_concept_by_table: dict[str, str] = field(default_factory=dict)
    _edge_keys: set[tuple[str, str, str]] = field(default_factory=set, repr=False)

    def __bool__(self) -> bool:
        return bool(self.edges)

    def add_edge(self, edge: EdgeTuple) -> None:
        """Append an edge once; the same ``is_a`` chain is reached from many tables."""
        key = (str(edge[0]), str(edge[1]), edge[2])
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(edge)


def table_lookup_names(table_name: str) -> list[str]:
    """Names to try for a table, most specific first.

    ``public.customer_accounts`` yields ``customer_accounts`` then its singular
    ``customer_account``; the fuzzy matcher (80% cutoff) covers most plural forms
    on its own but not ``-ies``/``-es`` (``companies`` vs ``company`` is 75%).
    """
    bare = table_name.rsplit(".", 1)[-1].strip()
    candidates = [bare]
    lowered = bare.lower()
    if lowered.endswith("ies") and len(lowered) > 4:
        candidates.append(bare[:-3] + "y")
    elif lowered.endswith(("ses", "xes", "zes", "ches", "shes")) and len(lowered) > 4:
        candidates.append(bare[:-2])
    elif lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")) and len(lowered) > 3:
        candidates.append(bare[:-1])
    return [candidate for candidate in dict.fromkeys(candidates) if candidate]


def _edge(source_id: UUID, target_id: UUID, relationship_name: str) -> EdgeTuple:
    return (
        source_id,
        target_id,
        relationship_name,
        {
            "source_node_id": str(source_id),
            "target_node_id": str(target_id),
            "relationship_name": relationship_name,
        },
    )


def _entity_type_for(node: AttachedOntologyNode) -> EntityType:
    normalized_name = generate_node_name(node.name)
    return EntityType(
        id=EntityType.id_for(node.name),
        name=normalized_name,
        description=normalized_name,
        ontology_valid=True,
        ontology_uri=str(node.uri) if node.uri is not None else None,
    )


def _resolve_table_class(
    table_name: str, resolver: BaseOntologyResolver
) -> tuple[list[AttachedOntologyNode], list[tuple[str, str, str]], AttachedOntologyNode] | None:
    for candidate in table_lookup_names(table_name):
        try:
            nodes, edges, root = resolver.get_subgraph(
                node_name=candidate, node_type=_CLASS_CATEGORY
            )
        except Exception as error:  # a broken lookup must not fail ingestion
            logger.debug(
                "Ontology lookup failed for table %r: %s", table_name, error, exc_info=True
            )
            return None
        if root is not None:
            return nodes, edges, root
    return None


def align_tables_with_ontology(
    tables: list[tuple[UUID, str]],
    resolver: BaseOntologyResolver | None,
) -> SchemaOntologyAlignment:
    """Resolve each ``(table_node_id, table_name)`` against the ontology's classes.

    Returns the ``EntityType`` nodes to add (the matched class and the classes its
    ontology subgraph reaches, deduplicated across tables) and the edges:
    ``table -realizes-> class`` plus the ontology's own edges between those classes
    (``is_a`` parents, object properties). With no resolver the alignment is empty.
    """
    alignment = SchemaOntologyAlignment()
    if resolver is None or not tables:
        return alignment

    for table_node_id, table_name in tables:
        resolved = _resolve_table_class(table_name, resolver)
        if resolved is None:
            continue
        ontology_nodes, ontology_edges, root = resolved

        class_nodes = [node for node in ontology_nodes if node.category == _CLASS_CATEGORY]
        class_by_key = {generate_edge_name(node.name): node for node in class_nodes}
        if generate_edge_name(root.name) not in class_by_key:
            class_by_key[generate_edge_name(root.name)] = root
            class_nodes.append(root)

        for node in class_nodes:
            entity_type = _entity_type_for(node)
            alignment.entity_types.setdefault(entity_type.id, entity_type)

        # Same subgraph cognify writes for a matched class: ``is_a`` chain plus the
        # object properties between classes, so both paths meet on identical nodes.
        for source_name, relationship_name, target_name in ontology_edges:
            source = class_by_key.get(generate_edge_name(source_name))
            target = class_by_key.get(generate_edge_name(target_name))
            if source is None or target is None:
                continue
            alignment.add_edge(
                _edge(
                    EntityType.id_for(source.name),
                    EntityType.id_for(target.name),
                    generate_edge_name(relationship_name),
                )
            )

        alignment.add_edge(
            _edge(table_node_id, EntityType.id_for(root.name), REALIZES_RELATIONSHIP)
        )
        alignment.realized_concept_by_table[table_name] = root.name

    if alignment:
        logger.info(
            "Ontology alignment: %d of %d table(s) realize an ontology class (%s).",
            len(alignment.realized_concept_by_table),
            len(tables),
            ", ".join(
                f"{table} -> {concept}"
                for table, concept in alignment.realized_concept_by_table.items()
            ),
        )
    return alignment
