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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from cognee.modules.engine.models import EntityType
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.models import ONTOLOGY_CLASS_CATEGORY, AttachedOntologyNode
from cognee.modules.ontology.term_matching import compact_key_index, lookup_words, tokens_of
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_schema_alignment")

REALIZES_RELATIONSHIP = "realizes"

EdgeTuple = tuple[UUID, UUID, str, dict[str, Any]]
Subgraph = tuple[list[AttachedOntologyNode], list[tuple[str, str, str]], AttachedOntologyNode]


@dataclass(frozen=True)
class SchemaTableSpec:
    """What alignment needs to know about one table node."""

    node_id: UUID
    name: str


@dataclass(frozen=True)
class SourceAuthority:
    authoritative: bool = False
    owner: str | None = None


@dataclass
class SchemaOntologyAlignment:
    """Nodes and edges that tie schema tables to the ontology."""

    entity_types: dict[UUID, EntityType] = field(default_factory=dict)
    edges: list[EdgeTuple] = field(default_factory=list)
    realized_concept_by_table: dict[str, str] = field(default_factory=dict)
    _edge_keys: set[tuple[str, str, str]] = field(default_factory=set, repr=False)

    def __bool__(self) -> bool:
        return bool(self.edges)

    @property
    def nodes(self) -> list[Any]:
        """Every node to write, in a dependency-friendly order."""
        return list(self.entity_types.values())

    def add_edge(self, edge: EdgeTuple) -> None:
        """Append an edge once; the same ``is_a`` chain is reached from many tables."""
        key = (str(edge[0]), str(edge[1]), edge[2])
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(edge)


# ----------------------------------------------------------------------- authority


def authority_for(
    table_name: str, authoritative_sources: Mapping[str, str | None] | None
) -> SourceAuthority:
    """Look a table up in the authoritative-source map by qualified or bare name.

    A listed table is authoritative, and its value is the owner.
    """
    if not authoritative_sources:
        return SourceAuthority()
    lowered = {key.lower(): owner for key, owner in authoritative_sources.items()}
    for name in (table_name.lower(), table_name.rsplit(".", 1)[-1].lower()):
        if name in lowered:
            return SourceAuthority(authoritative=True, owner=lowered[name])
    return SourceAuthority()


# ------------------------------------------------------------------------- lookups


def _edge(source_id: UUID, target_id: UUID, relationship_name: str, **attributes: Any) -> EdgeTuple:
    return (
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


def _entity_type_for(node: AttachedOntologyNode) -> EntityType:
    normalized_name = generate_node_name(node.name)
    return EntityType(
        id=EntityType.id_for(node.name),
        name=normalized_name,
        description=normalized_name,
        ontology_valid=True,
        ontology_uri=str(node.uri) if node.uri is not None else None,
    )


def _add_class_subgraph(alignment: SchemaOntologyAlignment, subgraph: Subgraph) -> None:
    """Write the classes of a subgraph and the ontology's own edges among them.

    Same subgraph cognify writes for a matched class: ``is_a`` chain plus the
    object properties between classes, so both paths meet on identical nodes.
    """
    ontology_nodes, ontology_edges, root = subgraph
    class_by_key = {
        generate_edge_name(node.name): node
        for node in ontology_nodes
        if node.category == ONTOLOGY_CLASS_CATEGORY
    }
    if root.category == ONTOLOGY_CLASS_CATEGORY:
        class_by_key.setdefault(generate_edge_name(root.name), root)

    for node in class_by_key.values():
        entity_type = _entity_type_for(node)
        alignment.entity_types.setdefault(entity_type.id, entity_type)

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


# ----------------------------------------------------------------------- alignment


def align_tables_with_ontology(
    tables: list[SchemaTableSpec],
    resolver: BaseOntologyResolver | None,
    authoritative_sources: Mapping[str, str | None] | None = None,
) -> SchemaOntologyAlignment:
    """Resolve each table against the ontology.

    Returns the ``EntityType`` nodes to add (deduplicated across tables) and
    the edges: ``table -realizes-> class`` (with ``authoritative`` / ``owner`` from
    ``authoritative_sources``) and the ontology's own edges between the classes
    reached. With no resolver the alignment is empty.
    """
    alignment = SchemaOntologyAlignment()
    if resolver is None or not tables:
        return alignment

    class_index = compact_key_index(resolver, ONTOLOGY_CLASS_CATEGORY)
    for table in tables:
        authority = authority_for(table.name, authoritative_sources)
        class_key = lookup_words(tokens_of(table.name.rsplit(".", 1)[-1]), class_index)
        if class_key is not None:
            table_subgraph = resolver.get_subgraph(
                node_name=class_key, node_type=ONTOLOGY_CLASS_CATEGORY
            )
            root = table_subgraph[2]
            _add_class_subgraph(alignment, table_subgraph)
            alignment.add_edge(
                _edge(
                    table.node_id,
                    EntityType.id_for(root.name),
                    REALIZES_RELATIONSHIP,
                    authoritative=authority.authoritative,
                    owner=authority.owner,
                    ontology_valid=True,
                )
            )
            alignment.realized_concept_by_table[table.name] = root.name

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
