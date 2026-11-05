"""Link technical schema nodes to the business concepts they realize.

Relational ingestion (``migrate_relational_database``, the DLT schema graph) writes one
node per source table. When an ontology is configured, this module resolves each table
name against the ontology's classes — the same fuzzy lookup cognify uses for extracted
entity types — and emits

* an ``EntityType`` node for the matched class (and its ``is_a`` parents), under the
  same deterministic id cognify would give it, so the two ingestion paths meet on one
  node,
* a ``realizes`` edge from the table node to that class, stamped with whether the
  table is the *authoritative* source for the concept and who owns it
  (``ontology_config["authoritative_sources"]`` / ``ONTOLOGY_AUTHORITATIVE_SOURCES``),
* and, per column whose name resolves to an ontology property, a ``SchemaColumn`` node
  with ``table -has_column-> column -realizes-> OntologyProperty`` plus the property's
  ``domain`` / ``range`` classes.

That is the business-to-technical mapping: "table ``crm.customers`` (authoritative,
owned by sales-ops) realizes concept ``Customer``; its column ``cust_id`` realizes
``hasCustomerId``". No LLM is involved; a table or column the ontology does not know
is left alone — the ``mapping`` proposals of ``cognee.modules.proposals`` pick those up.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.engine.models import EntityType, OntologyProperty
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.ontology.term_matching import compact, multiword_match_is_sound, singular
from cognee.shared.logging_utils import get_logger
from cognee.tasks.schema.models import SchemaColumn

logger = get_logger("ontology_schema_alignment")

REALIZES_RELATIONSHIP = "realizes"
HAS_COLUMN_RELATIONSHIP = "has_column"
_CLASS_CATEGORY = "classes"
_PROPERTY_CATEGORY = "properties"
_MIN_STEM_LENGTH = 3

EdgeTuple = tuple[UUID, UUID, str, dict[str, Any]]
Subgraph = tuple[list[AttachedOntologyNode], list[tuple[str, str, str]], AttachedOntologyNode]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: str = ""


@dataclass(frozen=True)
class SchemaTableSpec:
    """What alignment needs to know about one table node."""

    node_id: UUID
    name: str
    columns: tuple[ColumnSpec, ...] = ()


@dataclass(frozen=True)
class SourceAuthority:
    authoritative: bool = False
    owner: str | None = None


@dataclass
class SchemaOntologyAlignment:
    """Nodes and edges that tie schema tables and columns to the ontology."""

    entity_types: dict[UUID, EntityType] = field(default_factory=dict)
    properties: dict[UUID, OntologyProperty] = field(default_factory=dict)
    columns: dict[UUID, SchemaColumn] = field(default_factory=dict)
    edges: list[EdgeTuple] = field(default_factory=list)
    realized_concept_by_table: dict[str, str] = field(default_factory=dict)
    realized_property_by_column: dict[str, str] = field(default_factory=dict)
    _edge_keys: set[tuple[str, str, str]] = field(default_factory=set, repr=False)

    def __bool__(self) -> bool:
        return bool(self.edges)

    @property
    def nodes(self) -> list[Any]:
        """Every node to write, in a dependency-friendly order."""
        return [*self.entity_types.values(), *self.properties.values(), *self.columns.values()]

    def add_edge(self, edge: EdgeTuple) -> None:
        """Append an edge once; the same ``is_a`` chain is reached from many tables."""
        key = (str(edge[0]), str(edge[1]), edge[2])
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(edge)


# --------------------------------------------------------------------------- specs


def column_specs_from(columns: Any) -> tuple[ColumnSpec, ...]:
    """Normalise the column shapes the ingestion paths carry.

    Accepts the SQLite list (``[{"name", "type"}]``), the Postgres dict
    (``{name: {"type": ...}}``), a JSON string of either (``SchemaTable.columns``), or a
    plain list of names. Anything unreadable yields no columns.
    """
    if not columns:
        return ()
    if isinstance(columns, str):
        try:
            columns = json.loads(columns)
        except ValueError:
            return ()
    specs: list[ColumnSpec] = []
    if isinstance(columns, Mapping):
        for name, info in columns.items():
            data_type = info.get("type", "") if isinstance(info, Mapping) else str(info or "")
            specs.append(ColumnSpec(str(name), str(data_type or "")))
    elif isinstance(columns, Iterable):
        for entry in columns:
            if isinstance(entry, Mapping):
                name = entry.get("name") or entry.get("column") or entry.get("column_name")
                if name:
                    specs.append(ColumnSpec(str(name), str(entry.get("type") or "")))
            elif isinstance(entry, str):
                specs.append(ColumnSpec(entry))
    return tuple(specs)


def coerce_table_specs(tables: Iterable[Any]) -> list[SchemaTableSpec]:
    """Accept ``SchemaTableSpec`` items or the older ``(node_id, name[, columns])`` tuples."""
    specs: list[SchemaTableSpec] = []
    for table in tables:
        if isinstance(table, SchemaTableSpec):
            specs.append(table)
            continue
        node_id, name, *rest = table
        columns = column_specs_from(rest[0]) if rest else ()
        specs.append(SchemaTableSpec(node_id, name, columns))
    return specs


def column_node_id(table_node_id: UUID, column_name: str) -> UUID:
    return uuid5(NAMESPACE_OID, f"{table_node_id}:column:{column_name}")


# ----------------------------------------------------------------------- authority


def authority_for(
    table_name: str, authoritative_sources: Mapping[str, Any] | None
) -> SourceAuthority:
    """Look a table up in the authoritative-source map by qualified or bare name.

    Values: ``True`` (authoritative, no owner), an owner string, or
    ``{"authoritative": bool, "owner": str}``.
    """
    if not authoritative_sources:
        return SourceAuthority()
    bare = table_name.rsplit(".", 1)[-1]
    lowered = {str(key).lower(): value for key, value in authoritative_sources.items()}
    value = lowered.get(table_name.lower(), lowered.get(bare.lower()))
    if value is None:
        return SourceAuthority()
    if isinstance(value, Mapping):
        owner = value.get("owner")
        return SourceAuthority(
            authoritative=bool(value.get("authoritative", True)),
            owner=str(owner) if owner else None,
        )
    if isinstance(value, str):
        return SourceAuthority(authoritative=True, owner=value or None)
    return SourceAuthority(authoritative=bool(value))


# ---------------------------------------------------------------------- name forms


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


def column_lookup_names(column_name: str, table_class_name: str | None = None) -> list[str]:
    """Names to try for a column against the ontology's properties, most specific first.

    ``email`` tries ``email`` and ``has_email`` (the ``hasX`` convention). An ``*_id``
    column whose stem abbreviates the table's class — ``cust_id`` on a table that
    realizes ``Customer`` — also tries ``has_customer_id`` / ``customer_id``, since
    ``cust`` alone is too far from ``hascustomerid`` for the fuzzy cutoff.
    """
    bare = column_name.strip()
    lowered = bare.lower()
    candidates = [bare, f"has_{bare}"]
    if table_class_name and lowered.endswith("_id"):
        stem = compact(lowered[:-3])
        class_key = compact(singular(table_class_name.lower()))
        if len(stem) >= _MIN_STEM_LENGTH and class_key.startswith(stem) and stem != class_key:
            candidates.extend((f"has_{class_key}_id", f"{class_key}_id"))
    return [candidate for candidate in dict.fromkeys(candidates) if candidate]


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


def _property_for(node: AttachedOntologyNode, kind: str) -> OntologyProperty:
    normalized_name = generate_node_name(node.name)
    return OntologyProperty(
        id=OntologyProperty.id_for(node.name),
        name=normalized_name,
        description=normalized_name,
        property_kind=kind,
        ontology_valid=True,
        ontology_uri=str(node.uri) if node.uri is not None else None,
    )


def _resolve(
    candidates: list[str], category: str, resolver: BaseOntologyResolver, what: str
) -> Subgraph | None:
    for candidate in candidates:
        try:
            nodes, edges, root = resolver.get_subgraph(node_name=candidate, node_type=category)
        except Exception as error:  # a broken lookup must not fail ingestion
            logger.debug(
                "Ontology lookup failed for %s %r: %s", what, candidate, error, exc_info=True
            )
            return None
        if root is not None and multiword_match_is_sound(candidate, root.name):
            return nodes, edges, root
    return None


def _add_class_subgraph(alignment: SchemaOntologyAlignment, subgraph: Subgraph) -> None:
    """Write the classes of a subgraph and the ontology's own edges among them.

    Same subgraph cognify writes for a matched class: ``is_a`` chain plus the
    object properties between classes, so both paths meet on identical nodes.
    """
    ontology_nodes, ontology_edges, root = subgraph
    class_by_key = {
        generate_edge_name(node.name): node
        for node in ontology_nodes
        if node.category == _CLASS_CATEGORY
    }
    if root.category == _CLASS_CATEGORY:
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


def _property_class_edges(subgraph: Subgraph) -> dict[str, list[str]]:
    """``{"domain": [class keys], "range": [class keys]}`` of the subgraph's root property."""
    _, ontology_edges, root = subgraph
    root_key = generate_edge_name(root.name)
    targets: dict[str, list[str]] = {"domain": [], "range": []}
    for source_name, relationship_name, target_name in ontology_edges:
        if generate_edge_name(source_name) == root_key and relationship_name in targets:
            targets[relationship_name].append(generate_edge_name(target_name))
    return targets


def _class_keys_reached(subgraph: Subgraph) -> set[str]:
    nodes, _, root = subgraph
    keys = {generate_edge_name(node.name) for node in nodes if node.category == _CLASS_CATEGORY}
    if root.category == _CLASS_CATEGORY:
        keys.add(generate_edge_name(root.name))
    return keys


def _guess_property_kind(resolver: BaseOntologyResolver, root: AttachedOntologyNode) -> str:
    graph = getattr(resolver, "graph", None)
    if graph is None:
        return "datatype"
    try:
        from rdflib import OWL, RDF

        if (root.uri, RDF.type, OWL.ObjectProperty) in graph:
            return "object"
    except Exception:  # rdflib absent or a non-rdflib resolver: kind is cosmetic
        logger.debug("Could not determine property kind for %r", root.name, exc_info=True)
    return "datatype"


# ----------------------------------------------------------------------- alignment


def _align_columns(
    alignment: SchemaOntologyAlignment,
    table: SchemaTableSpec,
    table_subgraph: Subgraph | None,
    resolver: BaseOntologyResolver,
    authority: SourceAuthority,
) -> None:
    table_class_name = table_subgraph[2].name if table_subgraph else None
    table_class_keys = _class_keys_reached(table_subgraph) if table_subgraph else set()

    for column in table.columns:
        resolved = _resolve(
            column_lookup_names(column.name, table_class_name),
            _PROPERTY_CATEGORY,
            resolver,
            f"column {table.name}.{column.name}",
        )
        if resolved is None:
            continue
        _, _, root = resolved
        class_edges = _property_class_edges(resolved)
        # ``hasName`` with domain Customer says nothing about a column of ``orders``:
        # when both the property's domain and the table's class are known they must meet.
        if (
            class_edges["domain"]
            and table_class_keys
            and not table_class_keys.intersection(class_edges["domain"])
        ):
            logger.debug(
                "Skipping %s.%s -> %s: property domain %s does not cover table class %s",
                table.name,
                column.name,
                root.name,
                class_edges["domain"],
                table_class_name,
            )
            continue

        ontology_property = _property_for(root, _guess_property_kind(resolver, root))
        alignment.properties.setdefault(ontology_property.id, ontology_property)
        _add_class_subgraph(alignment, resolved)
        for relationship_name, class_keys in class_edges.items():
            for class_key in class_keys:
                target_id = next(
                    (
                        entity_type.id
                        for entity_type in alignment.entity_types.values()
                        if generate_edge_name(entity_type.name) == class_key
                    ),
                    None,
                )
                if target_id is not None:
                    alignment.add_edge(_edge(ontology_property.id, target_id, relationship_name))

        node_id = column_node_id(table.node_id, column.name)
        alignment.columns.setdefault(
            node_id,
            SchemaColumn(
                id=node_id,
                name=column.name,
                table_name=table.name,
                data_type=column.data_type,
                description=(
                    f"Column '{column.name}' of table '{table.name}'"
                    f"{f' ({column.data_type})' if column.data_type else ''}, "
                    f"realizing the business property '{root.name}'."
                ),
            ),
        )
        alignment.add_edge(_edge(table.node_id, node_id, HAS_COLUMN_RELATIONSHIP))
        alignment.add_edge(
            _edge(
                node_id,
                ontology_property.id,
                REALIZES_RELATIONSHIP,
                authoritative=authority.authoritative,
                owner=authority.owner,
                ontology_valid=True,
            )
        )
        alignment.realized_property_by_column[f"{table.name}.{column.name}"] = root.name


def align_tables_with_ontology(
    tables: Iterable[Any],
    resolver: BaseOntologyResolver | None,
    authoritative_sources: Mapping[str, Any] | None = None,
) -> SchemaOntologyAlignment:
    """Resolve each table (and its columns) against the ontology.

    ``tables`` holds ``SchemaTableSpec`` items or ``(table_node_id, table_name[,
    columns])`` tuples. Returns the ``EntityType`` / ``OntologyProperty`` /
    ``SchemaColumn`` nodes to add (deduplicated across tables) and the edges:
    ``table -realizes-> class`` (with ``authoritative`` / ``owner`` from
    ``authoritative_sources``), ``table -has_column-> column -realizes-> property``,
    the property's ``domain`` / ``range``, and the ontology's own edges between the
    classes reached. With no resolver the alignment is empty.
    """
    alignment = SchemaOntologyAlignment()
    specs = coerce_table_specs(tables)
    if resolver is None or not specs:
        return alignment

    for table in specs:
        authority = authority_for(table.name, authoritative_sources)
        table_subgraph = _resolve(
            table_lookup_names(table.name), _CLASS_CATEGORY, resolver, f"table {table.name}"
        )
        if table_subgraph is not None:
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

        if table.columns:
            _align_columns(alignment, table, table_subgraph, resolver, authority)

    if alignment:
        logger.info(
            "Ontology alignment: %d of %d table(s) realize an ontology class (%s)%s.",
            len(alignment.realized_concept_by_table),
            len(specs),
            ", ".join(
                f"{table} -> {concept}"
                for table, concept in alignment.realized_concept_by_table.items()
            ),
            (
                f"; {len(alignment.realized_property_by_column)} column(s) realize a property"
                if alignment.realized_property_by_column
                else ""
            ),
        )
    return alignment
