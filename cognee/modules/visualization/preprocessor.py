"""Visualization preprocessor.

Single Python entry point that turns raw ``(nodes_data, edges_data)`` from
the graph adapter into a fully-enriched ``PreprocessedGraph`` ready for the
HTML/JS renderer to consume.

The renderer should *only* read fields produced here — synthesising stage
or bundling information in JavaScript is the source of the current
visualization's mess. By doing it once in Python, every view (Story,
Schema, Context, Retrieval) sees the same enrichment.
"""

import colorsys
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────


SCHEMA_GRAPH_NODE_TYPES = {
    "DatabaseSchema",
    "SchemaTable",
    "SchemaRelationship",
    "TableType",
}

# Maximum sample instance names attached to each schema type node.
SCHEMA_SAMPLES_PER_TYPE: int = 5

# Maximum semantic entity-type cards in the Schema view's Entity column.
# Entity-type diversity grows with the data (every new EntityType the LLM
# extracts becomes its own card), so beyond this cap the long tail is rolled
# up into a single "Other entities" card — the renderer stacks one card per
# type per rank column, which otherwise made the Entity column endless.
SCHEMA_MAX_ENTITY_TYPES: int = 12

# Display name of the rollup card holding the entity-type long tail.
OTHER_ENTITY_TYPES_LABEL: str = "Other entities"

# Internal graph taxonomy types that must not appear as separate type groups in
# the schema view. EntityType is now surfaced as its own schema type group
# alongside the resolved semantic entity types (Person/Field/...); Entity
# instances still collapse to their semantic type via the is_a edge. This set is
# kept (currently empty) so future genuinely-internal types can be added without
# re-plumbing the guards that reference it.
_INTERNAL_TYPES: frozenset = frozenset()


# Stage assignment by node type — drives the left-to-right Story layout.
# Unknown types fall through to "other".
_STAGE_BY_TYPE: Dict[str, str] = {
    "TextDocument": "document",
    "DocumentChunk": "chunk",
    "TextSummary": "summary",
    "GlobalContextSummary": "context",
    "Entity": "entity",
    "EntityType": "type",
    "DatabaseSchema": "schema",
    "SchemaTable": "schema",
    "SchemaRelationship": "schema",
    "TableType": "schema",
    "TableRow": "schema",
    "ColumnValue": "schema",
}


# Visual ordering of stages along the Story view's left-to-right spine.
STAGE_ORDER: Tuple[str, ...] = (
    "document",
    "chunk",
    "entity",
    "type",
    "summary",
    "context",
    "schema",
    "other",
)


# Relationship names that connect a structural parent to its children. Edges
# of these types are bundled in the Story view to cut visual noise on dense
# graphs (Alice has 769 edges, mostly contains/is_a).
_STRUCTURAL_RELATIONS: frozenset = frozenset(
    {
        "contains",
        "is_a",
        "part_of",
        "is_part_of",
        "has_relationship",
        "made_from",
        "summarized_in",
    }
)


# Default colors per node type — preserved verbatim from the original
# monolith so existing test tokens continue to match.
_TYPE_COLOR_MAP: Dict[str, str] = {
    "TextDocument": "#A550FF",
    "DocumentChunk": "#0DFF00",
    "Entity": "#6510F4",
    "EntityType": "#D5C2FF",
    "TextSummary": "#FFB454",
    "GlobalContextSummary": "#00C2FF",
    "TableRow": "#A550FF",
    "TableType": "#6510F4",
    "ColumnValue": "#747470",
    "SchemaTable": "#A550FF",
    "DatabaseSchema": "#6510F4",
    "SchemaRelationship": "#323332",
    "default": "#7c3aed",
}


# Ontology-grounded nodes get a distinct fill: the old #D8D8D8 gray was
# indistinguishable from the #DBD8D8 unknown-type fallback, so ontology
# matches visually disappeared into untyped nodes.
_ONTOLOGY_VALID_COLOR = "#FF5CA8"
_UNKNOWN_TYPE_COLOR = "#DBD8D8"


# Minimum number of structural edges between the same (source_stage,
# target_stage, relation) for the view to render them as a single bundle
# ribbon instead of individual lines.
DEFAULT_BUNDLE_MIN = 5


# ── Helpers reused by orchestrator and views ─────────────────────────────────


def generate_provenance_colors(values):
    """Generate a deterministic color map for a set of provenance values.

    Identical to the original ``_generate_provenance_colors`` — preserved
    verbatim so existing string-token tests continue to pass.
    """
    color_map = {}
    unique = sorted(set(v for v in values if v))
    for i, name in enumerate(unique):
        hue = (i * 137.5) % 360
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.6, 0.65)
        color_map[name] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return color_map


def derive_node_name(node_info, node_id):
    """Pick a human-readable label for a node, falling back through name/title/text/etc."""
    name = node_info.get("name")
    if name:
        return name

    for key in ("title", "text", "summary", "description", "content"):
        value = node_info.get(key)
        if isinstance(value, str) and value.strip():
            normalized = " ".join(value.split())
            return normalized[:120]

    return str(node_id)


def node_type_rank(node_type):
    """Fallback ordering when ``topological_rank`` is missing (0 or None).

    Order matches the Story view's pipeline column order
    (Documents → Chunks → Entities → Types → Summaries → Context) so the
    Schema diagram uses the same left-to-right narrative as the Graph tab.

    Actor / ownership types occupy negative ranks so they flow in *before*
    the document pipeline (Organization → People → Agents → Sessions →
    Brain → Documents → … ): agents write sessions, which are recorded into
    the brains they belong to.
    """
    type_ranks = {
        # Actor & ownership layer (left of the document pipeline)
        "Tenant": -5,
        "User": -4,
        "Agent": -3,
        "Session": -2,
        "Dataset": -1,
        # Document → memory pipeline
        "TextDocument": 0,
        "DocumentChunk": 1,
        "Entity": 2,
        "EntityType": 3,
        "TextSummary": 4,
        "GlobalContextSummary": 5,
        "DatabaseSchema": 0,
        "SchemaTable": 1,
        "SchemaRelationship": 2,
        "TableType": 1,
        "TableRow": 2,
        "ColumnValue": 3,
    }
    return type_ranks.get(node_type, 4)


def _coerce_json_value(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _field_from_column(column):
    if isinstance(column, str):
        return {"name": column, "type": "column", "required": False}

    if not isinstance(column, dict):
        return None

    name = (
        column.get("name") or column.get("column_name") or column.get("field") or column.get("key")
    )
    if not name:
        return None

    column_type = (
        column.get("type")
        or column.get("data_type")
        or column.get("python_type")
        or column.get("nullable")
        or "column"
    )
    required = bool(column.get("primary_key") or column.get("required"))
    if column.get("nullable") is False:
        required = True

    return {"name": str(name), "type": str(column_type), "required": required}


def extract_schema_fields(node):
    fields = []
    columns = _coerce_json_value(node.get("columns"))

    if isinstance(columns, dict):
        for name, column in columns.items():
            if isinstance(column, dict):
                f = _field_from_column({"name": name, **column})
            else:
                f = {"name": str(name), "type": str(column), "required": False}
            if f:
                fields.append(f)
    elif isinstance(columns, list):
        for column in columns:
            f = _field_from_column(column)
            if f:
                fields.append(f)

    if fields:
        return fields

    fallback_keys = (
        "database_type",
        "primary_key",
        "source_table",
        "source_column",
        "target_table",
        "target_column",
        "relationship_type",
        "row_count_estimate",
    )
    for key in fallback_keys:
        value = node.get(key)
        if value is not None and value != "":
            fields.append({"name": key, "type": str(value), "required": False})

    return fields


def extract_schema_graph_data(nodes_list, links_list):
    """Build the DLT/structured-schema graph from SchemaTable/SchemaRelationship nodes.

    Falls back to ``extract_type_schema_graph_data`` when no schema nodes are present.
    """
    schema_nodes = []
    schema_node_ids = set()

    for node in nodes_list:
        if node.get("type") not in SCHEMA_GRAPH_NODE_TYPES:
            continue

        schema_node_ids.add(node["id"])
        schema_nodes.append(
            {
                "id": node["id"],
                "name": node.get("name") or node["id"],
                "type": node.get("type"),
                "description": node.get("description"),
                "fields": extract_schema_fields(node),
                "source_table": node.get("source_table"),
                "target_table": node.get("target_table"),
                "relationship_type": node.get("relationship_type"),
            }
        )

    if schema_nodes:
        schema_links = []
        seen_links = set()
        for link in links_list:
            source = str(link["source"])
            target = str(link["target"])
            if source not in schema_node_ids or target not in schema_node_ids:
                continue

            label = _link_relation(link)
            link_key = (source, target, label)
            if link_key in seen_links:
                continue

            seen_links.add(link_key)
            schema_links.append({"source": source, "target": target, "label": label})

        return {"nodes": schema_nodes, "links": schema_links}

    return extract_type_schema_graph_data(nodes_list, links_list)


def _schema_value_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "nullable"
    return "string"


def extract_type_schema_fields(type_nodes):
    field_counts: Counter = Counter()
    field_types: Dict[str, str] = {}
    preferred_fields = (
        "source_task",
        "source_pipeline",
        "source_node_set",
        "source_user",
        "global_context_bucket_id",
        "level",
        "is_root",
        "topological_rank",
    )
    excluded_fields = {
        "id",
        "type",
        "name",
        "color",
        "text",
        "summary",
        "content",
        "description",
        "metadata",
        "properties",
        "source_content_hash",
        "belongs_to_set",
        "ontology_valid",
        "feedback_weight",
        "importance_weight",
    }

    for node in type_nodes:
        for key, value in node.items():
            if key.startswith("_") or key in excluded_fields or value in (None, ""):
                continue
            field_counts[key] += 1
            field_types.setdefault(key, _schema_value_type(value))

    fields: List[Dict[str, Any]] = [
        {"name": "count", "type": str(len(type_nodes)), "required": True}
    ]
    ordered_field_names: List[str] = []
    for key in preferred_fields:
        if key in field_counts:
            ordered_field_names.append(key)

    for key, _ in field_counts.most_common():
        if key not in ordered_field_names:
            ordered_field_names.append(key)

    for key in ordered_field_names[:5]:
        count = field_counts[key]
        coverage = int(round(count / max(1, len(type_nodes)) * 100))
        fields.append(
            {
                "name": key,
                "type": f"{field_types.get(key, 'any')} {coverage}%",
                "required": count == len(type_nodes),
            }
        )

    return fields


def _relationship_label(relation_counts):
    total = sum(relation_counts.values())
    top_relations = relation_counts.most_common(2)
    parts = [f"{name} ({count})" for name, count in top_relations]
    if len(relation_counts) > len(top_relations):
        parts.append(f"+{len(relation_counts) - len(top_relations)} more")
    return ", ".join(parts) if parts else f"{total} edges"


# Relationship name of the Entity -> EntityType edge used to resolve the
# semantic type of extracted entities (mirrors get_schema_inventory).
ENTITY_TYPE_RELATION: str = "is_a"


def _link_relation(link: Dict[str, Any]) -> str:
    """Read a link's relation name across the shapes the preprocessor emits."""
    edge_info = link.get("edge_info") or {}
    return (
        link.get("relationship_type")
        or edge_info.get("relationship_name")
        or edge_info.get("relationship_type")
        or link.get("relation")
        or "related"
    )


def resolve_semantic_types(
    nodes_list: List[Dict[str, Any]], links_list: List[Dict[str, Any]]
) -> Dict[str, str]:
    """Map each node id to its semantic type name.

    Non-Entity nodes keep their raw ``type`` property. Entity nodes (``type ==
    "Entity"``) resolve to the EntityType ``name`` reached via the ``is_a`` edge,
    so semantic types (Person/Tool/Broker) surface instead of the literal
    "Entity". Mirrors ``get_schema_inventory._resolve_node_types`` adapted to the
    preprocessor's normalized node/link shape.
    """
    nodes_by_id = {node["id"]: node for node in nodes_list}

    # Collect the EntityType target name for each Entity source via the is_a edge
    entity_type_name = {}
    for link in links_list:
        source = str(link["source"])
        target = str(link["target"])
        if _link_relation(link) == ENTITY_TYPE_RELATION and target in nodes_by_id:
            entity_type_name[source] = nodes_by_id[target].get("name")

    node_type = {}
    for node in nodes_list:
        node_id = node["id"]
        raw_type = node.get("type")
        if raw_type == "Entity" and entity_type_name.get(node_id):
            node_type[node_id] = entity_type_name[node_id]
        else:
            node_type[node_id] = raw_type or "Node"
    return node_type


def extract_type_schema_graph_data(
    nodes_list: List[Dict[str, Any]], links_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Fallback schema view: collapse the graph to one node per semantic type."""
    node_type_by_id = resolve_semantic_types(nodes_list, links_list)

    # Names reached via the is_a edge are semantic *entity* types (Person, Broker,
    # …) — rank them in the Entity column rather than letting them fall through to
    # the default ("Summaries") rank.
    nodes_by_id_lookup = {node["id"]: node for node in nodes_list}
    semantic_type_names = set()
    for link in links_list:
        if _link_relation(link) == ENTITY_TYPE_RELATION:
            target_node = nodes_by_id_lookup.get(str(link["target"]))
            if target_node and target_node.get("name"):
                semantic_type_names.add(target_node["name"])

    # Bound the Entity column: keep the most-populated semantic entity types
    # as their own cards and remap the long tail onto one rollup type. The
    # remap happens on node_type_by_id *before* any downstream aggregation, so
    # relationship distributions, pair edges, instance drill-down, and the
    # operation layer all treat the rollup as an ordinary type.
    rolled_up_types: List[Dict[str, Any]] = []
    entity_type_counts = Counter(
        type_name for type_name in node_type_by_id.values() if type_name in semantic_type_names
    )
    if len(entity_type_counts) > SCHEMA_MAX_ENTITY_TYPES:
        kept_types = {
            name for name, _ in entity_type_counts.most_common(SCHEMA_MAX_ENTITY_TYPES - 1)
        }
        rolled_types = set(entity_type_counts) - kept_types
        rolled_up_types = [
            {"name": name, "count": count}
            for name, count in entity_type_counts.most_common()
            if name in rolled_types
        ]
        for node_id, type_name in node_type_by_id.items():
            if type_name in rolled_types:
                node_type_by_id[node_id] = OTHER_ENTITY_TYPES_LABEL
        semantic_type_names = (semantic_type_names - rolled_types) | {OTHER_ENTITY_TYPES_LABEL}

    def _rank_for(type_name):
        if type_name in semantic_type_names:
            return node_type_rank("Entity")
        return node_type_rank(type_name)

    nodes_by_type: Dict[str, List[Dict]] = defaultdict(list)
    for node in nodes_list:
        type_name = node_type_by_id[node["id"]]
        if type_name not in _INTERNAL_TYPES:
            nodes_by_type[type_name].append(node)

    # Aggregate the full per-source-type relationship distribution keyed by
    # (relation, target_type). Built once and shared with the per-type "samples"
    # records and the lossy pair-edge labels below.
    # Track both outgoing AND incoming so types like DocumentChunk whose primary
    # connections are incoming (TextDocument→contains→DocumentChunk) are not
    # shown as isolated nodes.
    relationships_by_type: Dict[str, Counter] = defaultdict(Counter)
    for link in links_list:
        source_type = node_type_by_id.get(str(link["source"]))
        target_type = node_type_by_id.get(str(link["target"]))
        if source_type is None or target_type is None:
            continue
        if source_type in _INTERNAL_TYPES or target_type in _INTERNAL_TYPES:
            continue
        relation = _link_relation(link)
        relationships_by_type[source_type][(relation, target_type)] += 1
        relationships_by_type[target_type][(f"\u2190 {relation}", source_type)] += 1

    schema_nodes = []
    for node_type_name, type_nodes in sorted(nodes_by_type.items()):
        # Surface the most-common pipeline / task / user that produced this
        # type so the Schema card can show "produced by cognify_pipeline /
        # extract_graph_from_data" prominently rather than burying it as
        # one of many fields with a "string 100%" coverage label.
        pipe_counter: Counter = Counter()
        task_counter: Counter = Counter()
        user_counter: Counter = Counter()
        for tn in type_nodes:
            sp = tn.get("source_pipeline")
            st = tn.get("source_task")
            su = tn.get("source_user")
            if sp:
                pipe_counter[sp] += 1
            if st:
                task_counter[st] += 1
            if su:
                user_counter[su] += 1
        top_pipeline = pipe_counter.most_common(1)[0][0] if pipe_counter else None
        top_task = task_counter.most_common(1)[0][0] if task_counter else None
        top_user = user_counter.most_common(1)[0][0] if user_counter else None

        # Rank instances by descending degree, then name, so the sample list is
        # deterministic rather than dict-order-dependent. PR3's side panel reads
        # these names directly.
        ranked = sorted(
            type_nodes,
            key=lambda tn: (-(tn.get("degree") or 0), tn.get("name") or ""),
        )
        samples = [tn["name"] for tn in ranked[:SCHEMA_SAMPLES_PER_TYPE]]

        # Full per-pair relationship distribution for this source type, sorted by
        # descending count then target/relation as stable tiebreakers.
        relationships = sorted(
            (
                {"to_type": target_type, "relation": relation, "count": count}
                for (relation, target_type), count in relationships_by_type[node_type_name].items()
            ),
            key=lambda rel: (-rel["count"], rel["to_type"] or "", rel["relation"]),
        )

        schema_node = {
            "id": f"type:{node_type_name}",
            "name": node_type_name,
            "type": "GraphNodeType",
            "rank": _rank_for(node_type_name),
            "fields": extract_type_schema_fields(type_nodes),
            "source_pipeline": top_pipeline,
            "source_task": top_task,
            "source_user": top_user,
            "instance_count": len(type_nodes),
            "samples": samples,
            "sample_size": len(samples),
            "relationships": relationships,
        }
        if node_type_name == OTHER_ENTITY_TYPES_LABEL and rolled_up_types:
            schema_node["rollup"] = True
            schema_node["rolled_up_types"] = rolled_up_types
            # Lead the card with the tail size and its largest types so the
            # rollup is self-explanatory without inspector drill-down.
            top_tail = ", ".join(f"{t['name']} ({t['count']})" for t in rolled_up_types[:3])
            schema_node["fields"].insert(
                1,
                {
                    "name": "entity types",
                    "type": f"{len(rolled_up_types)} rolled up: {top_tail}, …",
                    "required": True,
                },
            )
        schema_nodes.append(schema_node)

    relation_counts_by_pair: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for link in links_list:
        source_type = node_type_by_id.get(str(link["source"]))
        target_type = node_type_by_id.get(str(link["target"]))
        if source_type is None or target_type is None:
            continue
        if source_type in _INTERNAL_TYPES or target_type in _INTERNAL_TYPES:
            continue
        relation_counts_by_pair[(source_type, target_type)][_link_relation(link)] += 1

    schema_links: List[Dict[str, Any]] = []
    for index, ((source_type, target_type), relation_counts) in enumerate(
        sorted(
            relation_counts_by_pair.items(),
            key=lambda item: (-sum(item[1].values()), item[0][0], item[0][1]),
        )
    ):
        rel_id = f"rel:{index}:{source_type}:{target_type}"
        source_rank = _rank_for(source_type)
        target_rank = _rank_for(target_type)
        if source_rank == target_rank:
            rel_rank = source_rank + 0.5
        else:
            rel_rank = (source_rank + target_rank) / 2

        schema_nodes.append(
            {
                "id": rel_id,
                "name": (
                    f"{source_type} self-links"
                    if source_type == target_type
                    else f"{source_type} to {target_type}"
                ),
                "type": "GraphRelationshipType",
                "rank": rel_rank,
                "source_type": source_type,
                "target_type": target_type,
                "relationship_label": _relationship_label(relation_counts),
                "edge_count": sum(relation_counts.values()),
                "fields": [
                    {
                        "name": "edges",
                        "type": str(sum(relation_counts.values())),
                        "required": True,
                    },
                    {
                        "name": "top relation",
                        "type": relation_counts.most_common(1)[0][0],
                        "required": True,
                    },
                    {
                        "name": "relation types",
                        "type": str(len(relation_counts)),
                        "required": True,
                    },
                ],
            }
        )
        schema_links.append({"source": f"type:{source_type}", "target": rel_id, "label": "from"})
        schema_links.append({"source": rel_id, "target": f"type:{target_type}", "label": "to"})

    # Instance-level drill-down data so the inspector can navigate
    # Type → instance → neighbours without dropping to the global graph:
    #   * instances_by_type: every instance name (not just the 5 samples), per type
    #   * instance_index: a compact per-instance adjacency (outgoing/incoming edges)
    # NOTE: for very large graphs this should be scoped/paginated; it is sized to
    # the schema graph (one entry per instance) which is fine for typical graphs.
    instances_by_type: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    instance_index: Dict[str, Dict[str, Any]] = {}
    for node in nodes_list:
        nid = str(node["id"])
        type_name = node_type_by_id[node["id"]]
        if type_name in _INTERNAL_TYPES:
            continue
        display_name = node.get("name") or nid
        instances_by_type[type_name].append({"id": nid, "name": display_name})
        instance_index[nid] = {
            "id": nid,
            "name": display_name,
            "type": type_name,
            "out": [],
            "in": [],
        }
    for type_name in instances_by_type:
        instances_by_type[type_name].sort(key=lambda rec: rec["name"])
    for link in links_list:
        source = str(link["source"])
        target = str(link["target"])
        if source not in instance_index or target not in instance_index:
            continue
        relation = _link_relation(link)
        instance_index[source]["out"].append({"relation": relation, "id": target})
        instance_index[target]["in"].append({"relation": relation, "id": source})

    return {
        "nodes": schema_nodes,
        "links": schema_links,
        "instances_by_type": dict(instances_by_type),
        "instance_index": instance_index,
    }


def build_operation_layer(
    schema_graph: Dict[str, Any],
    nodes_list: List[Dict[str, Any]],
    links_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach a transformation impact-layer to ``schema_graph`` in place.

    For each catalog operation whose effects touch a schema type present in the
    graph, emit an operation node and typed impact links
    (produces/enriches/modifies/removes). ``"Entity"`` effects expand to the
    semantic entity types actually present (Person/Broker/…). Links are flagged
    ``observed`` when the live provenance (a type's top ``source_pipeline``)
    matches the operation's pipeline. Existing nodes/links are left untouched.
    """
    from cognee.modules.visualization.operations_catalog import get_operations_catalog

    type_nodes = [n for n in schema_graph.get("nodes", []) if n.get("type") == "GraphNodeType"]
    present = {n["name"] for n in type_nodes}
    pipeline_by_type = {n["name"]: n.get("source_pipeline") for n in type_nodes}

    # Semantic entity types are those reached via the is_a edge (Person/Broker/…).
    nodes_by_id = {n["id"]: n for n in nodes_list}
    semantic_entity_types = set()
    for link in links_list:
        if _link_relation(link) == ENTITY_TYPE_RELATION:
            target = nodes_by_id.get(str(link["target"]))
            if target and target.get("name"):
                semantic_entity_types.add(target["name"])

    def resolve_targets(effect):
        names = set()
        target_type = effect.get("target_type")
        if target_type == "Entity":
            names |= semantic_entity_types & present
            if "Entity" in present:
                names.add("Entity")
        elif target_type and target_type in present:
            names.add(target_type)
        node_set = effect.get("target_node_set")
        if node_set and node_set in present:
            names.add(node_set)
        return names

    operations = []
    operation_links = []
    for op in get_operations_catalog():
        seen = set()
        links_for_op = []
        for effect in op.get("effects", []):
            for type_name in resolve_targets(effect):
                key = (effect["effect"], type_name)
                if key in seen:
                    continue
                seen.add(key)
                observed = op.get("pipeline_name") is not None and pipeline_by_type.get(
                    type_name
                ) == op.get("pipeline_name")
                links_for_op.append(
                    {
                        "source": "op:" + op["name"],
                        "target": "type:" + type_name,
                        "effect": effect["effect"],
                        "property": effect.get("property"),
                        "observed": bool(observed),
                    }
                )
        if not links_for_op:
            continue  # operation doesn't touch any type present in this graph
        operations.append(
            {
                "id": "op:" + op["name"],
                "name": op["label"],
                "type": "GraphOperation",
                "op_kind": op.get("kind", "pipeline"),
                "scope": op.get("scope", "subset"),
                "summary": op.get("summary", ""),
            }
        )
        operation_links.extend(links_for_op)

    schema_graph["operations"] = operations
    schema_graph["operation_links"] = operation_links
    return schema_graph


# ── Story-view enrichment ────────────────────────────────────────────────────


def _stage_for_node(node_info):
    node_type = node_info.get("type")
    return _STAGE_BY_TYPE.get(node_type, "other")


def _visual_rank(node_info, stage):
    """Pick the rank the Story layout reads.

    Prefer the runtime-stamped ``topological_rank`` (1-based, set by
    ``run_tasks_base._stamp_provenance`` as of Phase 1a). Fall back to a
    fixed stage order so legacy graphs without stamped ranks still render.
    """
    rank = node_info.get("topological_rank")
    if isinstance(rank, int) and rank > 0:
        return rank
    if isinstance(rank, float) and rank > 0:
        return int(rank)
    return STAGE_ORDER.index(stage) + 1 if stage in STAGE_ORDER else len(STAGE_ORDER)


def _edge_class(relation, edge_info):
    """Classify an edge so the renderer can bundle structural noise and keep
    semantic relations visible."""
    rel = (relation or "").lower()
    edge_relation = (edge_info or {}).get("relationship_name", "") if edge_info else ""
    edge_relation = (edge_relation or "").lower()

    if rel in _STRUCTURAL_RELATIONS or edge_relation in _STRUCTURAL_RELATIONS:
        return "structural"
    return "semantic"


def _bundle_key(source_stage, target_stage, edge_class, relation):
    return f"{source_stage}|{target_stage}|{edge_class}|{relation or ''}"


def _compact_provenance(node_info):
    """Only emit a provenance dict when at least one field is present, so the
    inspector can hide the section cleanly on legacy graphs."""
    keys = ("source_task", "source_pipeline", "source_node_set", "source_user")
    payload = {k: node_info.get(k) for k in keys if node_info.get(k)}
    return payload or None


# ── Public API ───────────────────────────────────────────────────────────────


@dataclass
class PreprocessedGraph:
    """Renderer-facing snapshot of a cognee graph.

    Every Python-side derivation the JavaScript renderer needs to make a
    correct, readable visualization is computed once here. The renderer
    should not synthesize stage/rank/edge_class/etc. on its own.
    """

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    color_maps: Dict[str, Dict[str, str]] = field(default_factory=dict)
    schema_graph: Dict[str, Any] = field(default_factory=lambda: {"nodes": [], "links": []})
    schema_data: Optional[Dict[str, Any]] = None
    pipeline_stages: List[str] = field(default_factory=list)
    edge_classes: Dict[str, int] = field(default_factory=dict)
    bundles: Dict[str, int] = field(default_factory=dict)
    provenance_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_meaningful_topological_rank: bool = False


def _label_priority_threshold(importances: List[float], percentile: float = 0.75) -> float:
    """Return the importance threshold above which a node earns a Key-mode label."""
    if not importances:
        return 0.0
    finite = sorted(v for v in importances if math.isfinite(v))
    if not finite:
        return 0.0
    rank = max(0, min(len(finite) - 1, int(percentile * (len(finite) - 1))))
    return finite[rank]


# Node types whose labels we always show in Key mode regardless of degree —
# documents/types are the natural landmarks of the Story view.
_ALWAYS_LABEL_STAGES = frozenset({"document", "type"})


def preprocess(graph_data, schema_data: Optional[Dict[str, Any]] = None) -> PreprocessedGraph:
    """Turn raw ``(nodes_data, edges_data)`` into a fully-enriched snapshot.

    Mirrors the data shape the existing ``cognee_network_visualization``
    function builds, plus new renderer-facing fields the Story view needs:
    ``stage``, ``visual_rank``, ``degree``, ``importance``, ``label_priority``,
    ``provenance`` on nodes, and ``edge_class``, ``bundle_key`` on links.

    The original color maps (task/pipeline/node_set/user) and schema graph
    are preserved verbatim so any existing renderer code that reads them
    continues to work without modification.
    """
    nodes_data, edges_data = graph_data

    # ── Nodes pass 1: normalize, color, name, stage ────────────────────────
    nodes: List[Dict[str, Any]] = []
    node_ids_seen: set = set()
    has_meaningful_rank = False

    for node_id, node_info in nodes_data:
        node_info = dict(node_info) if not isinstance(node_info, dict) else node_info.copy()
        sid = str(node_id)
        node_info["id"] = sid
        node_info["color"] = _TYPE_COLOR_MAP.get(
            node_info.get("type", "default"), _UNKNOWN_TYPE_COLOR
        )
        if node_info.get("ontology_valid") is True:
            node_info["color"] = _ONTOLOGY_VALID_COLOR
        node_info["name"] = derive_node_name(node_info, node_id)
        node_info.pop("updated_at", None)
        node_info.pop("created_at", None)

        stage = _stage_for_node(node_info)
        node_info["stage"] = stage
        node_info["visual_rank"] = _visual_rank(node_info, stage)
        node_info["degree"] = 0  # filled in pass 2
        node_info["importance"] = 0.0  # filled in pass 2
        node_info["label_priority"] = False  # filled in pass 3

        prov = _compact_provenance(node_info)
        if prov is not None:
            node_info["provenance"] = prov

        rank = node_info.get("topological_rank")
        if (isinstance(rank, int) or isinstance(rank, float)) and rank not in (None, 0):
            has_meaningful_rank = True

        nodes.append(node_info)
        node_ids_seen.add(sid)

    nodes_by_id = {n["id"]: n for n in nodes}

    # ── Links pass: normalize, classify, weight, bundle ────────────────────
    links: List[Dict[str, Any]] = []
    edge_class_counts: Counter = Counter()
    bundle_counts: Counter = Counter()
    degree_counter: Counter = Counter()

    for edge in edges_data:
        # graph adapter may return 3- or 4-tuples; tolerate both
        if len(edge) >= 4:
            source, target, relation, edge_info = edge[0], edge[1], edge[2], edge[3]
        else:
            source, target, relation = edge[0], edge[1], edge[2]
            edge_info = {}

        source = str(source)
        target = str(target)

        all_weights: Dict[str, float] = {}
        primary_weight: Optional[float] = None
        if edge_info:
            if "weight" in edge_info:
                all_weights["default"] = edge_info["weight"]
                primary_weight = edge_info["weight"]
            if "weights" in edge_info and isinstance(edge_info["weights"], dict):
                all_weights.update(edge_info["weights"])
                if primary_weight is None and edge_info["weights"]:
                    primary_weight = next(iter(edge_info["weights"].values()))
            for key, value in edge_info.items():
                if key.startswith("weight_") and isinstance(value, (int, float)):
                    all_weights[key[7:]] = value

        edge_cls = _edge_class(relation, edge_info)
        source_stage = nodes_by_id.get(source, {}).get("stage", "other")
        target_stage = nodes_by_id.get(target, {}).get("stage", "other")
        bundle_key = _bundle_key(source_stage, target_stage, edge_cls, relation)

        edge_class_counts[edge_cls] += 1
        bundle_counts[bundle_key] += 1
        degree_counter[source] += 1
        degree_counter[target] += 1

        links.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "weight": primary_weight,
                "all_weights": all_weights,
                "relationship_type": (edge_info.get("relationship_type") if edge_info else None),
                "edge_info": edge_info if edge_info else {},
                "edge_class": edge_cls,
                "bundle_key": bundle_key,
                "source_stage": source_stage,
                "target_stage": target_stage,
            }
        )

    # ── Nodes pass 2: degree, importance ───────────────────────────────────
    for node in nodes:
        deg = degree_counter.get(node["id"], 0)
        node["degree"] = deg
        # log-scaled, capped — importance is a normalized 0..1 visual weight,
        # not a semantic score, so the renderer can size labels/halos cleanly.
        node["importance"] = math.log1p(deg) / math.log1p(
            max(1, max(degree_counter.values() or [1]))
        )

    # ── Nodes pass 3: label priority budget ────────────────────────────────
    importances = [n["importance"] for n in nodes]
    threshold = _label_priority_threshold(importances, percentile=0.75)
    for node in nodes:
        if node["stage"] in _ALWAYS_LABEL_STAGES:
            node["label_priority"] = True
        elif node["importance"] >= threshold and threshold > 0:
            node["label_priority"] = True
        else:
            node["label_priority"] = False

    # ── Color maps (verbatim shape from the original orchestrator) ─────────
    color_maps = {
        "task": generate_provenance_colors([n.get("source_task") for n in nodes]),
        "pipeline": generate_provenance_colors([n.get("source_pipeline") for n in nodes]),
        "node_set": generate_provenance_colors([n.get("source_node_set") for n in nodes]),
        "user": generate_provenance_colors([n.get("source_user") for n in nodes]),
    }

    schema_graph = extract_schema_graph_data(nodes, links)
    build_operation_layer(schema_graph, nodes, links)

    # Stages present in the graph, in canonical left-to-right order
    present_stages = [s for s in STAGE_ORDER if any(n["stage"] == s for n in nodes)]

    provenance_index = {n["id"]: n["provenance"] for n in nodes if n.get("provenance")}

    return PreprocessedGraph(
        nodes=nodes,
        links=links,
        color_maps=color_maps,
        schema_graph=schema_graph,
        schema_data=schema_data,
        pipeline_stages=present_stages,
        edge_classes=dict(edge_class_counts),
        bundles=dict(bundle_counts),
        provenance_index=provenance_index,
        has_meaningful_topological_rank=has_meaningful_rank,
    )
