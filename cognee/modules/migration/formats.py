"""Pure format emitters for graph export.

Each emitter takes the ``get_graph_data()`` shapes — nodes as
``(node_id, properties)`` tuples, edges as ``(source_id, target_id,
relationship_name, properties)`` tuples — and writes a file. No database
access here; emitters are pure and unit-testable.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape, quoteattr

Node = Tuple[Any, Dict[str, Any]]
Edge = Tuple[Any, Any, str, Dict[str, Any]]

# Properties that are internal bookkeeping rather than knowledge content.
_SKIP_EDGE_KEYS = ("source_node_id", "target_node_id")


def _scalar(value: Any) -> Any:
    """Coerce a property value to a JSON/graph-safe scalar."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str)


def write_json(nodes: List[Node], edges: List[Edge], destination: Path) -> None:
    """Full-fidelity JSON: every node and edge with all properties."""
    payload = {
        "nodes": [{"id": str(node_id), **properties} for node_id, properties in nodes],
        "edges": [
            {
                "source": str(source),
                "target": str(target),
                "relationship_name": relationship,
                **{k: v for k, v in (properties or {}).items() if k not in _SKIP_EDGE_KEYS},
            }
            for source, target, relationship, properties in edges
        ],
    }
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_graphml(nodes: List[Node], edges: List[Edge], destination: Path) -> None:
    """GraphML for Gephi/yEd/NetworkX interop. Property values become strings."""
    node_keys: Dict[str, None] = {}
    edge_keys: Dict[str, None] = {}
    for _, properties in nodes:
        for key in properties or {}:
            node_keys.setdefault(key)
    # Every edge emits a "relationship_name" datum below. It comes from the edge
    # tuple rather than the per-edge properties dict, so the loop below would never
    # register it; declare it explicitly whenever edges are present, otherwise the
    # emitted <data key="e_relationship_name"> has no matching <key> declaration and
    # strict GraphML readers (e.g. networkx.read_graphml) reject the file.
    if edges:
        edge_keys.setdefault("relationship_name")
    for _, _, _, properties in edges:
        for key in properties or {}:
            if key not in _SKIP_EDGE_KEYS:
                edge_keys.setdefault(key)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    for key in node_keys:
        lines.append(
            f'  <key id="n_{escape(key)}" for="node" attr.name={quoteattr(key)} attr.type="string"/>'
        )
    for key in edge_keys:
        lines.append(
            f'  <key id="e_{escape(key)}" for="edge" attr.name={quoteattr(key)} attr.type="string"/>'
        )
    lines.append('  <graph id="cognee" edgedefault="directed">')

    for node_id, properties in nodes:
        lines.append(f"    <node id={quoteattr(str(node_id))}>")
        for key, value in (properties or {}).items():
            if value is None:
                continue
            lines.append(f'      <data key="n_{escape(key)}">{escape(str(_scalar(value)))}</data>')
        lines.append("    </node>")

    for source, target, relationship, properties in edges:
        lines.append(f"    <edge source={quoteattr(str(source))} target={quoteattr(str(target))}>")
        lines.append(f'      <data key="e_relationship_name">{escape(str(relationship))}</data>')
        for key, value in (properties or {}).items():
            if value is None or key in _SKIP_EDGE_KEYS or key == "relationship_name":
                continue
            lines.append(f'      <data key="e_{escape(key)}">{escape(str(_scalar(value)))}</data>')
        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")
    destination.write_text("\n".join(lines), encoding="utf-8")


def _cypher_props(properties: Dict[str, Any]) -> str:
    parts = []
    for key, value in properties.items():
        if value is None:
            continue
        safe_key = "".join(ch for ch in key if ch.isalnum() or ch == "_") or "prop"
        parts.append(f"`{safe_key}`: {json.dumps(_scalar(value), default=str)}")
    return "{" + ", ".join(parts) + "}"


def _cypher_label(value: Any) -> str:
    label = str(value or "Node")
    return "".join(ch for ch in label if ch.isalnum() or ch == "_") or "Node"


# Shared label on every exported node so edge MATCH clauses are index-backed.
_SHARED_LABEL = "CogneeNode"


def write_cypher(nodes: List[Node], edges: List[Edge], destination: Path) -> None:
    """A Cypher script of MERGE statements loadable into any Neo4j-compatible DB.

    Every node gets the shared ``:CogneeNode`` label (its sanitized ``type``
    becomes a secondary label), and an index on ``(:CogneeNode).id`` is created
    up front so per-edge MATCH clauses are index lookups instead of full scans.
    """
    lines = [
        "// Cognee graph export — load with cypher-shell or neo4j browser",
        f"CREATE INDEX IF NOT EXISTS FOR (n:{_SHARED_LABEL}) ON (n.id);",
    ]
    for node_id, properties in nodes:
        properties = dict(properties or {})
        label = _cypher_label(properties.get("type"))
        properties["id"] = str(node_id)
        lines.append(
            f"MERGE (n:{_SHARED_LABEL} {{id: {json.dumps(str(node_id))}}}) "
            f"SET n:`{label}`, n += {_cypher_props(properties)};"
        )

    for source, target, relationship, properties in edges:
        properties = {
            k: v
            for k, v in (properties or {}).items()
            if k not in _SKIP_EDGE_KEYS and v is not None
        }
        rel_type = _cypher_label(relationship)
        lines.append(
            f"MATCH (a:{_SHARED_LABEL} {{id: {json.dumps(str(source))}}}), "
            f"(b:{_SHARED_LABEL} {{id: {json.dumps(str(target))}}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) SET r += {_cypher_props(properties)};"
        )
    destination.write_text("\n".join(lines), encoding="utf-8")
