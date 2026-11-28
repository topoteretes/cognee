"""Pure format emitters for graph export.

Each emitter takes the ``get_graph_data()`` shapes — nodes as
``(node_id, properties)`` tuples, edges as ``(source_id, target_id,
relationship_name, properties)`` tuples — and writes a file. No database
access here; emitters are pure and unit-testable.
"""

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

Node = tuple[Any, dict[str, Any]]
Edge = tuple[Any, Any, str, dict[str, Any]]

# Properties that are internal bookkeeping rather than knowledge content.
_SKIP_EDGE_KEYS = ("source_node_id", "target_node_id")


def _xml_illegal(char: str) -> bool:
    """Whether a character falls outside the XML 1.0 ``Char`` production.

    That production allows tab/newline/carriage-return, then U+0020-U+D7FF,
    U+E000-U+FFFD and U+10000-U+10FFFF. Everything else — the other C0 controls,
    the surrogate block (lone surrogates survive in a Python str) and
    U+FFFE/U+FFFF — cannot appear in a well-formed document, escaped or not.
    """
    code_point = ord(char)
    return not (
        code_point in (0x09, 0x0A, 0x0D)
        or 0x20 <= code_point <= 0xD7FF
        or 0xE000 <= code_point <= 0xFFFD
        or 0x10000 <= code_point <= 0x10FFFF
    )


def _xml_text(value: str, field: str) -> str:
    """Return ``value`` unchanged, or raise if it holds a character GraphML can't carry.

    GraphML is XML 1.0, so a code point outside its ``Char`` range cannot be
    escaped into a well-formed file. We fail here rather than write a document no
    parser will read; ``write_json`` keeps full fidelity for values that need it.
    """
    for char in value:
        if _xml_illegal(char):
            raise ValueError(
                f"graphml cannot represent U+{ord(char):04X} in {field}; "
                "export as json to keep it"
            )
    return value


def _scalar(value: Any) -> Any:
    """Coerce a property value to a JSON/graph-safe scalar."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str)


def write_json(nodes: list[Node], edges: list[Edge], destination: Path) -> None:
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


def write_graphml(nodes: list[Node], edges: list[Edge], destination: Path) -> None:
    """GraphML for Gephi/yEd/NetworkX interop. Property values become strings."""
    node_keys: dict[str, None] = {}
    edge_keys: dict[str, None] = {}
    for _, properties in nodes:
        for key in properties or {}:
            node_keys.setdefault(key)
    for _, _, _, properties in edges:
        for key in properties or {}:
            if key not in _SKIP_EDGE_KEYS:
                edge_keys.setdefault(key)
    # Every edge emits a <data key="e_relationship_name"> element, so its <key>
    # must be declared too or strict readers (e.g. networkx.read_graphml) reject
    # the document.
    if edges:
        edge_keys.setdefault("relationship_name")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    for key in node_keys:
        safe = _xml_text(key, f"node property name {key!r}")
        lines.append(
            f'  <key id="n_{escape(safe)}" for="node" attr.name={quoteattr(safe)} attr.type="string"/>'
        )
    for key in edge_keys:
        safe = _xml_text(key, f"edge property name {key!r}")
        lines.append(
            f'  <key id="e_{escape(safe)}" for="edge" attr.name={quoteattr(safe)} attr.type="string"/>'
        )
    lines.append('  <graph id="cognee" edgedefault="directed">')

    for node_id, properties in nodes:
        node_id = _xml_text(str(node_id), f"node id {node_id!r}")
        lines.append(f"    <node id={quoteattr(node_id)}>")
        for key, value in (properties or {}).items():
            if value is None:
                continue
            safe_key = _xml_text(key, f"node property name {key!r}")
            text = _xml_text(str(_scalar(value)), f"node {node_id!r} property {key!r}")
            lines.append(f'      <data key="n_{escape(safe_key)}">{escape(text)}</data>')
        lines.append("    </node>")

    for source, target, relationship, properties in edges:
        source = _xml_text(str(source), f"edge source {source!r}")
        target = _xml_text(str(target), f"edge target {target!r}")
        relationship = _xml_text(str(relationship), f"edge relationship_name {relationship!r}")
        lines.append(f"    <edge source={quoteattr(source)} target={quoteattr(target)}>")
        lines.append(f'      <data key="e_relationship_name">{escape(relationship)}</data>')
        for key, value in (properties or {}).items():
            if value is None or key in _SKIP_EDGE_KEYS or key == "relationship_name":
                continue
            safe_key = _xml_text(key, f"edge property name {key!r}")
            text = _xml_text(
                str(_scalar(value)), f"edge {source!r}->{target!r} property {key!r}"
            )
            lines.append(f'      <data key="e_{escape(safe_key)}">{escape(text)}</data>')
        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")
    destination.write_text("\n".join(lines), encoding="utf-8")


def _cypher_props(properties: dict[str, Any]) -> str:
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


def write_cypher(nodes: list[Node], edges: list[Edge], destination: Path) -> None:
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
