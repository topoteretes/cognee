"""Render a CODE search result as an architecture diagram (Mermaid or Graphviz DOT).

Every ``SearchType.CODE`` operation returns structured facts and edges. When
the caller asks for one (``code_query={"diagram": "mermaid"}``), the same
result is additionally rendered as diagram source text, so a UI, a README,
an agent's reply or a chat client can show the architecture instead of a
JSON blob. Rendering is deterministic and purely textual: the same result
always produces byte-identical source, and nothing here talks to a model,
a database or the network.

Mermaid is the default because it renders natively on GitHub, in most docs
sites and chat clients; DOT is for Graphviz tooling.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional

DIAGRAM_FORMATS = ("mermaid", "dot")

# Fact kinds -> Mermaid node shape (open, close) and Graphviz shape name.
_MERMAID_SHAPES = {
    "module": ("[[", "]]"),
    "symbol": ("([", "])"),
    "route": (">", "]"),
    "storage": ("[(", ")]"),
    "dependency": ("{{", "}}"),
    "service": ("[/", "/]"),
    "insight": ("{", "}"),
    "intent": ("[/", "\\]"),
    "repository": ("[[", "]]"),
}
_DOT_SHAPES = {
    "module": "component",
    "symbol": "ellipse",
    "route": "cds",
    "storage": "cylinder",
    "dependency": "hexagon",
    "service": "box3d",
    "insight": "diamond",
    "intent": "parallelogram",
    "repository": "folder",
}
_ROOT_MODULE_LABEL = "(root)"


def _label(node: Mapping[str, Any]) -> str:
    name = str(node.get("name") or node.get("id") or "")
    if node.get("kind") == "module" and name == ".":
        return _ROOT_MODULE_LABEL
    return name


def _mermaid_text(text: str) -> str:
    """Escape a node/edge label for Mermaid using its entity codes.

    '#' goes first, or the codes below would be re-escaped; '&' is escaped
    because Mermaid renders labels as HTML and would read it as an entity.
    """
    return (
        text.replace("#", "#35;")
        .replace("&", "#38;")
        .replace('"', "#quot;")
        .replace("<", "#lt;")
        .replace(">", "#gt;")
        .replace("\n", " ")
    )


def _yaml_string(text: str) -> str:
    """A double-quoted YAML scalar, for the Mermaid front-matter title.

    Mermaid HTML-escapes the title and then draws the escaped text literally
    (``Vec<T>`` shows as ``Vec&lt;T&gt;``), so angle brackets become their
    single-guillemet look-alikes, which read the same and render verbatim.
    """
    text = text.replace("<", "\u2039").replace(">", "\u203a")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _dot_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _edge_label(edge: Mapping[str, Any]) -> str:
    label = str(edge.get("type") or "")
    count = edge.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 1:
        label = f"{label} x{count}"
    return label


def render_mermaid(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
    *,
    highlight_ids: Iterable[str] = (),
    title: Optional[str] = None,
) -> str:
    """A Mermaid ``flowchart LR`` of the given facts and edges.

    Nodes are grouped into one subgraph per repository when more than one
    repository is present; shapes follow the fact kind; highlighted nodes
    (the seeds, focus or path of the operation) get a distinct class.
    """
    nodes = list(nodes)
    alias = {str(node["id"]): f"n{index}" for index, node in enumerate(nodes)}
    highlighted = [alias[node_id] for node_id in highlight_ids if node_id in alias]

    lines = []
    if title:
        # The front matter is YAML, not Mermaid text: quote it as YAML, or a
        # '#' in a title starts a comment and a ':' breaks the mapping.
        lines.extend(["---", f"title: {_yaml_string(title)}", "---"])
    lines.append("flowchart LR")

    by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_repo[str(node.get("repo") or "")].append(node)

    def node_line(node: Mapping[str, Any], indent: str) -> str:
        open_shape, close_shape = _MERMAID_SHAPES.get(str(node.get("kind")), ("[", "]"))
        return f'{indent}{alias[str(node["id"])]}{open_shape}"{_mermaid_text(_label(node))}"{close_shape}'

    if len(by_repo) > 1:
        for index, repo in enumerate(sorted(by_repo)):
            lines.append(f'    subgraph repo{index}["{_mermaid_text(repo or "(no repo)")}"]')
            lines.extend(node_line(node, "        ") for node in by_repo[repo])
            lines.append("    end")
    else:
        lines.extend(node_line(node, "    ") for node in nodes)

    for edge in edges:
        source = alias.get(str(edge.get("source_id")))
        target = alias.get(str(edge.get("target_id")))
        if source is None or target is None:
            continue
        label = _mermaid_text(_edge_label(edge))
        arrow = f'-- "{label}" -->' if label else "-->"
        lines.append(f"    {source} {arrow} {target}")

    if highlighted:
        lines.append("    classDef highlight stroke:#d64545,stroke-width:3px")
        lines.append(f"    class {','.join(highlighted)} highlight")
    return "\n".join(lines) + "\n"


def render_dot(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
    *,
    highlight_ids: Iterable[str] = (),
    title: Optional[str] = None,
) -> str:
    """A Graphviz ``digraph`` of the given facts and edges (same grouping as Mermaid)."""
    nodes = list(nodes)
    alias = {str(node["id"]): f"n{index}" for index, node in enumerate(nodes)}
    highlight = {node_id for node_id in highlight_ids if node_id in alias}

    lines = ["digraph code_graph {", "    rankdir=LR;", '    node [fontname="Helvetica"];']
    if title:
        lines.append(f'    label="{_dot_text(title)}"; labelloc=t;')

    by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_repo[str(node.get("repo") or "")].append(node)

    def node_line(node: Mapping[str, Any], indent: str) -> str:
        node_id = str(node["id"])
        attributes = [
            f'label="{_dot_text(_label(node))}"',
            f"shape={_DOT_SHAPES.get(str(node.get('kind')), 'box')}",
        ]
        if node_id in highlight:
            attributes.append('color="#d64545" penwidth=3')
        return f"{indent}{alias[node_id]} [{' '.join(attributes)}];"

    if len(by_repo) > 1:
        for index, repo in enumerate(sorted(by_repo)):
            lines.append(f"    subgraph cluster_{index} {{")
            lines.append(f'        label="{_dot_text(repo or "(no repo)")}";')
            lines.extend(node_line(node, "        ") for node in by_repo[repo])
            lines.append("    }")
    else:
        lines.extend(node_line(node, "    ") for node in nodes)

    for edge in edges:
        source = alias.get(str(edge.get("source_id")))
        target = alias.get(str(edge.get("target_id")))
        if source is None or target is None:
            continue
        label = _dot_text(_edge_label(edge))
        lines.append(
            f'    {source} -> {target} [label="{label}"];'
            if label
            else f"    {source} -> {target};"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def collect_diagram_graph(
    result: Mapping[str, Any],
) -> Optional[tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]]:
    """Extract (nodes, edges, highlight_ids) from any CODE operation result.

    Each operation shapes its result differently; this is the one place that
    knows those shapes, so the renderers stay format-only. Returns None for
    results that carry no graph (``delta``).
    """
    operation = result.get("operation")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    highlight: list[str] = []
    seen: set[str] = set()

    def add_node(fact: Any) -> None:
        if isinstance(fact, Mapping) and fact.get("id") is not None:
            node_id = str(fact["id"])
            if node_id not in seen:
                seen.add(node_id)
                nodes.append(dict(fact))

    if operation in {"explore", "traverse", "architecture"}:
        for fact in result.get("nodes") or []:
            add_node(fact)
        edges = [dict(edge) for edge in result.get("edges") or []]
        focus = result.get("focus")
        if isinstance(focus, Mapping):
            add_node(focus)
            highlight.append(str(focus["id"]))
        highlight.extend(
            str(fact["id"])
            for fact in result.get("nodes") or []
            if isinstance(fact, Mapping) and fact.get("depth") == 0
        )
    elif operation == "find_path":
        for key in ("from", "to", "matched_to"):
            add_node(result.get(key))
        for fact in result.get("path") or []:
            add_node(fact)
            highlight.append(str(fact["id"]))
        edges = [dict(edge) for edge in result.get("edges") or []]
    elif operation == "impact_analysis":
        for key in ("targets", "impact_seeds"):
            for fact in result.get(key) or []:
                add_node(fact)
                highlight.append(str(fact["id"]))
        by_depth = result.get("by_depth") or {}
        for depth in sorted(by_depth, key=lambda value: int(value)):
            for fact in by_depth[depth]:
                add_node(fact)
        edges = [dict(edge) for edge in result.get("edges") or []]
    elif operation == "query_facts":
        for fact in result.get("facts") or []:
            add_node(fact)
        for fact in result.get("facts") or []:
            for relation in fact.get("relations") or []:
                target_id = str(relation.get("target_id"))
                if target_id in seen:
                    edges.append(
                        {
                            "source_id": str(fact["id"]),
                            "target_id": target_id,
                            "type": relation.get("type"),
                        }
                    )
    elif operation == "insights":
        for insight in result.get("insights") or []:
            add_node(insight)
            highlight.append(str(insight["id"]))
            for fact in insight.get("evidence") or []:
                add_node(fact)
                edges.append(
                    {
                        "source_id": str(insight["id"]),
                        "target_id": str(fact["id"]),
                        "type": "evidences",
                    }
                )
    else:
        return None

    # Deterministic, de-duplicated edge order regardless of how the
    # operation happened to list them.
    unique_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (str(edge.get("source_id")), str(edge.get("target_id")), str(edge.get("type")))
        unique_edges.setdefault(key, edge)
    ordered_edges = [unique_edges[key] for key in sorted(unique_edges)]
    unique_highlight = list(dict.fromkeys(highlight))
    return nodes, ordered_edges, unique_highlight


def render_result_diagram(result: Mapping[str, Any], diagram_format: str) -> dict[str, Any]:
    """The ``diagram`` block attached to a CODE result: format, source and counts."""
    if diagram_format not in DIAGRAM_FORMATS:
        raise ValueError(f"Unsupported diagram format {diagram_format!r}.")
    collected = collect_diagram_graph(result)
    if collected is None:
        return {
            "format": diagram_format,
            "source": None,
            "nodes": 0,
            "edges": 0,
            "note": f"The {result.get('operation')!r} operation carries no graph to draw.",
        }
    nodes, edges, highlight = collected
    title = _diagram_title(result)
    renderer = render_mermaid if diagram_format == "mermaid" else render_dot
    return {
        "format": diagram_format,
        "source": renderer(nodes, edges, highlight_ids=highlight, title=title),
        "nodes": len(nodes),
        "edges": len(edges),
    }


def _diagram_title(result: Mapping[str, Any]) -> Optional[str]:
    operation = str(result.get("operation") or "")
    if operation == "explore" and isinstance(result.get("focus"), Mapping):
        return f"explore: {result['focus'].get('name')}"
    if operation == "find_path":
        source = result.get("from") or {}
        target = result.get("to") or {}
        return f"path: {source.get('name')} -> {target.get('name')}"
    if operation == "impact_analysis":
        names = [str(fact.get("name")) for fact in result.get("targets") or []]
        return f"impact of {', '.join(names)}" if names else "impact analysis"
    if operation == "architecture":
        repos = result.get("repos") or []
        return f"architecture: {', '.join(repos)}" if repos else "architecture"
    return operation or None
