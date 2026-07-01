"""Graph insight report helpers.

This module works on the raw ``GraphDBInterface.get_graph_data()`` shape:
``(nodes, edges)``, where nodes are ``(id, properties)`` and edges are
``(source, target, relation, properties)``.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, Field

_IDENTIFIER_LIKE_RE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,64})$",
    re.IGNORECASE,
)

_NODE_SET_KEYS = ("source_node_set", "node_set", "node_sets", "belongs_to_set")
_SOURCE_KEYS = (
    "source_content_hash",
    "source_document",
    "source_file",
    "document_name",
    "file_name",
    "raw_data_location",
    "data_id",
)
_CONFIDENCE_KEYS = (
    "confidence_tag",
    "confidence_type",
    "extraction_type",
    "source_type",
    "edge_origin",
    "origin",
)


class GraphReportQuestions(BaseModel):
    """Structured output for LLM-generated report questions."""

    questions: list[str] = Field(
        default_factory=list,
        description="Four to five short questions suggested by the graph insight report.",
    )


def build_graph_report(
    graph_data: tuple[Iterable[Any], Iterable[Any]],
    *,
    top_n: int = 10,
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
    suggested_questions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a deterministic graph insight report from raw graph data."""
    top_n = max(int(top_n), 0)
    nodes_data, edges_data = graph_data
    nodes = _normalize_nodes(nodes_data)
    edges = _normalize_edges(edges_data)
    scoped_node_ids, scoped_edges = _scope_graph(nodes, edges, node_name, node_name_filter_operator)

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in scoped_node_ids}
    degree: Counter[str] = Counter({node_id: 0 for node_id in scoped_node_ids})
    usable_edges = []

    for edge in scoped_edges:
        source = edge["source"]
        target = edge["target"]
        if source == target or source not in nodes or target not in nodes:
            continue
        usable_edges.append(edge)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        degree[source] += 1
        degree[target] += 1

    page_rank = _compute_page_rank(scoped_node_ids, adjacency)
    hub_nodes = _rank_hub_nodes(nodes, scoped_node_ids, degree, page_rank, top_n)
    confidence_tags = _summarize_confidence_tags(usable_edges)
    surprising_connections = _rank_surprising_connections(nodes, usable_edges, top_n)
    questions = suggested_questions or suggest_questions_from_hubs(hub_nodes)

    return {
        "summary": {
            "node_count": len(scoped_node_ids),
            "edge_count": len(usable_edges),
            "scoped_node_sets": sorted(node_name or []),
        },
        "hub_nodes": hub_nodes,
        "surprising_connections": surprising_connections,
        "confidence_tags": confidence_tags,
        "suggested_questions": questions,
    }


async def build_graph_report_with_suggested_questions(
    graph_data: tuple[Iterable[Any], Iterable[Any]],
    *,
    top_n: int = 10,
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
    use_llm_questions: bool = True,
) -> dict[str, Any]:
    """Build a graph report and optionally ask the configured LLM for questions."""
    report = build_graph_report(
        graph_data,
        top_n=top_n,
        node_name=node_name,
        node_name_filter_operator=node_name_filter_operator,
    )
    if use_llm_questions:
        llm_questions = await generate_suggested_questions(report)
        if llm_questions:
            report["suggested_questions"] = llm_questions
    return report


async def generate_suggested_questions(
    report: dict[str, Any],
    *,
    question_count: int = 5,
) -> list[str]:
    """Generate suggested recall questions using one best-effort LLM call."""
    hubs = report.get("hub_nodes") or []
    connections = report.get("surprising_connections") or []
    if not hubs and not connections:
        return []

    hub_lines = [
        f"- {hub.get('name')} ({hub.get('type')}), degree {hub.get('degree')}" for hub in hubs[:8]
    ]
    connection_lines = [
        f"- {item.get('source', {}).get('name')} {item.get('relation')} "
        f"{item.get('target', {}).get('name')}"
        for item in connections[:5]
    ]
    prompt = "\n".join(
        [
            "Create concise recall questions for this knowledge graph report.",
            "Focus on useful questions a user could ask Cognee next.",
            "",
            "Hub nodes:",
            *hub_lines,
            "",
            "Surprising links:",
            *connection_lines,
        ]
    )
    system_prompt = (
        "You write short, specific graph-recall questions. "
        "Return only questions that can be answered from the graph context."
    )

    try:
        from cognee.infrastructure.llm import LLMGateway

        response = await LLMGateway.acreate_structured_output(
            text_input=prompt,
            system_prompt=system_prompt,
            response_model=GraphReportQuestions,
        )
    except Exception:
        return []

    questions = [question.strip() for question in response.questions if question.strip()]
    return questions[:question_count]


def suggest_questions_from_hubs(
    hub_nodes: list[dict[str, Any]],
    *,
    question_count: int = 5,
) -> list[str]:
    """Create deterministic fallback questions when LLM question generation is unavailable."""
    if not hub_nodes:
        return []

    questions = []
    for hub in hub_nodes[:question_count]:
        name = hub.get("name") or hub.get("id")
        questions.append(f"What makes {name} central in this knowledge graph?")

    if len(hub_nodes) >= 2 and len(questions) < question_count:
        questions.append(
            f"How are {hub_nodes[0].get('name')} and {hub_nodes[1].get('name')} connected?"
        )

    return questions[:question_count]


def render_graph_report_markdown(report: dict[str, Any]) -> str:
    """Render a graph report dictionary as Markdown."""
    summary = report.get("summary") or {}
    lines = [
        "# Graph Insight Report",
        "",
        "## Summary",
        "",
        f"- Nodes analyzed: {summary.get('node_count', 0)}",
        f"- Edges analyzed: {summary.get('edge_count', 0)}",
    ]
    scoped_node_sets = summary.get("scoped_node_sets") or []
    dataset = report.get("dataset") or {}
    if dataset.get("name"):
        lines.append(f"- Dataset: {dataset['name']}")
    if scoped_node_sets:
        lines.append(f"- Scoped node sets: {', '.join(scoped_node_sets)}")

    lines.extend(
        [
            "",
            "## Hub Nodes",
            "",
            "| Rank | Node | Type | Degree | PageRank | Node sets |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for rank, hub in enumerate(report.get("hub_nodes") or [], 1):
        lines.append(
            "| {rank} | {name} | {type} | {degree} | {pagerank:.4f} | {node_sets} |".format(
                rank=rank,
                name=_markdown_cell(hub.get("name")),
                type=_markdown_cell(hub.get("type")),
                degree=hub.get("degree", 0),
                pagerank=hub.get("pagerank", 0.0),
                node_sets=_markdown_cell(", ".join(hub.get("node_sets") or [])),
            )
        )

    lines.extend(
        [
            "",
            "## Surprising Connections",
            "",
            "| Rank | Source | Relation | Target | Novelty | Confidence | Why it stands out |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for rank, item in enumerate(report.get("surprising_connections") or [], 1):
        source = item.get("source") or {}
        target = item.get("target") or {}
        connection_row = (
            "| {rank} | {source} | {relation} | {target} | {score:.2f} | {confidence} | {reason} |"
        )
        lines.append(
            connection_row.format(
                rank=rank,
                source=_markdown_cell(source.get("name")),
                relation=_markdown_cell(item.get("relation")),
                target=_markdown_cell(target.get("name")),
                score=item.get("novelty_score", 0.0),
                confidence=_markdown_cell(item.get("confidence_tag")),
                reason=_markdown_cell("; ".join(item.get("reasons") or [])),
            )
        )

    confidence_tags = report.get("confidence_tags") or {}
    lines.extend(["", "## Confidence Tags", ""])
    for tag in ("EXTRACTED", "INFERRED", "UNKNOWN"):
        lines.append(f"- {tag}: {confidence_tags.get(tag, 0)}")

    lines.extend(["", "## Suggested Questions", ""])
    questions = report.get("suggested_questions") or []
    if questions:
        lines.extend(f"{index}. {question}" for index, question in enumerate(questions, 1))
    else:
        lines.append("No suggested questions were generated.")

    return "\n".join(lines).strip() + "\n"


def write_graph_report_markdown(report: dict[str, Any], destination_file_path: str) -> str:
    """Write a Markdown graph report and return the resolved file path."""
    destination = Path(destination_file_path).expanduser()
    if destination.suffix.lower() not in {".md", ".markdown"}:
        destination = destination / "graph_report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_graph_report_markdown(report), encoding="utf-8")
    return str(destination)


def _normalize_nodes(nodes_data: Iterable[Any]) -> dict[str, dict[str, Any]]:
    nodes = {}
    for item in nodes_data or []:
        node_id, properties = _unpack_node(item)
        if node_id is None:
            continue
        node_id = str(node_id)
        node_properties = dict(properties)
        node_properties["id"] = node_id
        node_properties["name"] = _display_name(node_properties, node_id)
        node_properties["node_sets"] = _extract_values(node_properties, _NODE_SET_KEYS)
        node_properties["source_documents"] = _extract_values(node_properties, _SOURCE_KEYS)
        nodes[node_id] = node_properties
    return nodes


def _normalize_edges(edges_data: Iterable[Any]) -> list[dict[str, Any]]:
    edges = []
    for item in edges_data or []:
        source, target, relation, properties = _unpack_edge(item)
        if source is None or target is None:
            continue
        edge_properties = dict(properties)
        confidence_tag = _infer_confidence_tag(str(relation), edge_properties)
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "relation": str(relation or ""),
                "properties": edge_properties,
                "confidence_tag": confidence_tag,
            }
        )
    return edges


def _unpack_node(item: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(item, Mapping):
        node_id = item.get("id")
        properties = item.get("properties", item)
        return node_id, _as_mapping(properties)
    if hasattr(item, "id") and hasattr(item, "properties"):
        return getattr(item, "id"), _as_mapping(getattr(item, "properties"))
    try:
        node_id, properties = item
        return node_id, _as_mapping(properties)
    except (TypeError, ValueError):
        return None, {}


def _unpack_edge(item: Any) -> tuple[Any, Any, str, Mapping[str, Any]]:
    if isinstance(item, Mapping):
        source = item.get("source") or item.get("source_id")
        target = item.get("target") or item.get("target_id")
        relation = item.get("relation") or item.get("relationship_name") or item.get("label") or ""
        properties = item.get("properties", item)
        return source, target, relation, _as_mapping(properties)
    if all(hasattr(item, attr) for attr in ("source", "target", "relation", "properties")):
        return (
            getattr(item, "source"),
            getattr(item, "target"),
            getattr(item, "relation"),
            _as_mapping(getattr(item, "properties")),
        )
    try:
        source, target, relation, properties = item
        return source, target, relation, _as_mapping(properties)
    except (TypeError, ValueError):
        return None, None, "", {}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _scope_graph(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> tuple[set[str], list[dict[str, Any]]]:
    if not node_name:
        return set(nodes), edges

    requested = {str(name).strip() for name in node_name if str(name).strip()}
    if not requested:
        return set(nodes), edges

    scoped = {
        node_id
        for node_id, properties in nodes.items()
        if _matches_requested_nodes(properties, requested, node_name_filter_operator)
    }
    scoped_edges = [
        edge
        for edge in edges
        if edge["source"] in nodes
        and edge["target"] in nodes
        and (edge["source"] in scoped or edge["target"] in scoped)
    ]
    touched = set(scoped)
    for edge in scoped_edges:
        touched.add(edge["source"])
        touched.add(edge["target"])
    return touched, scoped_edges


def _matches_requested_nodes(
    properties: Mapping[str, Any],
    requested: set[str],
    operator: str,
) -> bool:
    node_tokens = {
        str(value)
        for value in [
            properties.get("id"),
            properties.get("name"),
            *_extract_values(properties, _NODE_SET_KEYS),
        ]
        if value
    }
    if operator.upper() == "AND":
        return requested.issubset(node_tokens)
    return bool(node_tokens.intersection(requested))


def _compute_page_rank(
    node_ids: Iterable[str],
    adjacency: Mapping[str, set[str]],
    *,
    damping: float = 0.85,
    iterations: int = 25,
) -> dict[str, float]:
    node_ids = list(node_ids)
    node_count = len(node_ids)
    if node_count == 0:
        return {}

    initial_rank = 1.0 / node_count
    ranks = {node_id: initial_rank for node_id in node_ids}

    for _ in range(iterations):
        next_ranks = {node_id: (1.0 - damping) / node_count for node_id in node_ids}
        for node_id in node_ids:
            neighbors = adjacency.get(node_id) or set()
            if neighbors:
                share = damping * ranks[node_id] / len(neighbors)
                for neighbor in neighbors:
                    if neighbor in next_ranks:
                        next_ranks[neighbor] += share
            else:
                share = damping * ranks[node_id] / node_count
                for target_id in node_ids:
                    next_ranks[target_id] += share
        ranks = next_ranks

    return ranks


def _rank_hub_nodes(
    nodes: dict[str, dict[str, Any]],
    node_ids: Iterable[str],
    degree: Counter[str],
    page_rank: Mapping[str, float],
    top_n: int,
) -> list[dict[str, Any]]:
    ranked_nodes = []
    total_nodes = max(len(set(node_ids)), 1)
    for node_id in node_ids:
        properties = nodes[node_id]
        node_degree = degree.get(node_id, 0)
        ranked_nodes.append(
            {
                "id": node_id,
                "name": properties.get("name") or node_id,
                "type": properties.get("type") or "Unknown",
                "degree": node_degree,
                "degree_centrality": node_degree / max(total_nodes - 1, 1),
                "pagerank": page_rank.get(node_id, 0.0),
                "node_sets": properties.get("node_sets") or [],
                "source_documents": properties.get("source_documents") or [],
            }
        )

    ranked_nodes.sort(
        key=lambda node: (-node["degree"], -node["pagerank"], str(node["name"]).lower())
    )
    return ranked_nodes[: max(top_n, 0)]


def _rank_surprising_connections(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    relation_counts = Counter(edge["relation"] for edge in edges)
    cooccurrence_counts = Counter()
    edge_groups = []

    for edge in edges:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        groups = _cooccurrence_groups(source, target, edge["relation"])
        edge_groups.append(groups)
        cooccurrence_counts.update(groups)

    ranked = []
    for edge, groups in zip(edges, edge_groups):
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        source_sets = set(source.get("node_sets") or [])
        target_sets = set(target.get("node_sets") or [])
        source_docs = set(source.get("source_documents") or [])
        target_docs = set(target.get("source_documents") or [])
        cross_node_sets = bool(source_sets and target_sets and source_sets.isdisjoint(target_sets))
        cross_sources = bool(source_docs and target_docs and source_docs.isdisjoint(target_docs))

        if not cross_node_sets and not cross_sources:
            continue

        cooccurrence_count = min(cooccurrence_counts[group] for group in groups)
        novelty_score = 1.0
        if cross_node_sets:
            novelty_score += 2.0
        if cross_sources:
            novelty_score += 1.5
        novelty_score += 1.0 / max(relation_counts[edge["relation"]], 1)
        novelty_score += 1.0 / max(cooccurrence_count, 1)
        if edge["confidence_tag"] == "INFERRED":
            novelty_score += 0.25

        reasons = []
        if cross_node_sets:
            reasons.append("connects different node sets")
        if cross_sources:
            reasons.append("connects different source documents")
        if cooccurrence_count == 1:
            reasons.append("rare co-occurrence")

        ranked.append(
            {
                "source": _node_ref(source),
                "target": _node_ref(target),
                "relation": edge["relation"],
                "novelty_score": round(novelty_score, 4),
                "cooccurrence_count": cooccurrence_count,
                "confidence_tag": edge["confidence_tag"],
                "source_node_sets": sorted(source_sets),
                "target_node_sets": sorted(target_sets),
                "source_documents": sorted(source_docs),
                "target_documents": sorted(target_docs),
                "reasons": reasons,
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["novelty_score"],
            item["relation"],
            item["source"]["name"].lower(),
            item["target"]["name"].lower(),
        )
    )
    return ranked[: max(top_n, 0)]


def _cooccurrence_groups(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    relation: str,
) -> list[tuple[str, str, str]]:
    groups = []
    source_sets = source.get("node_sets") or []
    target_sets = target.get("node_sets") or []
    source_docs = source.get("source_documents") or []
    target_docs = target.get("source_documents") or []

    for source_set, target_set in product(source_sets, target_sets):
        if source_set != target_set:
            groups.append(("node_set", *sorted((source_set, target_set))))

    for source_doc, target_doc in product(source_docs, target_docs):
        if source_doc != target_doc:
            groups.append(("source", *sorted((source_doc, target_doc))))

    return groups or [("relation", relation, relation)]


def _summarize_confidence_tags(edges: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(edge["confidence_tag"] for edge in edges)
    return {tag: counts.get(tag, 0) for tag in ("EXTRACTED", "INFERRED", "UNKNOWN")}


def _infer_confidence_tag(relation: str, properties: Mapping[str, Any]) -> str:
    for key in _CONFIDENCE_KEYS:
        value = properties.get(key)
        if value is None:
            continue
        text = str(value).upper()
        if "INFER" in text:
            return "INFERRED"
        if "EXTRACT" in text:
            return "EXTRACTED"
        if "UNKNOWN" in text:
            return "UNKNOWN"

    if properties.get("ontology_valid") is False:
        return "INFERRED"
    if properties.get("ontology_valid") is True:
        return "EXTRACTED"
    if relation in {"associated_with", "semantically_related_to"}:
        return "INFERRED"

    return "EXTRACTED"


def _extract_values(properties: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    values = []
    for key in keys:
        values.extend(_flatten_value(properties.get(key)))
    return sorted(set(value for value in values if value))


def _flatten_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        separators = [",", ";"] if "," in value or ";" in value else []
        parts = [value]
        for separator in separators:
            parts = [subpart for part in parts for subpart in part.split(separator)]
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, Mapping):
        nested = value.get("name") or value.get("id") or value.get("title")
        return _flatten_value(nested)
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_value(item))
        return flattened
    name = getattr(value, "name", None)
    if name:
        return _flatten_value(name)
    return [str(value)]


def _display_name(properties: Mapping[str, Any], node_id: str) -> str:
    for key in ("name", "title", "text", "summary", "description", "content"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip() and not _looks_like_identifier(value):
            return " ".join(value.split())[:120]
    node_type = properties.get("type") or "node"
    return f"Unnamed {node_type} ({node_id[:8]})"


def _looks_like_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_LIKE_RE.match(value.strip()))


def _node_ref(properties: Mapping[str, Any]) -> dict[str, str]:
    return {
        "id": str(properties.get("id")),
        "name": str(properties.get("name") or properties.get("id")),
        "type": str(properties.get("type") or "Unknown"),
    }


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")
