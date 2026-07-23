from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.shared.logging_utils import get_logger

logger = get_logger("GraphReportRetriever")

# Organizational container nodes: excluded from the analysis graph. Their
# ``belongs_to_set`` edges are the source of node-set membership, not content.
_CONTAINER_TYPE = "NodeSet"
# The membership edge that ties any node to its NodeSet container.
_MEMBERSHIP_RELATIONSHIP = "belongs_to_set"
# Node types that make meaningful "god nodes" (the knowledge layer). Structural
# nodes (chunks, documents, summaries) are kept in the topology but not surfaced.
_HUB_TYPES = frozenset({"Entity", "EntityType"})


class _SuggestedQuestions(BaseModel):
    questions: List[str]


def _display_name(node_id: str, props: Dict[str, Any]) -> str:
    """Human-readable label for a node, falling back to ``type (short-id)``."""
    name = props.get("name") or props.get("label")
    if isinstance(name, str) and name.strip():
        return name
    node_type = props.get("type") or "Node"
    return f"{node_type} ({str(node_id)[:8]})"


def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a dict of floats to [0, 1]."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    if span == 0:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def _pagerank(graph: nx.DiGraph) -> Dict[str, float]:
    """PageRank via networkx (sparse; scipy-backed). Empty on failure so the
    caller can fall back to degree — never build a dense N×N matrix."""
    if graph.number_of_edges() == 0:
        return {}
    try:
        return nx.pagerank(graph)
    except Exception as exc:  # scipy missing or numerical issue
        logger.warning("PageRank unavailable, ranking hubs by degree only: %s", exc)
        return {}


class GraphReportRetriever(BaseRetriever):
    """Generate a Graph Insight Report from the current graph context.

    Computes, from ``get_graph_data()`` alone (no schema or storage changes):

    1. Hub nodes — top-N knowledge nodes by degree + PageRank (zero LLM cost).
    2. Surprising connections — content edges whose endpoints belong to
       different ``node_set``s (cross-source links).
    3. Edge provenance — EXTRACTED (entity-to-entity relationships the LLM
       derived) vs DERIVED (structural chunk/document scaffolding).
    4. Suggested questions — one cheap LLM call over the report context.
    """

    # The report is computed over the whole graph and ignores the query, so the
    # conversational session-turn analysis (which may call an LLM before
    # retrieval) is pointless here — opt out, like other deterministic retrievers.
    supports_session_turn_preparation = False

    def __init__(self, top_n: int = 10) -> None:
        """
        Args:
            top_n: Hub nodes and surprising connections to surface.
        """
        self.top_n = top_n

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    async def get_retrieved_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
    ) -> Tuple[list, list]:
        """Fetch all nodes and edges from the graph engine."""
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        logger.info("GraphReportRetriever fetched %d nodes and %d edges", len(nodes), len(edges))
        return (nodes, edges)

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> str:
        """Format the three deterministic report sections."""
        nodes, edges = retrieved_objects or ([], [])

        if not nodes:
            return "# Graph Insight Report\n\n_Graph is empty — run `cognify` first._\n"

        props_by_id = {node_id: props for node_id, props in nodes}
        node_type = {node_id: props.get("type") for node_id, props in nodes}
        node_sets = _resolve_node_sets(nodes, edges, node_type)

        # Analysis graph: content nodes only (drop NodeSet containers) and
        # content edges only (drop membership scaffolding).
        graph = nx.DiGraph()
        for node_id, node_t in node_type.items():
            if node_t != _CONTAINER_TYPE:
                graph.add_node(node_id)
        for src, tgt, rel, _ in edges:
            if rel == _MEMBERSHIP_RELATIONSHIP:
                continue
            if src in graph and tgt in graph:
                graph.add_edge(src, tgt, relationship=rel)

        degree = dict(graph.degree())
        pagerank = _pagerank(graph)

        hubs = _rank_hubs(graph, node_type, degree, pagerank, self.top_n)
        surprising = _cross_set_connections(graph, node_type, node_sets, pagerank, self.top_n)
        provenance = _edge_provenance(graph, node_type)

        def name_of(node_id: str) -> str:
            return _display_name(node_id, props_by_id.get(node_id, {}))

        def sets_of(node_id: str) -> str:
            return ", ".join(sorted(node_sets.get(node_id, []))) or "—"

        lines = [
            "# Graph Insight Report\n",
            f"**Nodes:** {graph.number_of_nodes()} | **Edges:** {graph.number_of_edges()}\n",
            "\n## 🏆 Hub Nodes\n",
            "Top nodes by degree + PageRank (most connected / influential):\n",
        ]
        for rank, node_id in enumerate(hubs, 1):
            lines.append(
                f"{rank}. **{name_of(node_id)}** (set: `{sets_of(node_id)}`, "
                f"degree: {degree.get(node_id, 0)}, PageRank: {pagerank.get(node_id, 0.0):.4f})"
            )

        lines.append("\n\n## 🔗 Surprising Cross-Set Connections\n")
        if surprising:
            lines.append("Entity pairs linked across different node sets:\n")
            for src, tgt, rel in surprising:
                lines.append(
                    f"- **{name_of(src)}** (`{sets_of(src)}`) —[{rel}]→ "
                    f"**{name_of(tgt)}** (`{sets_of(tgt)}`)"
                )
        else:
            lines.append(
                "_No cross-node_set connections found (add data under multiple node_sets)._"
            )

        lines.append("\n\n## 🏷️ Edge Provenance\n")
        lines.append("How edges entered the graph:\n")
        total = sum(provenance.values()) or 1
        provenance_labels = {
            "EXTRACTED": "EXTRACTED (entity-to-entity, LLM-derived)",
            "DERIVED": "DERIVED (structural / chunk-linked)",
        }
        for tag, count in provenance.items():
            lines.append(
                f"- `{provenance_labels[tag]}`: {count} edges ({100.0 * count / total:.1f}%)"
            )

        lines.append("")
        return "\n".join(lines)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> List[str]:
        """Append LLM-suggested questions then return the full report."""
        context = context or ""
        questions_md = await self._suggest_questions(context)
        return [context + "\n" + questions_md]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _suggest_questions(self, context: str) -> str:
        """One cheap LLM call → 4-5 questions for ``cognee.search()``."""
        try:
            resp = await LLMGateway.acreate_structured_output(
                text_input=(
                    f"Knowledge graph report:\n{context[:2000]}\n\n"
                    "Based on the hub nodes and cross-set connections, suggest 4–5 "
                    "specific, concise questions a user could ask to explore this graph."
                ),
                system_prompt=(
                    "You are a knowledge graph analyst. Suggest short, specific questions "
                    "answerable by querying the described knowledge graph. "
                    "Return only the questions list."
                ),
                response_model=_SuggestedQuestions,
            )
            questions = resp.questions
        except Exception as exc:
            logger.warning("GraphReportRetriever: suggested-questions call failed: %s", exc)
            questions = ["What are the main topics in this knowledge graph?"]

        lines = [
            "\n## 💡 Suggested Questions\n",
            "Questions ready to pipe into `cognee.search()`:\n",
            *(f"- {q}" for q in questions),
        ]
        return "\n".join(lines)


def _resolve_node_sets(nodes: list, edges: list, node_type: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Map each node id to the node_set names it belongs to.

    Uses the real cognee mechanism: ``belongs_to_set`` edges pointing at
    ``NodeSet`` container nodes (this is how entities carry membership), plus
    the ``source_node_set`` string that chunks/documents also expose.
    """
    nodeset_name = {
        node_id: props.get("name")
        for node_id, props in nodes
        if node_type.get(node_id) == _CONTAINER_TYPE and props.get("name")
    }
    node_sets: Dict[str, Set[str]] = {}
    for src, tgt, rel, _ in edges:
        if rel == _MEMBERSHIP_RELATIONSHIP and tgt in nodeset_name:
            node_sets.setdefault(src, set()).add(nodeset_name[tgt])
    for node_id, props in nodes:
        source_node_set = props.get("source_node_set")
        if isinstance(source_node_set, str) and source_node_set:
            names = {name.strip() for name in source_node_set.split(",") if name.strip()}
            if names:
                node_sets.setdefault(node_id, set()).update(names)
    return node_sets


def _rank_hubs(
    graph: nx.DiGraph,
    node_type: Dict[str, Any],
    degree: Dict[str, int],
    pagerank: Dict[str, float],
    top_n: int,
) -> List[str]:
    """Top-N hub nodes by combined (normalised) degree + PageRank.

    Prefers the knowledge layer (entities/entity types); falls back to all
    content nodes when no such nodes exist.
    """
    norm_deg = _normalize({n: float(d) for n, d in degree.items()})
    norm_pr = _normalize(pagerank)
    score = {n: norm_deg.get(n, 0.0) + norm_pr.get(n, 0.0) for n in graph.nodes()}
    ranked = sorted(score, key=score.__getitem__, reverse=True)
    hubs = [n for n in ranked if node_type.get(n) in _HUB_TYPES]
    return (hubs or ranked)[:top_n]


def _cross_set_connections(
    graph: nx.DiGraph,
    node_type: Dict[str, Any],
    node_sets: Dict[str, Set[str]],
    pagerank: Dict[str, float],
    top_n: int,
) -> List[Tuple[str, str, str]]:
    """Entity pairs whose endpoints have differing node_set membership.

    Restricted to entity-to-entity edges (the knowledge layer) so structural
    scaffolding never shows up as a "surprising" link. Ranked by novelty: fully
    disjoint sets (true cross-source links) first, then combined endpoint PageRank.
    """
    surprising = []
    for src, tgt, edata in graph.edges(data=True):
        if node_type.get(src) != "Entity" or node_type.get(tgt) != "Entity":
            continue
        src_sets, tgt_sets = node_sets.get(src), node_sets.get(tgt)
        if src_sets and tgt_sets and src_sets != tgt_sets:
            disjoint = not (src_sets & tgt_sets)
            novelty = pagerank.get(src, 0.0) + pagerank.get(tgt, 0.0)
            surprising.append(
                (disjoint, novelty, src, tgt, edata.get("relationship", "relates_to"))
            )
    surprising.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(src, tgt, rel) for _, _, src, tgt, rel in surprising[:top_n]]


def _edge_provenance(graph: nx.DiGraph, node_type: Dict[str, Any]) -> Dict[str, int]:
    """Classify content edges by provenance.

    EXTRACTED = entity-to-entity relationships the LLM extracted; DERIVED =
    structural scaffolding (chunk→entity, entity→type, chunk→document, …).
    Cognee stores no explicit confidence tag, so this is derived from the
    endpoint node types.
    """
    counts = {"EXTRACTED": 0, "DERIVED": 0}
    for src, tgt in graph.edges():
        if node_type.get(src) == "Entity" and node_type.get(tgt) == "Entity":
            counts["EXTRACTED"] += 1
        else:
            counts["DERIVED"] += 1
    return counts
