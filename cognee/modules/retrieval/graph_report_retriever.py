from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.shared.logging_utils import get_logger

logger = get_logger("GraphReportRetriever")

# Property keys checked for edge confidence/extraction type (first match wins)
_CONFIDENCE_KEYS = ("confidence", "extraction_type", "relationship_type", "source_type")
# Property keys checked for node set/group membership (first match wins)
_NODE_SET_KEYS = ("node_set_name", "node_set", "belongs_to_set", "type_", "data_class", "category")


class _SuggestedQuestions(BaseModel):
    questions: List[str]


def _get_node_set(props: Dict[str, Any]) -> Optional[str]:
    """Return the node-set label from node properties, or None if absent."""
    for key in _NODE_SET_KEYS:
        val = props.get(key)
        if val and isinstance(val, str):
            return val
    return None


def _get_edge_confidence(props: Dict[str, Any]) -> str:
    """Return the confidence/extraction tag from edge properties."""
    for key in _CONFIDENCE_KEYS:
        val = props.get(key)
        if val and isinstance(val, str):
            return val.upper()
    return "UNKNOWN"


def _pagerank(G: nx.DiGraph, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
    """Power-iteration PageRank using numpy only (no scipy required)."""
    import numpy as np

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    idx = {node: i for i, node in enumerate(nodes)}
    M = np.zeros((n, n), dtype=float)
    for src, tgt in G.edges():
        out_deg = G.out_degree(src)
        if out_deg > 0:
            M[idx[tgt], idx[src]] += 1.0 / out_deg

    # Distribute dangling-node probability evenly
    dangling = np.where(M.sum(axis=0) == 0)[0]
    M[:, dangling] = 1.0 / n

    r = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        r_new = alpha * (M @ r) + (1.0 - alpha) / n
        if float(np.abs(r_new - r).sum()) < 1.0e-10:
            break
        r = r_new

    return {nodes[i]: float(r[i]) for i in range(n)}


def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a dict of floats to [0, 1]."""
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    span = max_v - min_v
    if span == 0:
        return {k: 1.0 for k in values}
    return {k: (v - min_v) / span for k, v in values.items()}


def _safe_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only primitive-valued entries so networkx is happy."""
    return {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))}


class GraphReportRetriever(BaseRetriever):
    """Generate a Graph Insight Report from the current graph context.

    Computes:
    1. Hub nodes — top-N by combined degree + PageRank (zero LLM cost).
    2. Surprising connections — entity pairs from different node_sets.
    3. Confidence tags — EXTRACTED / INFERRED distribution over edges.
    4. Suggested questions — one cheap LLM call over hub-node context.

    No schema or storage changes; reads solely via ``get_graph_data()``.
    """

    def __init__(
        self,
        top_n: int = 10,
        node_name: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            top_n: Hub nodes and surprising connections to surface.
            node_name: Optional name filter (informational; scoping is
                handled upstream by the dataset context).
        """
        self.top_n = top_n
        self.node_name = node_name

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
        logger.info(
            "GraphReportRetriever fetched %d nodes and %d edges", len(nodes), len(edges)
        )
        return (nodes, edges)

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> str:
        """Build networkx DiGraph and format the three deterministic report sections."""
        nodes, edges = retrieved_objects or ([], [])

        if not nodes:
            return "# Graph Insight Report\n\n_Graph is empty — run `cognify` first._\n"

        G = nx.DiGraph()
        for node_id, props in nodes:
            G.add_node(node_id, **_safe_props(props))
        for src, tgt, rel_type, props in edges:
            G.add_edge(src, tgt, relationship=rel_type, **_safe_props(props))

        # --- Section 1: Hub nodes ---
        degree: Dict[str, int] = dict(G.degree())
        pagerank: Dict[str, float] = _pagerank(G)

        norm_deg = _normalize({n: float(d) for n, d in degree.items()})
        norm_pr = _normalize(pagerank)
        hub_score = {n: norm_deg.get(n, 0.0) + norm_pr.get(n, 0.0) for n in G.nodes()}
        top_hubs = sorted(hub_score, key=hub_score.__getitem__, reverse=True)[: self.top_n]

        # --- Section 2: Surprising cross-node_set connections ---
        surprising: List[Tuple[str, str, str, str, str]] = []
        for src, tgt, edata in G.edges(data=True):
            src_set = _get_node_set(dict(G.nodes[src]))
            tgt_set = _get_node_set(dict(G.nodes[tgt]))
            if src_set and tgt_set and src_set != tgt_set:
                rel = edata.get("relationship", "relates_to")
                surprising.append((src, tgt, rel, src_set, tgt_set))
        # Rank by combined PageRank of endpoints
        surprising.sort(
            key=lambda x: pagerank.get(x[0], 0.0) + pagerank.get(x[1], 0.0),
            reverse=True,
        )
        surprising = surprising[: self.top_n]

        # --- Section 3: Confidence tags ---
        confidence_counts: Dict[str, int] = {}
        for _, _, edata in G.edges(data=True):
            tag = _get_edge_confidence(dict(edata))
            confidence_counts[tag] = confidence_counts.get(tag, 0) + 1

        # --- Format Markdown ---
        lines = [
            "# Graph Insight Report\n",
            f"**Nodes:** {G.number_of_nodes()} | **Edges:** {G.number_of_edges()}\n",
        ]

        lines.append("\n## 🏆 Hub Nodes\n")
        lines.append("Top nodes by degree + PageRank (most connected / influential):\n")
        for rank, nid in enumerate(top_hubs, 1):
            props = dict(G.nodes[nid])
            name = props.get("name") or props.get("label") or nid
            node_set = _get_node_set(props) or "—"
            deg = degree.get(nid, 0)
            pr = pagerank.get(nid, 0.0)
            lines.append(
                f"{rank}. **{name}** (set: `{node_set}`, degree: {deg}, PageRank: {pr:.4f})"
            )

        lines.append("\n\n## 🔗 Surprising Cross-Set Connections\n")
        if surprising:
            lines.append("Entity pairs linked across different node sets:\n")
            for src, tgt, rel, src_set, tgt_set in surprising:
                src_name = dict(G.nodes[src]).get("name") or src
                tgt_name = dict(G.nodes[tgt]).get("name") or tgt
                lines.append(
                    f"- **{src_name}** (`{src_set}`) —[{rel}]→ **{tgt_name}** (`{tgt_set}`)"
                )
        else:
            lines.append(
                "_No cross-node_set connections found. All nodes share the same source._"
            )

        lines.append("\n\n## 🏷️ Confidence Tags\n")
        lines.append("Edge extraction confidence distribution:\n")
        total = sum(confidence_counts.values()) or 1
        for tag, count in sorted(confidence_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / total
            lines.append(f"- `{tag}`: {count} edges ({pct:.1f}%)")

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
        questions_md = await self._suggest_questions(context, query or "")
        return [context + "\n" + questions_md]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _suggest_questions(self, context: str, topic: str) -> str:
        """One cheap LLM call → 4-5 questions for ``cognee.search()``."""
        try:
            prompt = (
                f"Knowledge graph report:\n{context[:2000]}\n\n"
                f"Focus topic (if any): {topic or 'general'}\n\n"
                "Based on the hub nodes and cross-set connections, suggest 4–5 specific, "
                "concise questions a user could ask to explore this graph via semantic search."
            )
            resp = await LLMGateway.acreate_structured_output(
                text_input=prompt,
                system_prompt=(
                    "You are a knowledge graph analyst. Suggest short, specific questions "
                    "that can be answered by querying the described knowledge graph. "
                    "Return only the questions list."
                ),
                response_model=_SuggestedQuestions,
            )
            questions = resp.questions
        except Exception as exc:
            logger.warning("GraphReportRetriever: LLM suggested-questions call failed: %s", exc)
            questions = ["What are the main topics in this knowledge graph?"]

        lines = ["\n## 💡 Suggested Questions\n"]
        lines.append("Questions ready to pipe into `cognee.search()`:\n")
        for q in questions:
            lines.append(f"- {q}")
        return "\n".join(lines)
