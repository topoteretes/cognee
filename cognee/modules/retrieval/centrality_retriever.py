from typing import Any, List, Optional, Union

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    get_memory_fragment,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("CentralityRetriever")

EXCLUDED_NODE_TYPES = {"EntityType", "NodeSet"}


def _compute_degree_centrality(graph: CogneeGraph, node_ids: List[str]) -> dict:
    scores = {}
    for node_id in node_ids:
        node = graph.get_node(node_id)
        if node is None:
            scores[node_id] = 0.0
            continue
        scores[node_id] = float(len(node.skeleton_edges))
    return scores


def _compute_pagerank(
    graph: CogneeGraph,
    node_ids: List[str],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    node_set = set(node_ids)
    n = len(node_set)
    if n == 0:
        return {}

    id_list = list(node_set)
    scores = {nid: 1.0 / n for nid in id_list}

    out_degree = {}
    for nid in id_list:
        node = graph.get_node(nid)
        if node is None:
            out_degree[nid] = 0
            continue
        out_degree[nid] = sum(1 for edge in node.skeleton_edges if edge.node1.id == nid)

    for _ in range(max_iter):
        max_diff = 0.0
        new_scores = {}
        dangling_sum = sum(scores[nid] for nid in id_list if out_degree.get(nid, 0) == 0)
        for nid in id_list:
            incoming = 0.0
            node = graph.get_node(nid)
            if node is not None:
                for edge in graph.edges:
                    if edge.node2.id == nid and edge.node1.id in node_set:
                        deg = out_degree.get(edge.node1.id, 0)
                        if deg > 0:
                            incoming += scores[edge.node1.id] / deg
            new_scores[nid] = (1.0 - damping) / n + damping * (incoming + dangling_sum / n)
            max_diff = max(max_diff, abs(new_scores[nid] - scores[nid]))
        scores = new_scores
        if max_diff < tol:
            break

    return scores


class CentralityRetriever(BaseRetriever):
    """
    Ranks entities by structural importance (centrality) within the graph.

    Two modes (configured via retriever_specific_config["mode"]):
        - "degree" (default): counts incident edges per entity node.
        - "pagerank": power iteration over the directed graph adjacency.

    EntityType label nodes are automatically excluded since they artificially
    dominate raw centrality due to every entity having an ``is_a`` edge to its type.
    """

    def __init__(
        self,
        top_k: int = 15,
        mode: str = "degree",
        max_nodes: int = 500,
        neighborhood_depth: Optional[int] = None,
        neighborhood_seed_top_k: int = 10,
    ):
        self.top_k = top_k
        self.mode = mode
        self.max_nodes = max_nodes
        self.neighborhood_depth = neighborhood_depth
        self.neighborhood_seed_top_k = neighborhood_seed_top_k

    def _is_entity_node(self, node: Node) -> bool:
        node_type = node.attributes.get("type", "")
        return node_type not in EXCLUDED_NODE_TYPES and bool(node_type)

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[dict]:
        if query_batch:
            logger.warning("Batch query not supported for CENTRALITY, using first query.")
            query = query_batch[0] if query_batch else query

        memory_fragment = await get_memory_fragment(
            neighborhood_depth=self.neighborhood_depth if query else None,
            neighborhood_seed_top_k=self.neighborhood_seed_top_k,
        )

        if not memory_fragment or not memory_fragment.nodes:
            logger.warning("Empty graph projected for centrality retrieval.")
            return []

        entity_node_ids = [
            nid for nid, node in memory_fragment.nodes.items() if self._is_entity_node(node)
        ]

        if not entity_node_ids:
            logger.warning("No entity nodes found in the projected graph.")
            return []

        if self.mode == "pagerank":
            scores = _compute_pagerank(memory_fragment, entity_node_ids)
        else:
            scores = _compute_degree_centrality(memory_fragment, entity_node_ids)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        top_results = []
        for node_id, score in ranked[: self.top_k]:
            node = memory_fragment.get_node(node_id)
            top_results.append(
                {
                    "node_id": node_id,
                    "node_name": node.attributes.get("name", node_id) if node else node_id,
                    "node_type": node.attributes.get("type", "") if node else "",
                    "score": score,
                    "description": node.attributes.get("description", "") if node else "",
                }
            )

        logger.info(
            "Centrality retrieval: mode=%s, entities=%d, returned=%d",
            self.mode,
            len(entity_node_ids),
            len(top_results),
        )

        return top_results

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
    ) -> Union[str, List[str]]:
        if not retrieved_objects:
            return ""

        lines = []
        for i, item in enumerate(retrieved_objects, start=1):
            name = item.get("node_name", "unknown")
            score = item.get("score", 0.0)
            desc = item.get("description", "")
            if desc:
                lines.append(f"{i}. {name} (score: {score:.4f}) - {desc}")
            else:
                lines.append(f"{i}. {name} (score: {score:.4f})")

        return "\n".join(lines)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> Union[List[str], List[dict]]:
        if not context:
            return []

        header = f"Top entities by {self.mode} centrality"
        if query:
            header += f" for query: {query}"

        return [f"{header}\n\n{context}"]
