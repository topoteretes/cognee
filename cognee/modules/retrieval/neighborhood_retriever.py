from collections import deque
from typing import Any, Dict, List, Optional, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch

logger = get_logger("NeighborhoodRetriever")


class NeighborhoodRetriever(BaseRetriever):
    """
    Structural, LLM-free retriever that returns the N-hop neighborhood subgraph
    around the entities a query mentions.

    The pipeline is:
    1. get_retrieved_objects: vector-resolve the query to seed node IDs, slice to
       ``seed_top_k``, and call the graph adapter's ``get_neighborhood`` primitive
       directly to fetch the raw subgraph (the primitive already unions and
       deduplicates the per-seed ego-graphs).
    2. get_context_from_objects: apply a deterministic ``max_nodes`` truncation
       (closest hops first) and serialize the subgraph to plain dicts.
    3. get_completion_from_context: return the structured subgraph as-is.

    No LLM is ever called, and the vector engine is touched exactly once (for seed
    resolution). Traversal is undirected and edge-type filtering is symmetric,
    matching the underlying primitive; directional traversal is intentionally out
    of scope.
    """

    def __init__(
        self,
        depth: int = 2,
        edge_types: Optional[List[str]] = None,
        seed_top_k: int = 5,
        max_nodes: int = 100,
        seed_collections: Optional[List[str]] = None,
    ):
        """
        Initialize the neighborhood retriever.

        Parameters:
        -----------

            - depth (int): Number of hops to traverse from each seed node. Defaults to 2.
            - edge_types (Optional[List[str]]): If provided, only traverse edges whose
              relationship type is in this allow-list. Applied symmetrically (undirected).
              Defaults to None (all edge types).
            - seed_top_k (int): Maximum number of seed nodes to resolve from the query.
              Defaults to 5.
            - max_nodes (int): Maximum number of nodes to keep after truncation. Seeds are
              always retained even if they exceed this cap. Defaults to 100.
            - seed_collections (Optional[List[str]]): Vector collections to resolve seed
              nodes from. Defaults to ["Entity_name", "EntityType_name"].
        """
        self.depth = depth
        self.edge_types = edge_types
        self.seed_top_k = seed_top_k
        self.max_nodes = max_nodes
        self.seed_collections = seed_collections or ["Entity_name", "EntityType_name"]

    @staticmethod
    def _ordered_seed_ids(vector_search: NodeEdgeVectorSearch) -> List[str]:
        """
        Build a deterministic, score-ordered list of candidate seed node IDs.

        ``node_distances`` maps each collection to its scored results. A node may
        appear in more than one collection, so we keep each node's lowest (best)
        score and sort by ``(score asc, node_id asc)``. The id tie-break makes the
        ordering deterministic even when scores are equal.

        Parameters:
        -----------

            - vector_search (NodeEdgeVectorSearch): A search whose distances have
              already been retrieved via ``embed_and_retrieve_distances``.

        Returns:
        --------

            - List[str]: Candidate seed node IDs, best match first.
        """
        best_score_by_id: Dict[str, float] = {}
        for scored_results in vector_search.node_distances.values():
            for scored_node in scored_results:
                node_id = getattr(scored_node, "id", None)
                if node_id is None:
                    continue
                node_id = str(node_id)
                score = getattr(scored_node, "score", float("inf"))
                if node_id not in best_score_by_id or score < best_score_by_id[node_id]:
                    best_score_by_id[node_id] = score

        return [
            node_id
            for node_id, _ in sorted(best_score_by_id.items(), key=lambda item: (item[1], item[0]))
        ]

    async def get_retrieved_objects(self, query: str) -> Dict[str, Any]:
        """
        Resolve the query to seed nodes and fetch their N-hop neighborhood.

        Returns the raw (pre-truncation) subgraph as returned by the graph
        adapter's ``get_neighborhood`` primitive. Returns an empty subgraph when
        the graph has no data or no seeds could be resolved.

        Parameters:
        -----------

            - query (str): The search query whose entities seed the neighborhood.

        Returns:
        --------

            - Dict[str, Any]: ``{"seeds": [...], "nodes": [...], "edges": [...]}``,
              where nodes/edges are the raw tuples from the primitive.
        """
        empty: Dict[str, Any] = {"seeds": [], "nodes": [], "edges": []}

        self._unified_engine = await get_unified_engine()
        is_empty = await self._unified_engine.graph.is_empty()

        if is_empty:
            logger.warning("Neighborhood search attempt on an empty knowledge graph")
            return empty

        vector_search = NodeEdgeVectorSearch(vector_engine=self._unified_engine.vector)
        await vector_search.embed_and_retrieve_distances(
            query=query,
            collections=self.seed_collections,
            wide_search_limit=self.seed_top_k,
        )

        if not vector_search.has_results():
            logger.warning("No seed nodes resolved from the query")
            return empty

        seeds = self._ordered_seed_ids(vector_search)[: self.seed_top_k]
        if not seeds:
            logger.warning("No seed nodes resolved from the query")
            return empty

        nodes, edges = await self._unified_engine.graph.get_neighborhood(
            node_ids=seeds,
            depth=self.depth,
            edge_types=self.edge_types,
        )

        return {"seeds": seeds, "nodes": nodes or [], "edges": edges or []}

    def _truncate(
        self,
        seeds: List[str],
        nodes: List[Any],
        edges: List[Any],
    ) -> Dict[str, Any]:
        """
        Truncate the subgraph to ``max_nodes`` using closest-hops-first ordering.

        A multi-source BFS (hop 0 = seeds) over an undirected adjacency built from
        ``edges`` assigns each node its shortest hop distance from any seed. Nodes
        are sorted by ``(hop asc, node_id asc)`` and the first ``max_nodes`` are
        kept; seeds present in the subgraph are always retained even if they would
        otherwise exceed the cap.

        Returns the kept node IDs (in deterministic order), the kept edge tuples,
        and whether any node was dropped.
        """
        node_map: Dict[str, Any] = {str(node_id): properties for node_id, properties in nodes}
        all_ids = list(node_map.keys())

        # Undirected adjacency, matching the primitive's undirected traversal.
        adjacency: Dict[str, set] = {node_id: set() for node_id in all_ids}
        for source_id, target_id, *_ in edges:
            source_id = str(source_id)
            target_id = str(target_id)
            if source_id in adjacency and target_id in adjacency:
                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

        seeds_present = [seed for seed in seeds if seed in node_map]

        # Multi-source BFS for shortest hop distance from any seed.
        hop_by_id: Dict[str, int] = {seed: 0 for seed in seeds_present}
        queue = deque(seeds_present)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in hop_by_id:
                    hop_by_id[neighbor] = hop_by_id[current] + 1
                    queue.append(neighbor)

        ordered_ids = sorted(
            all_ids, key=lambda node_id: (hop_by_id.get(node_id, float("inf")), node_id)
        )

        kept = set(ordered_ids[: self.max_nodes])
        kept.update(seeds_present)  # never drop seeds
        truncated = len(kept) < len(all_ids)

        kept_node_ids = [node_id for node_id in ordered_ids if node_id in kept]
        kept_edges = [edge for edge in edges if str(edge[0]) in kept and str(edge[1]) in kept]
        kept_edges = sorted(kept_edges, key=lambda edge: (str(edge[0]), str(edge[1]), str(edge[2])))

        return {
            "node_map": node_map,
            "node_ids": kept_node_ids,
            "edges": kept_edges,
            "truncated": truncated,
        }

    async def get_context_from_objects(
        self, query: str, retrieved_objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Truncate and serialize the raw subgraph into a structured dict.

        Parameters:
        -----------

            - query (str): The original search query (unused; kept for interface parity).
            - retrieved_objects (Dict[str, Any]): Output of ``get_retrieved_objects``.

        Returns:
        --------

            - Dict[str, Any]: ``{"seeds", "nodes", "edges", "truncated", "depth"}`` where
              nodes are ``{id, name, type, properties}`` and edges are
              ``{source, target, relationship_name, properties}``.
        """
        retrieved_objects = retrieved_objects or {}
        seeds: List[str] = retrieved_objects.get("seeds", [])
        nodes: List[Any] = retrieved_objects.get("nodes", [])
        edges: List[Any] = retrieved_objects.get("edges", [])

        if not nodes:
            return {
                "seeds": seeds,
                "nodes": [],
                "edges": [],
                "truncated": False,
                "depth": self.depth,
            }

        truncation = self._truncate(seeds, nodes, edges)
        node_map = truncation["node_map"]

        serialized_nodes = []
        for node_id in truncation["node_ids"]:
            properties = dict(node_map[node_id])
            properties.pop("id", None)
            name = properties.pop("name", None)
            node_type = properties.pop("type", None)
            serialized_nodes.append(
                {
                    "id": node_id,
                    "name": name,
                    "type": node_type,
                    "properties": properties,
                }
            )

        serialized_edges = [
            {
                "source": str(source_id),
                "target": str(target_id),
                "relationship_name": relationship_name,
                "properties": properties,
            }
            for source_id, target_id, relationship_name, properties in truncation["edges"]
        ]

        return {
            "seeds": seeds,
            "nodes": serialized_nodes,
            "edges": serialized_edges,
            "truncated": truncation["truncated"],
            "depth": self.depth,
        }

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Return the structured neighborhood subgraph directly.

        No LLM is involved; this mirrors the non-generating retrievers
        (ChunksRetriever / SummariesRetriever) by returning the context payload.

        Parameters:
        -----------

            - query (str): The original search query (unused; kept for interface parity).
            - retrieved_objects (Any): Output of ``get_retrieved_objects`` (unused here).
            - context (Any): The structured subgraph from ``get_context_from_objects``.

        Returns:
        --------

            - List[dict]: A single-element list wrapping the structured subgraph.
        """
        return [context]
