import random
from typing import Any, List, Optional, Tuple

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.shared.logging_utils import get_logger

from .path import Path

logger = get_logger("AdaptivePathSearch")

DEFAULT_COLLECTIONS = [
    "Entity_name",
    "TextSummary_text",
    "EntityType_name",
    "DocumentChunk_text",
    "EdgeType_relationship_name",
]


class AdaptivePathSearch:
    """
    Standalone path-search core: samples reproducible walks over the in-memory graph,
    scores them against the query, and returns the best connected paths.

    Seeds are the vector-closest nodes, selected deterministically. From each seed,
    ``walks_per_seed`` random walks are sampled with a seeded RNG (uniform choice among
    unvisited neighbors, sorted by ``Edge.stable_id()`` before sampling), so a fixed
    ``random_seed`` reproduces the same paths. All scored, deduplicated candidates are
    retained in ``scored_candidates``; ``run()`` returns the ``top_k`` best.
    """

    def __init__(
        self,
        num_seeds: int = 5,
        walks_per_seed: int = 10,
        max_depth: int = 4,
        top_k: int = 5,
        random_seed: int = 0,
        memory_fragment: Optional[CogneeGraph] = None,
        vector_search: Optional[NodeEdgeVectorSearch] = None,
        collections: Optional[List[str]] = None,
        wide_search_top_k: int = 100,
    ):
        for name, value in (
            ("num_seeds", num_seeds),
            ("walks_per_seed", walks_per_seed),
            ("max_depth", max_depth),
            ("top_k", top_k),
            ("wide_search_top_k", wide_search_top_k),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        self.num_seeds = num_seeds
        self.walks_per_seed = walks_per_seed
        self.max_depth = max_depth
        self.top_k = top_k
        self.random_seed = random_seed
        self.memory_fragment = memory_fragment
        self.vector_search = vector_search
        self.collections = list(collections) if collections else list(DEFAULT_COLLECTIONS)
        self.wide_search_top_k = wide_search_top_k
        self.scored_candidates: List[Path] = []

    async def run(self, query: str) -> List[Path]:
        """Sample, score, and return the best paths for the query, ordered best to worst."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        self.scored_candidates = []

        vector_search = self.vector_search or NodeEdgeVectorSearch()
        await vector_search.embed_and_retrieve_distances(
            query=query,
            collections=self.collections,
            wide_search_limit=self.wide_search_top_k,
        )
        if not vector_search.has_results():
            return []

        memory_fragment = self.memory_fragment
        if memory_fragment is None:
            memory_fragment = await get_memory_fragment(
                relevant_ids_to_filter=vector_search.extract_relevant_node_ids(),
            )
        if not memory_fragment.nodes or not memory_fragment.edges:
            return []

        await memory_fragment.map_vector_distances_to_graph_nodes(
            node_distances=vector_search.node_distances
        )
        await memory_fragment.map_vector_distances_to_graph_edges(
            edge_distances=vector_search.edge_distances
        )

        penalty = memory_fragment.triplet_distance_penalty
        seeds = self._select_seeds(memory_fragment, penalty)
        if not seeds:
            return []

        # Fresh RNG per run so repeated run() calls with the same seed reproduce paths.
        rng = random.Random(self.random_seed)
        sampled_paths: List[Path] = []
        for seed in seeds:
            for _ in range(self.walks_per_seed):
                path = self._sample_walk(seed, rng)
                if path is not None:
                    sampled_paths.append(path)

        unique_paths = self._deduplicate(sampled_paths)
        for path in unique_paths:
            path.score = self._score_path(path, penalty)
        unique_paths.sort(key=lambda path: (path.score, path.dedup_key))

        self.scored_candidates = unique_paths
        return unique_paths[: self.top_k]

    def _select_seeds(self, memory_fragment: CogneeGraph, penalty: float) -> List[Node]:
        """Deterministically pick the vector-closest nodes as walk starting points."""
        candidates: List[Tuple[float, str, Node]] = []
        for node in memory_fragment.nodes.values():
            distance = self._element_distance(node)
            if distance is None or distance >= penalty:
                continue
            candidates.append((distance, node.id, node))
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        return [node for _, _, node in candidates[: self.num_seeds]]

    def _sample_walk(self, seed: Node, rng: random.Random) -> Optional[Path]:
        """Sample one walk from the seed, up to max_depth steps, never revisiting nodes."""
        nodes = [seed]
        edges: List[Edge] = []
        visited = {seed.id}
        current = seed
        for _ in range(self.max_depth):
            step = self._step(current, visited, rng)
            if step is None:
                break
            edge, neighbor = step
            edges.append(edge)
            nodes.append(neighbor)
            visited.add(neighbor.id)
            current = neighbor
        if not edges:
            return None
        return Path(nodes=nodes, edges=edges)

    def _step(self, current: Node, visited: set, rng: random.Random) -> Optional[Tuple[Edge, Node]]:
        """Pick the next edge uniformly among unvisited neighbors, in stable order."""
        candidates: List[Tuple[Edge, Node]] = []
        for edge in current.skeleton_edges:
            neighbor = edge.node2 if edge.node1 == current else edge.node1
            if neighbor.id in visited:
                continue
            candidates.append((edge, neighbor))
        if not candidates:
            return None
        # Sorting before sampling makes the seeded RNG's picks independent of the
        # (projection-order-dependent) skeleton_edges order.
        candidates.sort(key=lambda candidate: candidate[0].stable_id())
        return rng.choice(candidates)

    @staticmethod
    def _deduplicate(paths: List[Path]) -> List[Path]:
        unique: dict = {}
        for path in paths:
            unique.setdefault(path.dedup_key, path)
        return list(unique.values())

    def _score_path(self, path: Path, penalty: float) -> float:
        """First score: mean vector distance over the path's nodes and edges (lower is better)."""
        elements: List[Any] = [*path.nodes, *path.edges]
        total = 0.0
        for element in elements:
            distance = self._element_distance(element)
            total += penalty if distance is None else distance
        return total / len(elements)

    @staticmethod
    def _element_distance(element: Any) -> Optional[float]:
        """Read an element's mapped vector distance; supports list (per-query) or scalar."""
        distance = element.attributes.get("vector_distance")
        if isinstance(distance, list):
            distance = distance[0] if distance else None
        try:
            return float(distance)
        except (TypeError, ValueError):
            return None
