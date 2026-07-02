from typing import Any, Dict, List, Optional, Type, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.modules.retrieval.utils.centrality import (
    degree_centrality,
    pagerank,
    rank_top_k,
    select_rankable_node_ids,
)

logger = get_logger("CentralityRetriever")

# Vector collections used to resolve a query to seed entity nodes. Mirrors the
# entity-leaning defaults used by brute_force_triplet_search.
_SEED_COLLECTIONS = [
    "Entity_name",
    "EntityType_name",
    "TextSummary_text",
    "DocumentChunk_text",
]

_SUPPORTED_MODES = ("pagerank", "degree", "frequency")


class CentralityRetriever(BaseRetriever):
    """Rank graph entities by structural importance (SearchType.CENTRALITY).

    Unlike the vector-distance / feedback-weight retrievers, this surfaces the
    *hub* entities of the knowledge graph (or of a query-relevant subgraph),
    answering "what are the key entities/themes here?".

    Importance is computed directly on cognee's native graph projection
    (``project_graph_from_db`` / ``project_neighborhood_from_db`` via
    ``get_memory_fragment``) — no networkx or Neo4j GDS (issue #3378).

    Modes:
        - ``pagerank``  (default) query-personalized directed PageRank.
        - ``degree``    distinct-neighbor degree centrality (cheapest computed).
        - ``frequency`` stored node frequency weights (zero graph traversal);
          falls back to ``degree`` when the backend does not expose them.

    Scoping: with a query anchor, seeds are resolved by vector search and a
    ``neighborhood_depth``-hop fragment is projected, so importance is biased to
    the question; with no anchor the full graph is projected and ranked globally.
    """

    def __init__(
        self,
        top_k: Optional[int] = 10,
        mode: str = "pagerank",
        neighborhood_depth: int = 2,
        neighborhood_seed_top_k: int = 10,
        max_nodes: int = 5000,
        summarize: bool = True,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        response_model: Type = str,
        session_id: Optional[str] = None,
    ):
        self.top_k = top_k if top_k is not None else 10
        mode = (mode or "pagerank").lower()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported centrality mode '{mode}'. Expected one of {_SUPPORTED_MODES}."
            )
        self.mode = mode
        self.neighborhood_depth = neighborhood_depth
        self.neighborhood_seed_top_k = neighborhood_seed_top_k
        self.max_nodes = max_nodes
        self.summarize = summarize
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.response_model = response_model
        self.session_id = session_id

    async def _resolve_seed_ids(self, query: Optional[str]) -> List[str]:
        """Resolve a query to seed node ids via vector search (empty if no anchor)."""
        if not query:
            return []
        try:
            vector_search = NodeEdgeVectorSearch()
            await vector_search.embed_and_retrieve_distances(
                query=query,
                query_batch=None,
                collections=_SEED_COLLECTIONS,
                wide_search_limit=self.neighborhood_seed_top_k,
            )
            if not vector_search.has_results():
                return []
            return vector_search.extract_relevant_node_ids()
        except Exception as error:
            # A failed anchor lookup must degrade to global ranking, not crash.
            logger.warning("Centrality seed resolution failed, ranking globally: %s", error)
            return []

    async def _score_nodes(
        self, graph, rankable_ids: List[str], seed_ids: List[str]
    ) -> Dict[str, float]:
        """Compute centrality scores for the rankable nodes in the given mode."""
        if self.mode == "frequency":
            try:
                graph_engine = await get_graph_engine()
                weights = await graph_engine.get_node_frequency_weights(rankable_ids)
                if weights:
                    return {nid: float(weights.get(nid, 0.0)) for nid in rankable_ids}
                logger.info("No frequency weights returned; falling back to degree centrality.")
            except NotImplementedError:
                logger.info("Backend does not expose frequency weights; using degree centrality.")
            except Exception as error:
                logger.warning("Frequency lookup failed, using degree centrality: %s", error)
            return degree_centrality(graph, rankable_ids)

        if self.mode == "degree":
            return degree_centrality(graph, rankable_ids)

        # pagerank: bias the teleport toward the (rankable) query seeds.
        rankable_set = set(rankable_ids)
        personalization = {nid: 1.0 for nid in seed_ids if nid in rankable_set} or None
        return pagerank(graph, rankable_ids, personalization=personalization)

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Project the relevant graph and return top-k entities by importance."""
        graph_engine = await get_graph_engine()
        if await graph_engine.is_empty():
            logger.warning("Centrality search attempted on an empty knowledge graph.")
            return []

        seed_ids = await self._resolve_seed_ids(query)

        if seed_ids:
            graph = await get_memory_fragment(
                relevant_ids_to_filter=seed_ids,
                neighborhood_depth=self.neighborhood_depth,
                neighborhood_seed_top_k=self.neighborhood_seed_top_k,
                graph_engine=graph_engine,
            )
        else:
            graph = await get_memory_fragment(graph_engine=graph_engine)

        rankable_ids = select_rankable_node_ids(graph)
        if not rankable_ids:
            return []

        # Safety valve: PageRank is O(iter * E); on very large global projections
        # pre-trim to the highest-degree nodes (cheap, O(E)) before ranking.
        if self.mode == "pagerank" and len(rankable_ids) > self.max_nodes:
            logger.info(
                "Projection has %d rankable nodes (> max_nodes=%d); pre-trimming by degree.",
                len(rankable_ids),
                self.max_nodes,
            )
            degree_scores = degree_centrality(graph, rankable_ids)
            rankable_ids = rank_top_k(degree_scores, self.max_nodes)

        scores = await self._score_nodes(graph, rankable_ids, seed_ids)
        top_ids = rank_top_k(scores, self.top_k)

        results: List[Dict[str, Any]] = []
        for rank, node_id in enumerate(top_ids, start=1):
            node = graph.get_node(node_id)
            attributes = node.attributes if node else {}
            results.append(
                {
                    "id": node_id,
                    "name": attributes.get("name") or attributes.get("text"),
                    "type": attributes.get("type"),
                    "score": scores.get(node_id, 0.0),
                    "rank": rank,
                }
            )
        return results

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        retrieved_objects: Any = None,
        query_batch: Optional[List[str]] = None,
    ) -> str:
        """Render the ranked entities into a human/LLM-readable context block."""
        if not retrieved_objects:
            return ""
        lines = []
        for item in retrieved_objects:
            label = item.get("name") or item.get("id")
            node_type = item.get("type")
            type_suffix = f" ({node_type})" if node_type else ""
            lines.append(
                f"{item.get('rank')}. {label}{type_suffix} — importance {item.get('score', 0.0):.4f}"
            )
        return "\n".join(lines)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        retrieved_objects: Any = None,
        context: Any = None,
        query_batch: Optional[List[str]] = None,
    ) -> Union[List[str], List[dict]]:
        """Return the ranked entities, optionally summarized by an LLM.

        The structural ranking is the primary product; summarization is an
        optional convenience that turns the ranked hubs into prose.
        """
        if not context:
            return [""]
        if not self.summarize:
            return [context]
        completion = await generate_completion(
            query=query or "",
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
            response_model=self.response_model,
        )
        return [completion]
