from typing import Any, List, Optional, Union

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger

logger = get_logger("MMRRetriever")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    top_k: int,
    lambda_mult: float,
) -> List[int]:
    """Return indices of selected candidates via Maximal Marginal Relevance.

    Score = lambda_mult * sim(query, candidate) - (1 - lambda_mult) * max sim(candidate, selected)
    """
    if not candidate_embeddings:
        return []
    top_k = min(top_k, len(candidate_embeddings))

    selected: List[int] = []
    remaining = list(range(len(candidate_embeddings)))

    for _ in range(top_k):
        best_idx, best_score = -1, float("-inf")
        for i in remaining:
            relevance = _cosine_similarity(query_embedding, candidate_embeddings[i])
            if selected:
                redundancy = max(
                    _cosine_similarity(candidate_embeddings[i], candidate_embeddings[j])
                    for j in selected
                )
            else:
                redundancy = 0.0
            score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


class MMRRetriever(BaseRetriever):
    """Retrieves document chunks reranked by Maximal Marginal Relevance.

    Fetches `fetch_k` candidates from the vector store, then selects
    `top_k` of them that balance relevance to the query against
    redundancy among themselves.
    """

    def __init__(
        self,
        top_k: Optional[int] = 5,
        fetch_k: Optional[int] = 20,
        lambda_mult: Optional[float] = 0.5,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ):
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator

    async def get_retrieved_objects(self, query: str) -> Any:
        logger.info(f"MMR retrieval for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")

        unified = await get_unified_engine()
        vector_engine = unified.vector

        try:
            candidates = await vector_engine.search(
                "DocumentChunk_text",
                query,
                limit=self.fetch_k,
                include_payload=True,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
            )
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

        if not candidates:
            return []

        texts = [c.payload.get("text", "") for c in candidates]
        all_embeddings = await vector_engine.embedding_engine.embed_text([query] + texts)
        query_embedding = all_embeddings[0]
        candidate_embeddings = all_embeddings[1:]

        selected_indices = mmr_select(
            query_embedding, candidate_embeddings, self.top_k, self.lambda_mult
        )
        logger.info(f"MMR selected {len(selected_indices)} of {len(candidates)} candidates")
        return [candidates[i] for i in selected_indices]

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        if not retrieved_objects:
            return ""
        return "\n".join(c.payload["text"] for c in retrieved_objects)

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        if not retrieved_objects:
            return []
        return [c.payload for c in retrieved_objects]
