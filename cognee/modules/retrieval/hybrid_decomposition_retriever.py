"""Hybrid retrieval with context-primed query decomposition.

Multi-part questions under-retrieve on one hybrid search: a single embedding
cannot cover every quantity, date and role the question asks about. This
retriever runs the plain hybrid retrieval for the full question first (pass 1),
lets the LLM decompose question plus pass-1 context into focused subqueries,
retrieves every subquery through the hybrid lanes and merges all results into
one of the usual hybrid shape.

Pass-1 objects always contribute, so the merged result never holds less than
plain ``HYBRID_COMPLETION``. Errors propagate: a missing decomposition prompt, a
failed planning call and a failed retrieval each fail the search.

Cost. The context is not capped. The union holds up to ``1 + max_subqueries``
retrievals, so with the default 7 subqueries each lane can reach 8 times its
top-k (``chunks_top_k``, ``entities_top_k``, ``facts_top_k``), and an entity
keeps the edges every retrieval showed for it, which can exceed
``max_edges_per_entity``. The pass-1 context is sent to the model twice, once to
the planner and again inside the merged context for the answer. One search makes
two LLM calls and up to ``1 + max_subqueries`` embedding calls and lane
retrievals. Size ``top_k`` and ``max_subqueries`` with that in mind.
"""

import asyncio
from typing import Any

from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import decompose_query
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridDecompositionRetriever")

DEFAULT_MAX_SUBQUERIES = 7
# Each subquery is one more hybrid retrieval, all run concurrently, so the caller-set
# count is bounded.
MAX_SUBQUERIES = 20
DEFAULT_DECOMPOSITION_PROMPT_PATH = "hybrid_decomposition_system_prompt.txt"


class HybridDecompositionRetriever(HybridRetriever):
    """Hybrid retriever that decomposes the question after a first retrieval pass.

    Only the per-query retrieval is overridden. Validation, the empty-graph
    error, context formatting, references, sessions and completion are inherited.
    """

    def __init__(
        self,
        *,
        max_subqueries: int = DEFAULT_MAX_SUBQUERIES,
        decomposition_system_prompt_path: str = DEFAULT_DECOMPOSITION_PROMPT_PATH,
        decomposition_system_prompt: str | None = None,
        **hybrid_kwargs: Any,
    ):
        super().__init__(**hybrid_kwargs)
        # bool is an int subclass; True must not pass as 1.
        if (
            isinstance(max_subqueries, bool)
            or not isinstance(max_subqueries, int)
            or not 1 <= max_subqueries <= MAX_SUBQUERIES
        ):
            raise QueryValidationError(
                message=f"max_subqueries must be an integer between 1 and {MAX_SUBQUERIES}."
            )
        self.max_subqueries = max_subqueries
        self.decomposition_system_prompt_path = decomposition_system_prompt_path
        # Inline text wins over the file, as with ``system_prompt`` / ``system_prompt_path``.
        self.decomposition_system_prompt = decomposition_system_prompt
        # Resolved here so a wrong prompt path fails before any retrieval runs.
        self._decomposition_prompt = decomposition_system_prompt or self._read_prompt_file()

    async def get_retrieved_objects(
        self, query: str | None = None, query_batch: list[str] | None = None
    ) -> Any:
        if query_batch is not None:
            raise QueryValidationError(
                message="HybridDecompositionRetriever accepts only a single query."
            )
        return await super().get_retrieved_objects(query=query)

    def _read_prompt_file(self) -> str:
        # read_query_prompt refuses an absolute path or one that leaves the prompt directory.
        if read_query_prompt(self.decomposition_system_prompt_path) is None:
            raise QueryValidationError(
                message=(
                    f"Decomposition prompt file not found: {self.decomposition_system_prompt_path}"
                )
            )
        return render_prompt(
            self.decomposition_system_prompt_path, {"max_subqueries": self.max_subqueries}
        )

    async def _retrieve_one(self, query: str) -> dict[str, Any]:
        """Pass 1, context-primed decomposition, one hybrid retrieval per new subquery, union."""
        first = await super()._retrieve_one(query)
        subqueries = await decompose_query(
            query,
            system_prompt=self._decomposition_prompt,
            context=await self.get_context_from_objects(query=query, retrieved_objects=first),
            max_subqueries=self.max_subqueries,
        )
        logger.debug("Hybrid decomposition subqueries for %r: %s", query, subqueries)
        retrieve = super()._retrieve_one
        others = await asyncio.gather(*(retrieve(q) for q in subqueries if q != query))
        return merge_hybrid_results_union([first, *others])
