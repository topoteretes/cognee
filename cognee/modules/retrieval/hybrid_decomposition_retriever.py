"""Hybrid retrieval with context-primed query decomposition.

Multi-part questions under-retrieve on one hybrid search: a single embedding
cannot cover every quantity, date and role the question asks about. This
retriever runs the plain hybrid retrieval for the full question first (pass 1),
lets the LLM decompose question plus pass-1 context into focused subqueries,
retrieves every subquery through the hybrid lanes and merges all results into
one of the usual hybrid shape.

Pass-1 objects always contribute, so the merged result never holds less than
plain ``HYBRID_COMPLETION``. A missing prompt or a failed planning call falls
back to the original query; retrieval errors propagate as they do for plain
hybrid.
"""

import asyncio
from typing import Any

from jinja2 import TemplateNotFound

from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import decompose_query
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridDecompositionRetriever")

DEFAULT_MAX_SUBQUERIES = 7
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
        if max_subqueries < 1:
            raise ValueError("max_subqueries must be at least 1")
        self.max_subqueries = max_subqueries
        self.decomposition_system_prompt_path = decomposition_system_prompt_path
        # Inline text wins over the file, as with ``system_prompt`` / ``system_prompt_path``.
        self.decomposition_system_prompt = decomposition_system_prompt

    async def get_retrieved_objects(
        self, query: str | None = None, query_batch: list[str] | None = None
    ) -> Any:
        if query_batch is not None:
            raise QueryValidationError(
                message="HybridDecompositionRetriever accepts only a single query."
            )
        return await super().get_retrieved_objects(query=query)

    def _decomposition_prompt(self) -> str | None:
        if self.decomposition_system_prompt:
            return self.decomposition_system_prompt
        try:
            return render_prompt(
                self.decomposition_system_prompt_path, {"max_subqueries": self.max_subqueries}
            )
        except TemplateNotFound:
            return None

    async def _retrieve_one(self, query: str) -> dict[str, Any]:
        """Pass 1, context-primed decomposition, one hybrid retrieval per new subquery, union."""
        first = await super()._retrieve_one(query)
        subqueries = await decompose_query(
            query,
            system_prompt=self._decomposition_prompt(),
            context=await self.get_context_from_objects(query=query, retrieved_objects=first),
            max_subqueries=self.max_subqueries,
        )
        logger.debug("Hybrid decomposition subqueries for %r: %s", query, subqueries)
        retrieve = super()._retrieve_one
        others = await asyncio.gather(*(retrieve(q) for q in subqueries if q != query))
        return merge_hybrid_results_union([first, *others])
