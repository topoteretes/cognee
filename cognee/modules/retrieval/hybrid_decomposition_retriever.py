"""Hybrid retrieval with context-primed query decomposition.

Multi-part questions under-retrieve on one hybrid search: a single embedding
cannot cover every quantity, date and role the question asks about. This
retriever runs the plain hybrid retrieval for the full question first (pass 1),
lets the LLM decompose question plus pass-1 context into focused subqueries,
retrieves every subquery through the hybrid lanes as one batch, and merges all
legs into one result of the usual hybrid shape.

Two invariants hold: pass-1 objects always contribute, so the merged result is
never smaller than plain ``HYBRID_COMPLETION``; and a failed decomposition falls
back to the original query, so decomposition can never fail a search.

Scope: a single query per call. ``query_batch`` is rejected and the
session-cache completion path is not used.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import (
    QueryDecomposition,
    normalize_subqueries,
)
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridDecompositionRetriever")

DEFAULT_MAX_SUBQUERIES = 7
DEFAULT_DECOMPOSITION_PROMPT_PATH = "hybrid_decomposition_system_prompt.txt"


@dataclass
class HybridDecompositionRunState:
    """Everything one decomposed retrieval produced, kept so the separate
    context and completion calls ``search()`` makes do not retrieve again."""

    original_query: str
    subqueries: list[str] = field(default_factory=list)
    pass_one: dict = field(default_factory=dict)
    legs: list[dict] = field(default_factory=list)
    merged: dict = field(default_factory=dict)


class HybridDecompositionRetriever(HybridRetriever):
    """Hybrid retriever that decomposes the question after a first retrieval pass.

    Everything expensive is inherited: the chunk, entity and fact lanes, context
    formatting, references and completion. This class only orchestrates the two
    passes, the decomposition call and the merge.
    """

    def __init__(
        self,
        *,
        max_subqueries: int = DEFAULT_MAX_SUBQUERIES,
        decomposition_system_prompt_path: str = DEFAULT_DECOMPOSITION_PROMPT_PATH,
        decomposition_system_prompt: str | None = None,
        merged_chunks_limit: int | None = None,
        merged_entities_limit: int | None = None,
        merged_facts_limit: int | None = None,
        **hybrid_kwargs: Any,
    ):
        super().__init__(**hybrid_kwargs)
        self.max_subqueries = max(1, int(max_subqueries or DEFAULT_MAX_SUBQUERIES))
        # Inline text wins over the file, the same convention the answer prompts use
        # via ``system_prompt`` / ``system_prompt_path``.
        self.decomposition_system_prompt = decomposition_system_prompt or None
        self.decomposition_system_prompt_path = (
            decomposition_system_prompt_path or DEFAULT_DECOMPOSITION_PROMPT_PATH
        )
        # None means "per-leg budget times (legs + 1)": a scaled union in which
        # nothing any leg returned is dropped.
        self.merged_chunks_limit = merged_chunks_limit
        self.merged_entities_limit = merged_entities_limit
        self.merged_facts_limit = merged_facts_limit
        self._decomposition_state: HybridDecompositionRunState | None = None

    def _use_session_cache(self) -> bool:
        # Out of scope: the session path merges a conversational retrieval into the
        # result and rewrites the query, which would interleave with decomposition.
        # Always take the plain branch for validation and completion.
        return False

    def _validate_single_query_input(self, query: str | None, query_batch: list[str] | None):
        if query_batch is not None:
            raise QueryValidationError(
                message=(
                    "HybridDecompositionRetriever accepts only a single query. "
                    "Decomposition batching is handled internally."
                )
            )
        validate_retriever_input(query, None, False)

    def _decomposition_prompt(self) -> str | None:
        if self.decomposition_system_prompt:
            return self.decomposition_system_prompt
        return read_query_prompt(self.decomposition_system_prompt_path)

    @staticmethod
    def _decomposition_input(query: str, pass_one_context: str) -> str:
        """Question plus the delimited pass-1 context the decomposition should read."""
        if not pass_one_context:
            return query
        return (
            f"Question:\n{query}\n\n"
            "Context retrieved for the question so far:\n"
            f"<context>\n{pass_one_context}\n</context>"
        )

    async def _decompose_query(self, query: str, pass_one_context: str) -> list[str]:
        """Subqueries for ``query``; the query itself whenever decomposition cannot run."""
        system_prompt = self._decomposition_prompt()
        if not system_prompt:
            logger.warning("Decomposition prompt not found, falling back to original query.")
            return [query]

        try:
            decomposition = await LLMGateway.acreate_structured_output(
                text_input=self._decomposition_input(query, pass_one_context),
                system_prompt=system_prompt,
                response_model=QueryDecomposition,
            )
        except Exception as error:
            logger.warning(
                "Query decomposition failed, falling back to original query: %s",
                error,
                exc_info=True,
            )
            return [query]

        return normalize_subqueries(
            query,
            getattr(decomposition, "subqueries", None),
            max_subqueries=self.max_subqueries,
        )

    def _merged_limits(self, leg_count: int) -> dict[str, int]:
        scale = leg_count + 1  # pass 1 counts as a leg
        return {
            "chunks_limit": self.merged_chunks_limit or self.chunks_top_k * scale,
            "entities_limit": self.merged_entities_limit or self.entities_top_k * scale,
            "facts_limit": self.merged_facts_limit or self.facts_top_k * scale,
        }

    async def get_retrieved_objects(
        self, query: str | None = None, query_batch: list[str] | None = None
    ) -> dict[str, Any]:
        """Pass 1, decomposition, per-leg retrieval, and the merged union."""
        self._validate_single_query_input(query, query_batch)

        self._decomposition_state = None
        self._unified_engine = await get_unified_engine()
        if await self._unified_engine.graph.is_empty():
            logger.warning("Search attempt on an empty knowledge graph")
            result = empty_hybrid_result()
            self._decomposition_state = HybridDecompositionRunState(
                original_query=query, subqueries=[query], pass_one=result, merged=result
            )
            return result

        # ``_retrieve_one`` is the parent's per-query core behind both its single and
        # batch paths; calling it directly keeps one graph check and one validation
        # for the whole run instead of one per leg.
        pass_one = await self._retrieve_one(query)
        pass_one_context = await super().get_context_from_objects(
            query=query, retrieved_objects=pass_one
        )
        subqueries = await self._decompose_query(
            query, pass_one_context if isinstance(pass_one_context, str) else ""
        )
        legs = list(
            await asyncio.gather(*[self._retrieve_one(subquery) for subquery in subqueries])
        )

        merged = merge_hybrid_results_union([pass_one, *legs], **self._merged_limits(len(legs)))
        self._decomposition_state = HybridDecompositionRunState(
            original_query=query,
            subqueries=subqueries,
            pass_one=pass_one,
            legs=legs,
            merged=merged,
        )
        return merged

    async def _ensure_state(self, query: str | None) -> HybridDecompositionRunState:
        state = self._decomposition_state
        if state is not None and state.original_query == query:
            return state
        if query is None:
            raise QueryValidationError(message="A non-empty query is required.")
        await self.get_retrieved_objects(query=query)
        return self._decomposition_state

    async def get_context_from_objects(
        self,
        query: str | None = None,
        query_batch: list[str] | None = None,
        retrieved_objects: Any = None,
    ) -> Any:
        """Format the merged objects for the original question."""
        self._validate_single_query_input(query, query_batch)
        if retrieved_objects is None:
            retrieved_objects = (await self._ensure_state(query)).merged
        return await super().get_context_from_objects(
            query=query, retrieved_objects=retrieved_objects
        )

    async def get_completion_from_context(
        self,
        query: str | None = None,
        query_batch: list[str] | None = None,
        retrieved_objects: Any = None,
        context: Any = None,
        effective_query: str | None = None,
        turn_preparation=None,
    ) -> list[Any]:
        """Answer the original question over the merged context."""
        self._validate_single_query_input(query, query_batch)
        if retrieved_objects is None:
            retrieved_objects = (await self._ensure_state(query)).merged
        if context is None:
            context = await self.get_context_from_objects(
                query=query, retrieved_objects=retrieved_objects
            )
        return await super().get_completion_from_context(
            query=query,
            retrieved_objects=retrieved_objects,
            context=context,
            effective_query=effective_query,
            turn_preparation=turn_preparation,
        )
