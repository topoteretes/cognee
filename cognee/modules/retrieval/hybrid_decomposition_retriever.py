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
from cognee.modules.retrieval.utils.resolved_context import user_resolution_scope
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

    supports_session_turn_preparation = False

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
        if type(max_subqueries) is not int or max_subqueries < 1:
            raise ValueError("max_subqueries must be a positive integer")
        self.max_subqueries = max_subqueries
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
        for name, value in (
            ("merged_chunks_limit", merged_chunks_limit),
            ("merged_entities_limit", merged_entities_limit),
            ("merged_facts_limit", merged_facts_limit),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        self._decomposition_state: HybridDecompositionRunState | None = None

    @property
    def decomposition_state(self) -> HybridDecompositionRunState | None:
        """Inspect the latest retrieval plan and evidence without running another search."""
        return self._decomposition_state

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
        prompt = read_query_prompt(self.decomposition_system_prompt_path)
        return prompt.replace("{max_subqueries}", str(self.max_subqueries)) if prompt else prompt

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
        try:
            system_prompt = self._decomposition_prompt()
            if not system_prompt or not system_prompt.strip():
                logger.warning("Decomposition prompt not found, falling back to original query.")
                return [query]
            decomposition = await LLMGateway.acreate_structured_output(
                text_input=self._decomposition_input(query, pass_one_context),
                system_prompt=system_prompt,
                response_model=QueryDecomposition,
            )
            return normalize_subqueries(
                query,
                getattr(decomposition, "subqueries", None),
                max_subqueries=self.max_subqueries,
            )
        except Exception as error:
            logger.warning(
                "Query decomposition failed, falling back to original query: %s",
                error,
                exc_info=True,
            )
            return [query]

    async def decompose_query(self, query: str, context: str = "") -> list[str]:
        """Generate subqueries only, with optional caller-supplied source context.

        Makes one decomposition LLM call. Does not open a graph/vector engine,
        retrieve, embed, generate a final answer, or change retrieval run state.
        ``context`` is supplied evidence, not evidence retrieved by this method.
        """
        self._validate_single_query_input(query, None)
        return await self._decompose_query(query, context)

    def _merged_limits(self, leg_count: int, results: list[dict] | None = None) -> dict[str, int]:
        scale = leg_count + 1  # pass 1 counts as a leg
        limits = {}
        for channel, explicit, budget in (
            ("chunks", self.merged_chunks_limit, self.chunks_top_k),
            ("entities", self.merged_entities_limit, self.entities_top_k),
            ("facts", self.merged_facts_limit, self.facts_top_k),
        ):
            # Hybrid may return more facts than nominal facts_top_k when the
            # entity lane is empty. Size the default from actual results too.
            limits[f"{channel}_limit"] = (
                explicit
                if explicit is not None
                else max(
                    budget * scale, sum(len(result.get(channel) or []) for result in results or [])
                )
            )
        return limits

    async def _retrieve_subqueries(
        self, query: str, subqueries: list[str], pass_one: dict
    ) -> list[dict]:
        """Retrieve expansion legs concurrently, reusing the baseline for the original query."""

        async def retrieve_leg(subquery: str) -> dict:
            if subquery == query:
                return pass_one
            try:
                return await self._retrieve_one(subquery)
            except Exception as error:
                logger.warning(
                    "Decomposition leg retrieval failed; retaining pass one: %s",
                    error,
                    exc_info=True,
                )
                return empty_hybrid_result()

        return list(await asyncio.gather(*(retrieve_leg(q) for q in subqueries)))

    @user_resolution_scope()
    async def get_retrieved_objects(
        self, query: str | None = None, query_batch: list[str] | None = None
    ) -> dict[str, Any]:
        """Pass 1, decomposition, per-leg retrieval, and the merged union."""
        self._validate_single_query_input(query, query_batch)

        self._decomposition_state = None
        # Inherit engine initialization, empty-graph errors and the base retrieval
        # semantics. The outer scope shares user-resolution reads across all legs.
        pass_one = await super().get_retrieved_objects(query=query)
        pass_one_context = await super().get_context_from_objects(
            query=query, retrieved_objects=pass_one
        )
        subqueries = await self.decompose_query(
            query, context=pass_one_context if isinstance(pass_one_context, str) else ""
        )

        legs = await self._retrieve_subqueries(query, subqueries, pass_one)

        results = [pass_one, *legs]
        merged = merge_hybrid_results_union(
            results,
            **self._merged_limits(len(legs), results),
            preserve_first=True,
        )
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
