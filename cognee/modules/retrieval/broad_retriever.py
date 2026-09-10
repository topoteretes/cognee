"""BROAD search type: graph completion sized to the model's context window (SDK-324).

GRAPH_COMPLETION ranks triplets against the query and keeps the ``top_k`` best,
drawn from the neighbourhood of the 100 nearest nodes. Questions that need the
whole dataset in view ("who has the most ...", "how many ...") see a sliver of it
that way. BROAD ranks every triplet in the graph with the same scorer and keeps
as many as the configured LLM's context window holds: ``top_k`` is not an input,
the model's window is the budget.
"""

import sys

import litellm

from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import build_completion_prompts
from cognee.shared.logging_utils import get_logger

logger = get_logger("BroadRetriever")

# Rank every triplet in the graph; the context window, not a count, cuts the list.
ALL_TRIPLETS = sys.maxsize

# Share of the window held back for what the budget cannot count exactly:
# tokenizer drift (context is counted with tiktoken, not the model's own
# tokenizer) and the session history / preference text added at completion time.
CONTEXT_SAFETY_MARGIN = 0.05


def resolve_context_window(context_window_tokens: int | None = None) -> int:
    """Input-token window of the configured LLM; an explicit value wins.

    Raises when litellm does not know the model, since BROAD cannot size its
    context without a window — pass ``context_window_tokens`` for such models.
    """
    if context_window_tokens is not None:
        return context_window_tokens

    model = get_llm_context_config().llm_model
    try:
        window = litellm.get_model_info(model).get("max_input_tokens")
    except Exception as error:
        raise ValueError(
            f"BROAD search needs the context window of {model!r}, which litellm does not "
            "know. Pass retriever_specific_config={'context_window_tokens': <int>}."
        ) from error
    if not window:
        raise ValueError(
            f"litellm reports no max_input_tokens for {model!r}. "
            "Pass retriever_specific_config={'context_window_tokens': <int>}."
        )
    return window


class BroadRetriever(GraphCompletionRetriever):
    """GRAPH_COMPLETION over the whole graph, cut by the LLM's context window."""

    def __init__(self, context_window_tokens: int | None = None, **kwargs):
        super().__init__(top_k=ALL_TRIPLETS, **kwargs)
        # Score the whole graph rather than the neighbourhood of the nearest
        # nodes: an aggregate question needs every record to be a candidate.
        self.wide_search_top_k = None
        self.context_window_tokens = context_window_tokens
        self.tokenizer = TikTokenTokenizer()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.extract_tokens(text))

    def context_token_budget(self, query: str) -> int:
        """Tokens the rendered graph context may use for this query."""
        window = resolve_context_window(self.context_window_tokens)
        user_prompt, system_prompt = build_completion_prompts(
            query=query,
            context="",
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
        )
        prompt_shell = self.count_tokens(user_prompt) + self.count_tokens(system_prompt)
        # litellm's max_input_tokens is the whole window for some models (shared
        # with the output) and input-only for others; reserving the configured
        # completion length is correct for both.
        output_reserve = get_llm_context_config().llm_max_completion_tokens
        budget = window - prompt_shell - output_reserve - int(window * CONTEXT_SAFETY_MARGIN)
        if budget <= 0:
            raise ValueError(
                f"A context window of {window} tokens leaves no room for BROAD context "
                f"after the prompt ({prompt_shell}) and output ({output_reserve}) reserves."
            )
        return budget

    async def fit_to_window(self, query: str, triplets: list[Edge]) -> list[Edge]:
        """Longest best-first prefix of ``triplets`` whose rendered text fits the budget.

        Rendered size only grows with the prefix, so a binary search finds the
        cut in log2(n) renders.
        """
        budget = self.context_token_budget(query)
        low, high = 0, len(triplets)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count_tokens(await self.resolve_edges_to_text(triplets[:mid])) <= budget:
                low = mid
            else:
                high = mid - 1

        logger.info(
            "BROAD context: %d of %d ranked triplets fit the %d-token budget",
            low,
            len(triplets),
            budget,
        )
        return triplets[:low]

    async def get_retrieved_objects(
        self, query: str | None = None, query_batch: list[str] | None = None
    ) -> list[Edge] | list[list[Edge]]:
        """Rank every triplet, then keep what fits the window.

        Cutting here rather than at rendering keeps session bookkeeping and
        evidence references limited to the triplets the LLM actually receives.
        """
        ranked = await super().get_retrieved_objects(query=query, query_batch=query_batch)
        if query_batch:
            return [
                await self.fit_to_window(batched_query, batched_triplets)
                for batched_query, batched_triplets in zip(query_batch, ranked)
            ]
        return await self.fit_to_window(query, ranked)
