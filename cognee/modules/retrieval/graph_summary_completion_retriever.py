from typing import Optional

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import summarize_text


class GraphSummaryCompletionRetriever(GraphCompletionRetriever):
    """Retriever for handling graph-based completion searches with summarized context."""

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        summarize_prompt_path: str = "summarize_search_results.txt",
        top_k: Optional[int] = 5,
    ):
        """Initialize retriever with default prompt paths and search parameters."""
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
        )
        self.summarize_prompt_path = summarize_prompt_path

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """Converts retrieved graph edges into a summary without redundancies."""
        direct_text = await super().resolve_edges_to_text(retrieved_edges)
        return await summarize_text(direct_text, self.summarize_prompt_path)
