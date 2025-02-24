from typing import Optional

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


class GraphSummaryCompletionRetriever(GraphCompletionRetriever):
    """Retriever for handling graph-based completion searches with summarized context."""

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        summarize_prompt_path: str = "summarize_search_results.txt",
        top_k: int = 5,
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
        system_prompt = read_query_prompt(self.summarize_prompt_path)

        llm_client = get_llm_client()
        return await llm_client.acreate_structured_output(
            text_input=direct_text,
            system_prompt=system_prompt,
            response_model=str,
        )
