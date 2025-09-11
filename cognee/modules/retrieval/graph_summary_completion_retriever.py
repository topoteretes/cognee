from typing import Optional, Type, List

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import summarize_text


class GraphSummaryCompletionRetriever(GraphCompletionRetriever):
    """
    Retriever for handling graph-based completion searches with summarized context.

    This class inherits from the GraphCompletionRetriever and is intended to manage the
    retrieval of graph edges with an added functionality to summarize the retrieved
    information efficiently. Public methods include:

    - __init__()
    - resolve_edges_to_text()
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        summarize_prompt_path: str = "summarize_search_results.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        save_interaction: bool = False,
    ):
        """Initialize retriever with default prompt paths and search parameters."""
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            save_interaction=save_interaction,
            system_prompt=system_prompt,
        )
        self.summarize_prompt_path = summarize_prompt_path

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """
        Convert retrieved graph edges into a summary without redundancies.

        This asynchronous method processes a list of retrieved edges and summarizes their
        content using a specified prompt path. It relies on the parent's implementation to
        convert the edges to text before summarizing. Raises an error if the summarization fails
        due to an invalid prompt path.

        Parameters:
        -----------

            - retrieved_edges (list): List of graph edges retrieved for summarization.

        Returns:
        --------

            - str: A summary string representing the content of the retrieved edges.
        """
        direct_text = await super().resolve_edges_to_text(retrieved_edges)
        return await summarize_text(direct_text, self.summarize_prompt_path, self.system_prompt)
