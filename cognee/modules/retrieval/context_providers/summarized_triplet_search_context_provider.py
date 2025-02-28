from typing import List

from cognee.modules.retrieval.utils.completion import summarize_text
from cognee.modules.retrieval.context_providers.triplet_search_context_provider import (
    TripletSearchContextProvider,
)


class SummarizedTripletSearchContextProvider(TripletSearchContextProvider):
    """Context provider that uses summarized triplet search results."""

    def __init__(
        self,
        top_k: int = 3,
        collections: List[str] = None,
        properties_to_project: List[str] = None,
        summarize_prompt_path: str = "summarize_search_results.txt",
    ):
        super().__init__(
            top_k=top_k,
            collections=collections,
            properties_to_project=properties_to_project,
        )
        self.summarize_prompt_path = summarize_prompt_path

    async def _format_triplets(self, triplets: List, entity_name: str) -> str:
        """Format triplets into a summarized text."""
        direct_text = await super()._format_triplets(triplets, entity_name)
        summary = await summarize_text(direct_text, self.summarize_prompt_path)
        return f"Summary for {entity_name}:\n{summary}\n---\n"
