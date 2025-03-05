from typing import List, Optional

from cognee.modules.retrieval.utils.completion import summarize_text
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)


class SummarizedTripletSearchContextProvider(TripletSearchContextProvider):
    """Context provider that uses summarized triplet search results."""

    async def _format_triplets(
        self, triplets: List, entity_name: str, summarize_prompt_path: Optional[str] = None
    ) -> str:
        """Format triplets into a summarized text."""
        direct_text = await super()._format_triplets(triplets, entity_name)

        if summarize_prompt_path is None:
            summarize_prompt_path = "summarize_search_results.txt"

        summary = await summarize_text(direct_text, summarize_prompt_path)
        return f"Summary for {entity_name}:\n{summary}\n---\n"
