from typing import List, Optional
import asyncio

from cognee.infrastructure.context.BaseContextProvider import BaseContextProvider
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
    format_triplets,
    get_memory_fragment,
)
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User


class TripletSearchContextProvider(BaseContextProvider):
    """Context provider that uses brute force triplet search for each entity."""

    def __init__(
        self,
        top_k: int = 3,
        collections: List[str] = None,
        properties_to_project: List[str] = None,
    ):
        self.top_k = top_k
        self.collections = collections
        self.properties_to_project = properties_to_project

    def _get_entity_text(self, entity: DataPoint) -> Optional[str]:
        """Concatenates available entity text fields with graceful fallback."""
        texts = []
        if hasattr(entity, "name") and entity.name:
            texts.append(entity.name)
        if hasattr(entity, "description") and entity.description:
            texts.append(entity.description)
        if hasattr(entity, "text") and entity.text:
            texts.append(entity.text)

        return " ".join(texts) if texts else None

    def _get_search_tasks(
        self,
        entities: List[DataPoint],
        query: str,
        memory_fragment: CogneeGraph,
    ) -> List:
        """Creates search tasks for valid entities."""
        tasks = [
            brute_force_triplet_search(
                query=f"{entity_text} {query}",
                top_k=self.top_k,
                collections=self.collections,
                properties_to_project=self.properties_to_project,
                memory_fragment=memory_fragment,
            )
            for entity in entities
            if (entity_text := self._get_entity_text(entity)) is not None
        ]
        return tasks

    async def _format_triplets(self, triplets: List, entity_name: str) -> str:
        """Format triplets into readable text."""
        direct_text = format_triplets(triplets)
        return f"Context for {entity_name}:\n{direct_text}\n---\n"

    async def _results_to_context(self, entities: List[DataPoint], results: List) -> str:
        """Formats search results into context string."""
        triplets = []

        for entity, entity_triplets in zip(entities, results):
            entity_name = (
                getattr(entity, "name", None)
                or getattr(entity, "description", None)
                or getattr(entity, "text", str(entity))
            )
            triplets.append(await self._format_triplets(entity_triplets, entity_name))

        return "\n".join(triplets) if triplets else "No relevant context found."

    async def get_context(self, entities: List[DataPoint], query: str) -> str:
        """Get context for each entity using brute force triplet search."""
        if not entities:
            return "No entities provided for context search."

        memory_fragment = await get_memory_fragment(self.properties_to_project)
        search_tasks = self._get_search_tasks(entities, query, memory_fragment)

        if not search_tasks:
            return "No valid entities found for context search."

        results = await asyncio.gather(*search_tasks)
        return await self._results_to_context(entities, results)
