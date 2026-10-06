import asyncio
from collections.abc import Coroutine
from typing import Any, NamedTuple

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
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_TEXT_FIELDS = ("name", "description", "text")

# Longest context block label; a name-less entity (a chunk, say) would otherwise
# be labelled with its full text.
MAX_LABEL_LENGTH = 80


class SearchableEntity(NamedTuple):
    """An entity that has searchable text, with its search text and display label."""

    entity: DataPoint
    search_text: str
    label: str


class TripletSearchContextProvider(BaseContextProvider):
    """Context provider that uses brute force triplet search for each entity."""

    def __init__(
        self,
        top_k: int = 3,
        collections: list[str] | None = None,
        properties_to_project: list[str] | None = None,
    ):
        self.top_k = top_k
        self.collections = collections
        self.properties_to_project = properties_to_project

    def _searchable_entities(self, entities: list[DataPoint]) -> list[SearchableEntity]:
        """Keeps the entities that have non-blank text, in order.

        Searches, results and labels are all derived from this one list, so a
        skipped entity can never shift a result onto the wrong entity.
        """
        searchable = []
        for entity in entities:
            values = (getattr(entity, field, None) for field in _TEXT_FIELDS)
            fields = [v.strip() for v in values if isinstance(v, str) and v.strip()]
            if not fields:
                continue
            # Label with the field the search text starts with, on one short line.
            label = " ".join(fields[0].split())
            if len(label) > MAX_LABEL_LENGTH:
                label = label[: MAX_LABEL_LENGTH - 1] + "…"
            searchable.append(
                SearchableEntity(entity=entity, search_text=" ".join(fields), label=label)
            )
        return searchable

    def _get_search_tasks(
        self,
        searchable: list[SearchableEntity],
        query: str,
        memory_fragment: CogneeGraph,
    ) -> list[Coroutine[Any, Any, list]]:
        """Creates one search coroutine per searchable entity, in the same order."""
        return [
            brute_force_triplet_search(
                query=f"{item.search_text} {query}",
                top_k=self.top_k,
                collections=self.collections,
                properties_to_project=self.properties_to_project,
                memory_fragment=memory_fragment,
            )
            for item in searchable
        ]

    async def _format_triplets(self, triplets: list, entity_name: str) -> str:
        """Format triplets into readable text."""
        direct_text = format_triplets(triplets)
        return f"Context for {entity_name}:\n{direct_text}\n---\n"

    async def _results_to_context(
        self, searchable: list[SearchableEntity], results: list[list]
    ) -> str:
        """Formats search results into context string, one block per entity."""
        # Formatting can call an LLM (the summarized subclass), so run the
        # entities concurrently; gather keeps the blocks in entity order.
        blocks = await asyncio.gather(
            *(
                self._format_triplets(entity_triplets, item.label)
                for item, entity_triplets in zip(searchable, results, strict=True)
            )
        )
        return "\n".join(blocks) if blocks else "No relevant context found."

    async def get_context(self, entities: list[DataPoint], query: str) -> str:
        """Get context for each entity using brute force triplet search."""
        if not entities:
            return "No entities provided for context search."

        searchable = self._searchable_entities(entities)
        if not searchable:
            # Checked before the projection so a request with nothing to search
            # never pays for loading the graph.
            return "No valid entities found for context search."

        memory_fragment = await get_memory_fragment(self.properties_to_project)
        search_tasks = self._get_search_tasks(searchable, query, memory_fragment)
        outcomes = await asyncio.gather(*search_tasks, return_exceptions=True)

        # One entity's failed search must not discard the others' context.
        succeeded = []
        for item, outcome in zip(searchable, outcomes, strict=True):
            if not isinstance(outcome, BaseException):
                succeeded.append((item, outcome))
            elif isinstance(outcome, Exception):
                logger.warning("Triplet search failed for entity %r: %s", item.label, outcome)
            else:
                raise outcome  # cancellation is not a search failure

        if not succeeded:
            # Every search failed, so the cause is not entity-specific; surface it.
            raise outcomes[0]

        items, results = zip(*succeeded)
        return await self._results_to_context(list(items), list(results))
