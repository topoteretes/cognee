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

# Longest label a context block header gets. Name-less entities (a chunk, an
# entity with only a description) would otherwise be headed by their full text.
MAX_LABEL_LENGTH = 80


def _shorten_label(text: str, limit: int = MAX_LABEL_LENGTH) -> str:
    """Collapses whitespace and cuts ``text`` to ``limit`` characters on a word boundary."""
    label = " ".join(text.split())
    if len(label) <= limit:
        return label
    cut = label[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


class SearchableEntity(NamedTuple):
    """An entity that has searchable text, with the strings derived from it."""

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

    @staticmethod
    def _entity_text_fields(entity: DataPoint) -> list[str]:
        """Returns the entity's non-blank text fields, stripped, in priority order."""
        fields = []
        for field in _TEXT_FIELDS:
            value = getattr(entity, field, None)
            if isinstance(value, str) and value.strip():
                fields.append(value.strip())
        return fields

    def _get_entity_text(self, entity: DataPoint) -> str | None:
        """Concatenates available entity text fields with graceful fallback."""
        fields = self._entity_text_fields(entity)
        return " ".join(fields) if fields else None

    def _searchable_entities(self, entities: list[DataPoint]) -> list[SearchableEntity]:
        """Keeps the entities that have searchable text.

        Each entry carries the entity together with its search text and display
        label, so every later step reads from this one list and a skipped entity
        can never shift results onto the wrong entity.
        """
        searchable = []
        for entity in entities:
            fields = self._entity_text_fields(entity)
            if not fields:
                continue
            # The label is the first text field the search text starts with, so
            # the context block is named after what was actually searched, kept
            # short so a name-less entity does not head its block with its full text.
            searchable.append(
                SearchableEntity(
                    entity=entity,
                    search_text=" ".join(fields),
                    label=_shorten_label(fields[0]),
                )
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

        succeeded: list[SearchableEntity] = []
        results: list[list] = []
        failures: list[BaseException] = []
        for item, outcome in zip(searchable, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    # Cancellation and interpreter exits are not search failures.
                    raise outcome
                logger.warning(
                    "Triplet search failed for entity %r; skipping its context: %s",
                    item.label,
                    outcome,
                )
                failures.append(outcome)
                continue
            succeeded.append(item)
            results.append(outcome)

        if failures and not succeeded:
            # Every search failed, so the failure is not entity-specific; surface it.
            raise failures[0]

        return await self._results_to_context(succeeded, results)
