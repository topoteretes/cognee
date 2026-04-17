"""
Field role annotations for DataPoint.

These make implicit DataPoint behaviors explicit and visible at definition time.
They are informational markers — the system also supports the legacy
metadata = {"index_fields": [...]} approach.

Usage:
    from typing import Annotated
    from cognee.infrastructure.engine.models import DataPoint, Embeddable, LLMContext, Dedup

    class Entity(DataPoint):
        name: Annotated[str, Embeddable("Primary search field"), Dedup()]
        description: Annotated[str, LLMContext("Provides entity context to LLM")]
        is_a: Optional[EntityType] = None  # Relationship (auto-detected from type)
"""


class _Embeddable:
    """Marker: this field will be embedded in the vector database."""

    def __init__(self, description: str = ""):
        self.description = description

    def __repr__(self):
        return f"Embeddable({self.description!r})" if self.description else "Embeddable()"


class _LLMContext:
    """Marker: this field is sent to the LLM during retrieval/extraction."""

    def __init__(self, description: str = ""):
        self.description = description

    def __repr__(self):
        return f"LLMContext({self.description!r})" if self.description else "LLMContext()"


class _Dedup:
    """Marker: this field is used for entity deduplication (UUID5 key)."""

    def __init__(self, description: str = ""):
        self.description = description

    def __repr__(self):
        return f"Dedup({self.description!r})" if self.description else "Dedup()"


def Embeddable(description: str = "Embedded in vector DB for semantic search"):
    """Mark a field as embedded in the vector database.

    Fields marked Embeddable are included in metadata["index_fields"]
    and will be vectorized by the embedding engine.

    Example:
        class Entity(DataPoint):
            name: Annotated[str, Embeddable()] = ""
    """
    return _Embeddable(description)


def LLMContext(description: str = "Sent to LLM during retrieval"):
    """Mark a field as sent to the LLM during context building.

    Fields marked LLMContext are concatenated and used as context
    when building search results or enriching entities.

    Example:
        class Entity(DataPoint):
            description: Annotated[str, LLMContext()] = ""
    """
    return _LLMContext(description)


def Dedup(description: str = "Used for entity deduplication"):
    """Mark a field as part of the deduplication key.

    Fields marked Dedup contribute to the identity_fields list,
    which drives deterministic UUID5 generation for entity deduplication.

    Example:
        class Entity(DataPoint):
            name: Annotated[str, Dedup()] = ""
    """
    return _Dedup(description)
