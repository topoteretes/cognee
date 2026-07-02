"""Normalized search result envelope.

``cognee.search`` routes to retrievers that produce wildly different
payloads: plain completion strings from graph/LLM retrievers, chunk
payload dicts from CHUNKS, summary dicts from SUMMARIES, raw graph
rows from CYPHER, etc. This module defines the single wire shape that
all search paths converge on.

A ``SearchResponse`` carries the original query, the effective search
type (after FEELING_LUCKY resolution if applicable), a flat list of
``SearchResultItem``, and the total count. Each item always has a
renderable ``text`` plus the original payload preserved under ``raw``.
"""

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from cognee.modules.retrieval.utils.citation_models import Citation
from cognee.modules.search.types.SearchType import SearchType


class SearchResultKind(str, Enum):
    """Semantic kind of a search result item.

    More precise than ``search_type`` alone — tells the caller which
    normalization shape was applied. LLM completion types all collapse
    to ``*_COMPLETION`` kinds; non-LLM retrievers produce structural
    kinds (``CHUNK``, ``SUMMARY``, ``CYPHER`` row).
    """

    GRAPH_COMPLETION = "graph_completion"
    RAG_COMPLETION = "rag_completion"
    TRIPLET_COMPLETION = "triplet_completion"
    CYPHER = "cypher"
    CHUNK = "chunk"
    SUMMARY = "summary"
    CODING_RULE = "coding_rule"
    NATURAL_LANGUAGE = "natural_language"
    TEMPORAL = "temporal"
    STRUCTURED = "structured"  # when a response_model was supplied
    UNKNOWN = "unknown"


class SearchResultItem(BaseModel):
    """One normalized search hit.

    ``text`` is always populated and always renderable — callers that
    just want to display results can stop there. ``metadata`` carries
    kind-specific details (chunk_id, doc_id, score, etc.) and ``raw``
    preserves the original payload for callers that need the full
    object. ``structured`` is populated when a Pydantic ``response_model``
    was supplied to the retriever and parsing succeeded.
    """

    model_config = ConfigDict(use_enum_values=True)

    kind: SearchResultKind
    search_type: SearchType

    text: str
    score: float | None = None
    # Normalized relevance in [0, 1], higher-is-better. Populated when
    # the underlying retriever can map its raw score onto the shared
    # scale (see cognee.infrastructure.databases.vector.models.
    # ScoredResult.normalize_distance_to_relevance). Left unset when no
    # calibrated signal is available; downstream confidence derivation
    # treats "unset" and "explicitly 0" differently.
    relevance: Optional[float] = None

    dataset_id: str | None = None
    dataset_name: str | None = None

    metadata: dict[str, Any] = {}
    raw: dict[str, Any] = {}

    structured: Any | None = None

    # Structured provenance an agent can cite. Empty list means "no
    # citation surfaced" rather than "the item has no source"; a
    # retriever that doesn't emit citations yet returns [] instead of
    # unset. Keep it additive to the existing text-based Evidence
    # blocks in cognee.modules.retrieval.utils.references.
    citations: List[Citation] = Field(default_factory=list)
