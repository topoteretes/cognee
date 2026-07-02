"""Envelope returned from the top-level search entrypoints.

Historically wrapped the retriever output as ``search_result: Any``,
which forced every consumer to guess at the payload shape. Pillar A of
issue #3604 upgrades the envelope so an agent can rank items by
normalized ``relevance``, cite them via structured
:class:`Citation` objects, and branch on a coarse
:class:`Confidence` signal, without changing any of the existing
fields.

``search_result: Any`` is preserved intact for backward compatibility;
new callers should prefer ``items`` and ``confidence``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cognee.modules.retrieval.utils.confidence import Confidence

if TYPE_CHECKING:
    # Imported for the type checker only. At runtime pydantic resolves
    # the forward reference below via ``model_rebuild`` so we avoid a
    # circular import through ``cognee.modules.search.types.__init__``
    # which is loaded transitively by ``SearchResultItem``.
    from cognee.modules.recall.types.SearchResultItem import SearchResultItem


class SearchResultDataset(BaseModel):
    id: UUID
    name: str


class SearchResult(BaseModel):
    """Search response wrapping either a legacy blob or the structured envelope.

    The legacy ``search_result`` field is preserved so existing
    callers, tests, and API consumers stay on their current path. New
    agent-facing consumers should read from ``items`` and
    ``confidence``, which are populated by retrievers as they migrate
    onto Pillar A.
    """

    model_config = ConfigDict(use_enum_values=True)

    search_result: Any
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]

    # Structured, ranked view over the same recall. A retriever that
    # hasn't been migrated yet returns an empty list; consumers can
    # treat empty-items as "fall back to search_result" without
    # branching on nulls.
    items: List["SearchResultItem"] = Field(default_factory=list)

    # Coarse confidence label derived from the top-k relevance
    # distribution of ``items``. ``None`` when the retriever did not
    # populate ``items`` at all; ``Confidence.ABSTAIN`` when it did but
    # the signal is weak enough that an agent should decline to answer.
    confidence: Optional[Confidence] = None


def _rebuild_search_result() -> None:
    """Resolve the ``SearchResultItem`` forward reference on ``SearchResult``.

    Pydantic v2 defers forward-reference resolution until a model is
    first validated. Callers that construct ``SearchResult`` after
    ``SearchResultItem`` is already importable get this for free. Kept
    exported so integration tests and startup checks can trigger the
    rebuild eagerly.
    """
    from cognee.modules.recall.types.SearchResultItem import SearchResultItem

    SearchResult.model_rebuild(_types_namespace={"SearchResultItem": SearchResultItem})
