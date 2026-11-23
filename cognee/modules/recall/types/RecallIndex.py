"""Shapes of recall's index mode and expand-by-id (``result_mode="index"``, ``expand_items``).

An index row is a ``ResponseGraphEntry`` whose ``metadata`` follows ``IndexRowMetadata``.
``SearchResultItem.kind`` keeps its usual meaning; the row's own kind is
``metadata["record_kind"]``.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

# Rows that name a stored memory and can be expanded by id.
MemoryRecordKind = Literal["passage", "entity", "fact"]
# Rows for the parts of the context and prompt that have no id; returned in full.
FrameRecordKind = Literal[
    "session_history",
    "template_prefix",
    "grounding",
    "global_context",
    "template_suffix",
    "session_guidance",
]
RecordKind = MemoryRecordKind | FrameRecordKind
ExpandStatus = Literal["available", "missing", "forbidden"]


class ExpandItem(BaseModel):
    """One record to expand: an index row's dataset, kind and id, sent back as given."""

    dataset_id: UUID
    record_kind: MemoryRecordKind
    record_id: str


class RecordSource(BaseModel):
    """A document a record was extracted from."""

    data_id: str
    document_name: str | None = None
    chunk_id: str | None = None
    created_at: str | None = None


class IndexRowMetadata(BaseModel):
    """``metadata`` of an index or expand row.

    Memory rows (passage, entity, fact) carry ``record_id``, ``rank`` (order within
    the kind, from 1), ``version`` (a content hash of the expanded body) and ``chars``
    (the expanded body's length; when it equals ``len(text)`` the index text is the
    whole body). ``score`` is comparable only within one kind: the fused rank score for
    passages (higher is better), the vector distance for entities and facts (lower is
    better). ``distance`` is the passage's raw query distance, for merging passages
    across datasets. ``sources`` is ``None`` for facts, whose sources are not tracked.
    Frame rows carry only ``record_kind`` and ``chars``. ``status`` is set on expand rows.
    """

    record_kind: RecordKind
    record_id: str | None = None
    rank: int | None = None
    score: float | None = None
    distance: float | None = None
    chars: int
    version: str | None = None
    conflicts: list[str] = []
    sources: list[RecordSource] | None = None
    status: ExpandStatus | None = None
