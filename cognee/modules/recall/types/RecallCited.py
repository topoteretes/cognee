"""Shapes of recall's cited mode and expand-by-id (``result_mode="cited"``, ``expand_items``).

A cited recall returns one item per dataset, shaped like a regular ``only_context`` item:
``text`` is the same user prompt with a tag line, ``[mem:<kind>/<dataset_id>/<record_id>@
<version>]``, directly before every passage, entity and fact, and ``metadata["records"]``
lists one ``CitedRecord`` per tag, in prompt order. Section headers, the conflict block and
the prompt's frame (template, history, grounding, global context, guidance) carry no tag:
they are not memories. Removing every tag line gives exactly the regular prompt.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from cognee.api.DTO import InDTO

# Records a tag names and expand reads by id.
MemoryRecordKind = Literal["passage", "entity", "fact"]
ExpandStatus = Literal["available", "missing", "forbidden"]


class ExpandItem(InDTO):
    """One record to expand: the kind, dataset and id its tag names.

    A request body model, so its keys may be camelCase or snake_case like every other
    request field. Every record id is a UUID; a malformed one is refused here instead of
    reaching the vector store.
    """

    dataset_id: UUID
    record_kind: MemoryRecordKind
    record_id: UUID


class RecordSource(BaseModel):
    """A document a record was extracted from."""

    data_id: str
    document_name: str | None = None
    chunk_id: str | None = None
    created_at: str | None = None


class CitedRecord(BaseModel):
    """One tagged record of a cited recall, or one record an expand returned.

    ``version`` is the content hash of the record's body without its tag and ``chars``
    that body's length, so a cited record and its expanded body agree. ``rank`` is the
    order within the kind, from 1. ``score`` is comparable only within one kind: the
    fused rank score for passages (higher is better), the vector distance for entities
    and facts (lower is better). ``distance`` is the passage's raw query distance, for
    merging passages across datasets. ``sources`` is ``None`` for facts, whose sources
    are not tracked. ``status`` is set on expand results only.
    """

    tag: str
    record_kind: MemoryRecordKind
    dataset_id: str
    record_id: str
    version: str | None = None
    chars: int
    rank: int | None = None
    score: float | None = None
    distance: float | None = None
    conflicts: list[str] = []
    sources: list[RecordSource] | None = None
    status: ExpandStatus | None = None
