"""Shapes of recall's cited mode and expand-by-id (``result_mode="cited"``, ``expand_items``).

A cited recall returns one item per dataset, shaped like a regular ``only_context`` item:
``text`` is the same user prompt with a tag line, ``[mem:<kind>/<dataset_id>/<record_id>@
<version>]``, directly before every passage, entity and fact, and ``metadata["records"]``
lists one ``CitedRecord`` per tag, in prompt order. Section headers, the conflict block and
the prompt's frame (template, history, grounding, global context, guidance) carry no tag:
they are not memories. An entity expand could not read back the same carries none either:
an ontology concept the query names, and an entity whose edges could not be read.
Removing every tag line gives exactly the regular prompt. Only
lines that are a whole tag are tag lines, and a tag is real only when
``metadata["records"]`` lists it: a stored body may quote an earlier tag.

With access control off, a search over several datasets reads one shared store, so its
records belong to no single dataset; their tags leave the dataset out:
``[mem:<kind>/<record_id>@<version>]``.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from cognee.api.DTO import InDTO

# Records a tag names and expand reads by id.
MemoryRecordKind = Literal["passage", "entity", "fact"]
ExpandStatus = Literal["available", "missing", "forbidden"]


class ExpandItem(InDTO):
    """One record to expand: the kind, dataset and id its tag names.

    A request body model, so its keys may be camelCase or snake_case like every other
    request field. Every record id is a UUID; a malformed one is refused here instead of
    reaching the vector store. ``dataset_id`` is None for a tag without a dataset (access
    control off); ``version`` is the tag's (a content hash: 32 lowercase hex characters),
    echoed in a ``missing`` or ``forbidden`` row.
    """

    dataset_id: UUID | None = None
    record_kind: MemoryRecordKind
    record_id: UUID
    version: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class RecordSource(BaseModel):
    """A document a record was extracted from."""

    data_id: str
    document_name: str | None = None
    chunk_id: str | None = None
    created_at: str | None = None


class CitedRecord(BaseModel):
    """One tagged record of a cited recall, or one record an expand returned.

    ``version`` is the content hash of the body expand returns, so a cited record and its
    expanded body agree. ``chars`` is the length of the body the prompt shows and
    ``full_chars`` that of the expanded body. They differ only for entities: the prompt
    shows the bullets ranked against the query, expand returns every one-hop edge, sorted
    by text, so the version of an entity does not depend on the query. ``rank`` is the
    order within the kind among the cited records, from 1. ``score`` is comparable only
    within one kind: the fused rank score for passages (higher is better), the vector
    distance for entities and facts (lower is better). ``distance`` is the passage's raw
    query distance, for merging passages across datasets. ``sources`` are the documents a
    record was extracted from: a passage's document in both calls; an entity's, those of
    all its edges, on expand only (None in a cited recall); None for facts, whose sources
    are not tracked. ``dataset_id`` is None for a record of the shared store (access
    control off). ``tag`` is None on a ``missing`` or ``forbidden`` row requested without a
    version. ``status`` is set on expand results only.
    """

    tag: str | None
    record_kind: MemoryRecordKind
    dataset_id: str | None
    record_id: str
    version: str | None = None
    chars: int
    full_chars: int | None = None
    rank: int | None = None
    score: float | None = None
    distance: float | None = None
    conflicts: list[str] = []
    sources: list[RecordSource] | None = None
    status: ExpandStatus | None = None
