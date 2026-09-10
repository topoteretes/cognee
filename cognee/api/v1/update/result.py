"""The result of ``update()``: one shape for every path.

``update()`` used to answer in two incompatible shapes — a summary dict from the
chunk-level path and a pipeline-run mapping from the full rebuild — so a caller
could not tell from the value alone whether the document was refreshed cheaply,
rebuilt from scratch, or left as it was, and the reason for a rebuild lived only
in the server log. This model answers those questions in the value itself.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from cognee.api.v1.update.incremental import RefusalReason


class ChunkChanges(BaseModel):
    """What the chunk-level path did to the document's chunks.

    ``regions`` is the number of disjoint edited spans the diff found and
    ``total`` the chunk count after the update, so "kept 29 of 30" reads off
    directly. The counts are work done: ``added`` can exceed the net change
    when a re-cut chunk with unchanged content is re-extracted in place.
    """

    regions: int = 0
    deleted: int = 0
    added: int = 0
    reused: int = 0
    kept: int = 0
    reindexed: int = 0
    total: int = 0


class Fallback(BaseModel):
    """Why the full rebuild ran instead of the chunk-level update."""

    reason: RefusalReason
    detail: str


class UpdateError(BaseModel):
    """Why the rebuild's cognify run failed."""

    error_class: str | None = None
    message: str | None = None


class UpdateResult(BaseModel):
    """Per-document outcome of ``update()``.

    ``status`` says what happened to the document; ``mode`` says how. A
    ``full_rebuild`` always carries ``fallback``, naming why the chunk-level
    path did not run (or that the caller switched it off), so an update that
    took far longer than usual explains itself; ``duration_seconds`` makes the
    "longer" visible. ``chunks`` is set only for the incremental mode: a
    rebuild re-extracts the whole document and has no diff to report.

    A ``failed`` status carries ``error`` so the caller can retry this one
    document; the document keeps its ``data_id`` on every path.
    """

    data_id: UUID
    dataset_id: UUID
    status: Literal["updated", "unchanged", "failed"]
    mode: Literal["incremental", "full_rebuild"]
    duration_seconds: float
    chunks: ChunkChanges | None = None
    fallback: Fallback | None = None
    pipeline_run_id: UUID | None = None
    error: UpdateError | None = None
