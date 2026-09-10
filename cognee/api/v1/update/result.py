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

    ``regions`` is the number of disjoint edited spans the diff found; the
    counts are work done, so ``added`` can exceed the net change when a re-cut
    chunk with unchanged content is re-extracted in place.
    """

    regions: int = 0
    deleted: int = 0
    added: int = 0
    reused: int = 0
    kept: int = 0
    reindexed: int = 0


class UpdateResult(BaseModel):
    """Per-document outcome of ``update()``.

    ``status`` says what happened to the document; ``mode`` says how. A
    ``full_rebuild`` carries ``fallback_reason`` whenever the chunk-level path
    was requested and could not run (or was switched off by the caller), so an
    update that took far longer than usual explains itself. ``chunks`` is set
    only for the incremental mode: a rebuild re-extracts the whole document
    and has no diff to report.

    A ``failed`` status names the error so the caller can retry this one
    document; the document keeps its ``data_id`` on every path.
    """

    data_id: UUID
    dataset_id: UUID
    status: Literal["updated", "unchanged", "failed"]
    mode: Literal["incremental", "full_rebuild"]
    chunks: ChunkChanges | None = None
    fallback_reason: RefusalReason | None = None
    fallback_detail: str | None = None
    pipeline_run_id: UUID | None = None
    error_class: str | None = None
    error_message: str | None = None
