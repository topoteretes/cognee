"""The result of ``update()``: one shape for every path.

``update()`` used to answer in two incompatible shapes — a summary dict from the
chunk-level path and a pipeline-run mapping from the full rebuild — so a caller
could not tell from the value alone whether the document was refreshed cheaply,
rebuilt from scratch, or left as it was, and the reason for a rebuild lived only
in the server log.

The result is a dict on every path, and a superset of the chunk-level summary
that shipped before: the same keys with the same values, plus the fields below.
This model is its schema — the HTTP route validates and documents the body
with it, and the remote client normalizes what it receives through it.

A batch (a list of inputs, or a directory) answers with ``UpdateBatchResult``:
the per-document results plus the counts a caller renders as
"N updated, N unchanged, N failed".
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from cognee.api.v1.update.incremental import RefusalReason


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

    ``status`` says what happened: the chunk-level path replaced chunks
    (``incremental``) or found nothing to change (``unchanged``), the whole
    document was rebuilt (``full_rebuild``), or the rebuild's cognify run
    errored (``failed``, with ``error`` naming the cause so the call can be
    retried). A rebuild always carries ``fallback``, naming why the chunk-level
    path did not run or that the caller switched it off, and
    ``duration_seconds`` makes a slow update visible next to its reason.

    The chunk counters are work done by the chunk-level path — ``added`` can
    exceed the net change when a re-cut chunk with unchanged content is
    re-extracted in place — and ``total_chunks`` is the count after the
    update. They are ``None`` on a rebuild, which has no diff. The document
    keeps its ``data_id`` on every path.
    """

    status: Literal["incremental", "unchanged", "full_rebuild", "failed"]
    regions: int | None = None
    deleted_chunks: int | None = None
    added_chunks: int | None = None
    reused_chunks: int | None = None
    kept_chunks: int | None = None
    reindexed_chunks: int | None = None
    total_chunks: int | None = None
    data_id: UUID
    dataset_id: UUID
    duration_seconds: float
    pipeline_run_id: UUID | None = None
    fallback: Fallback | None = None
    error: UpdateError | None = None


class UpdateBatchResult(BaseModel):
    """Outcome of ``update()`` over several documents.

    ``results`` holds one ``UpdateResult`` per input, in input order; a
    document whose update raised is recorded there with ``status``
    ``failed`` and the error, and the batch carries on with the rest, so a
    caller retries the failed ones by their ``data_id``. The counts summarize
    them: ``updated`` is every document whose content was replaced
    (``incremental`` or ``full_rebuild``). ``status`` is ``completed`` when
    nothing failed, ``partial`` when some documents failed and ``failed`` when
    every one did.
    """

    status: Literal["completed", "partial", "failed"]
    total: int
    updated: int
    unchanged: int
    failed: int
    dataset_id: UUID
    duration_seconds: float
    results: list[UpdateResult]

    @classmethod
    def of(
        cls, results: list[dict], dataset_id: UUID, duration_seconds: float
    ) -> "UpdateBatchResult":
        """Aggregate per-document result dicts."""
        statuses = [result["status"] for result in results]
        failed = statuses.count("failed")
        return cls(
            status="failed"
            if results and failed == len(results)
            else "partial"
            if failed
            else "completed",
            total=len(results),
            updated=statuses.count("incremental") + statuses.count("full_rebuild"),
            unchanged=statuses.count("unchanged"),
            failed=failed,
            dataset_id=dataset_id,
            duration_seconds=duration_seconds,
            results=[UpdateResult.model_validate(result) for result in results],
        )
