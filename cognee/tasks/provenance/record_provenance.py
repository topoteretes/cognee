"""Opt-in writer of the audit ledger: one row per data item.

Storage seam: ``add_data_points`` calls ``record_provenance_at_storage`` after
its graph/vector writes succeeded, when the ``provenance_tracking``
CognifyConfig flag is on (env ``PROVENANCE_TRACKING``, default off). Hooking
the one place every DataPoint batch passes through, not a cognify-only
pipeline task, is what makes the ledger cover ``improve()`` enrichment, the
code-graph routes, skills and custom pipelines alike.

What is recorded is the data item the batch was stored for, never the nodes
and edges it produced: one row, ``"{dataset_id}:data_item:{data_id}"``, with
the item's content hash, who stored it, in which pipeline run, and the
existing source-ref key. Which node or edge came from which data item is
already in the graph (its source refs), so the ledger does not repeat it, and
a row per node and edge costs about ten times the source text in the
always-shared relational database.

The row follows the item:

- a batch stored for an item whose row is live with the same content hash is
  a no-op, so every batch may record it;
- a different content hash versions the row (an update);
- deleting the item tombstones it (``cognee/modules/provenance/tombstones.py``),
  once its ``Data`` row is gone, and a later write brings it back; dropping
  only its memory (``forget(memory_only=True)``) leaves the row as it is.

``record_provenance`` is the same writer as a pipeline ``Task`` for custom
pipelines that store data points some other way; it records unconditionally
(the flag is the caller's call). A batch with no dataset or no data item in
its ``ctx`` has nothing to record.

The writer returns its input unchanged. It raises when the ledger cannot be
written: a batch that is stored but not recorded leaves a gap nothing reports,
so with tracking on the storing task fails and its run is rolled back.
"""

from cognee.infrastructure.databases.provenance import data_item_id, make_source_ref_key
from cognee.infrastructure.engine import DataPoint
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.modules.provenance import get_provenance_manager
from cognee.modules.provenance.tombstones import agent_id_for, ledger_data_item_key
from cognee.shared.logging_utils import get_logger

logger = get_logger("record_provenance")


def provenance_tracking_enabled() -> bool:
    """The ``PROVENANCE_TRACKING`` switch, read fresh (settings are cached)."""
    return bool(get_cognify_config().provenance_tracking)


async def record_provenance_at_storage(
    data_points: list, ctx: PipelineContext | None = None
) -> list:
    """The storage-seam hook ``add_data_points`` calls after a successful write.

    Records only when the flag is on; returns the input unchanged either way.
    A ledger write that fails raises.
    """
    if not provenance_tracking_enabled():
        return data_points
    return await record_provenance(data_points, ctx=ctx)


async def record_data_item_content(ctx: PipelineContext | None, content_hash: str | None) -> None:
    """Record that the data item in ``ctx`` now has ``content_hash``, when the flag is on.

    For a writer that changes an item's content outside ``add_data_points``:
    the incremental update stores its new chunks under the old ``Data`` row
    and only then publishes the new content.
    """
    if provenance_tracking_enabled():
        await _write_data_item(ctx, content_hash)


@task_summary("Recorded provenance for {n} item(s)")
async def record_provenance(
    data_points: list[DataPoint], ctx: PipelineContext | None = None
) -> list[DataPoint]:
    """Record the data item this batch was stored for.

    Receives the batch ``add_data_points`` stored and returns it unchanged, so
    it can also be appended to any pipeline as a ``Task``. The batch itself is
    not walked: the ledger holds the data item, not what it produced.
    """
    if not isinstance(data_points, list) or not data_points:
        return data_points
    data_item = getattr(ctx, "data_item", None) if ctx is not None else None
    await _write_data_item(ctx, getattr(data_item, "content_hash", None))
    return data_points


async def _write_data_item(ctx: PipelineContext | None, content_hash: str | None) -> None:
    """Queue and commit the row of the data item ``ctx`` names; nothing without one."""
    scope = _scope_from_ctx(ctx)
    data_id = data_item_id(getattr(ctx, "data_item", None)) if ctx is not None else None
    if not scope or not data_id:
        logger.debug("Ledger recording skipped: batch names no dataset or no data item")
        return

    run_id = str(ctx.pipeline_run_id) if ctx.pipeline_run_id else None
    batch = get_provenance_manager().batch()
    batch.track_data_item(
        ledger_data_item_key(scope, data_id),
        content_hash=content_hash,
        activity_id=f"cognify:{run_id}" if run_id else "cognify",
        agent_id=_agent_from_ctx(ctx),
        bundle_id=run_id,
        source_ref_key=_source_ref_key_from_ctx(ctx),
    )
    await batch.commit()


def _agent_from_ctx(ctx: PipelineContext | None) -> str:
    """The acting user's id, else ``"cognee"``."""
    return agent_id_for(getattr(ctx, "user", None))


def _source_ref_key_from_ctx(ctx: PipelineContext | None) -> str | None:
    """Reuse the graph-provenance source-ref key; None when either id is missing."""
    if ctx is None:
        return None
    dataset = getattr(ctx, "dataset", None)
    dataset_id = getattr(dataset, "id", None) if dataset is not None else None
    data_id = data_item_id(getattr(ctx, "data_item", None))
    if dataset_id and data_id:
        return make_source_ref_key(dataset_id, data_id)
    return None


def _scope_from_ctx(ctx: PipelineContext | None) -> str | None:
    """Dataset id as the ledger namespace (per-tenant/dataset isolation)."""
    dataset = getattr(ctx, "dataset", None) if ctx is not None else None
    dataset_id = getattr(dataset, "id", None) if dataset is not None else None
    return str(dataset_id) if dataset_id else None
