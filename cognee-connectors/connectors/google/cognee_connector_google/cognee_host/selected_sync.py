"""Ingest a whole selection through one of this family's own sources.

The body of ``cognee.modules.integrations.ingestion.sync_selected_resources`` (cognee
dev, 2026-10-28), with the source and its service passed in. That function looks
the source up in cognee's registry, which on cognee 1.6.x names cognee's bundled
copy (``cognee.tasks.ingestion.connectors.<provider>``), so a host sync through it
would never run a fix released in this family's packages, and the package's typed
errors would not match what the bundled source raises. Gmail and Google Calendar
call this; Google Drive, which syncs per folder, hands its own source to
``ingestion.sync_scopes`` instead.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from cognee.modules.integrations import ingestion


async def sync_selected_resources(
    provider: str,
    credential: Any,
    counts: dict[str, int],
    *,
    make_source: Callable[..., Any],
    build_service: Callable[[str], Any],
    selection: list[str] | None,
    selection_kwarg: str,
    dataset_name: str,
    access_token_for: Callable[[Any], Awaitable[str]],
) -> tuple[str, dict[str, int]]:
    """Ingest the connection's whole selection through one source and return the outcome.

    ``make_source`` takes the selection as the ``selection_kwarg`` keyword
    (``label_ids``, ``calendar_ids``) and keeps its own per-resource state, so a
    changed selection needs no extra resource here. An empty selection ingests
    nothing and retires everything the account synced. A failed ingestion keeps the
    old tables and marks the sync degraded; the deselected ones are retired only
    after a clean run. The caller checks the shape of ``selection``.
    """
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.users.methods import get_user

    if selection == []:
        await ingestion.retire_resources(provider, credential, dataset_name, set())
        return "ok", {"scanned": 0, "skipped": 0, "failed": 0}

    service = build_service(await access_token_for(credential))
    owner = await get_user(credential.user_id)
    name = ingestion.resource_name(provider, credential)
    progress, report = _progress(credential, counts)
    source = make_source(
        **{selection_kwarg: selection},
        service=service,
        resource_name=name,
        check_active=progress.checkpoint
        if progress
        else ingestion.extraction_checkpoint(credential),
    )
    if progress is not None:
        progress.source = source
    await ingestion.require_active_credential(credential)
    await report(phase="processing", counts=counts)
    try:
        result = await remember(
            source,
            dataset_name=dataset_name,
            user=owner,
            write_disposition="merge",
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )
    finally:
        ingestion.add_source_counts(counts, source)
        await report(counts=counts)
    if getattr(result, "status", None) == "errored":
        counts["failed"] += 1
        counts["failed_ingestion"] = 1
    else:
        await ingestion.retire_resources(provider, credential, dataset_name, {name})
    return ("degraded" if counts["failed"] else "ok"), counts


def _progress(credential: Any, counts: dict[str, int]):
    """The host's sync-progress bridge and reporter, or no-ops on a host without them.

    cognee added live sync progress after 1.6.5: there, the bridge publishes the
    source's counters at every extraction checkpoint. On a host without it the
    source gets the plain credential checkpoint, as cognee 1.6.5 itself does.
    """
    source_progress = getattr(ingestion, "_SourceProgress", None)
    try:
        from cognee.modules.integrations.sync_progress import report_sync_progress
    except ImportError:
        report_sync_progress = None

    if source_progress is None or report_sync_progress is None:

        async def report(**_kwargs: Any) -> None:
            return None

        return None, report
    return source_progress(credential, counts), report_sync_progress
