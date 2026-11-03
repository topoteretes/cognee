"""Shims for cognee versions the ``cognee`` extra accepts but that predate a seam the moved host code uses.

``cognee.modules.integrations.sync_progress`` (``track_sync``,
``report_sync_progress`` and ``get_current_sync_progress``, which the history
import reports through) arrived after cognee 1.6.6. On an older host the import
runs exactly as it did there — authorizing the selection, fetching every page,
writing the documents and indexing them — with every progress report dropped,
and a rate-limit wait finds no current reporter whose state it would restore.
:mod:`cognee_connector_slack.cognee_host.history` and
:mod:`cognee_connector_slack.cognee_host.history_client` import these names only
when cognee's own are not importable. The same shape as the GitHub connector's
``cognee_host/compat.py``.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class NoopSyncProgressReporter:
    """Stands in for cognee's ``SyncProgressReporter``: accepts every report, keeps nothing."""

    async def report(self, **_: Any) -> None:
        return

    async def finish(
        self,
        state: str = "succeeded",
        *,
        counts: dict[str, int] | None = None,
        reason: str | None = None,
    ) -> None:
        return


@asynccontextmanager
async def track_sync(
    provider: str, credential: Any, **_: Any
) -> AsyncIterator[NoopSyncProgressReporter]:
    """``cognee.modules.integrations.sync_progress.track_sync`` with no progress store."""
    yield NoopSyncProgressReporter()


async def report_sync_progress(**_: Any) -> None:
    """``cognee.modules.integrations.sync_progress.report_sync_progress``, dropped."""
    return


def get_current_sync_progress() -> None:
    """``cognee.modules.integrations.sync_progress.get_current_sync_progress``: no reporter."""
    return


__all__ = [
    "NoopSyncProgressReporter",
    "get_current_sync_progress",
    "report_sync_progress",
    "track_sync",
]
