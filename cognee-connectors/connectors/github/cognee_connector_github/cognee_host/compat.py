"""Shims for cognee versions the ``cognee`` extra accepts but that predate a seam the moved host code uses.

``cognee.modules.integrations.sync_progress`` (``track_sync`` and
``report_sync_progress``, which the GitHub sync reports through) arrived after
cognee 1.6.6. On an older host the sync runs exactly as it did there — minting
a token, listing repositories, running the code-graph pipeline — with every
progress report dropped. :mod:`cognee_connector_github.cognee_host.sync`
imports these names only when cognee's own are not importable.
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


__all__ = ["NoopSyncProgressReporter", "report_sync_progress", "track_sync"]
