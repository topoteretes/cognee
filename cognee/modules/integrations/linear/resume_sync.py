"""Continue Linear syncs that stopped early on the rate limit or the request budget.

``linear_source`` stops cleanly when Linear's hourly quota is nearly spent, and
keeps the cursor it reached. Nothing else would start the rest: a team without
webhook traffic would stay half-ingested. This worker looks, on a timer, for
connections whose last run reported ``failed_rate_limit`` or ``failed_budget``
and starts a full sync for them once the quota has had time to refill.

A full pass that was cut short, or a first one that failed outright before the
connection was seeded, writes ``resume_needed_at`` (see ``sync``); a
webhook's partial run does not touch it, which is why it is not read from
``last_sync_counts``. A run cut short by anything else (an auth failure, a team
that is gone, an ingestion error) is not retried here: it would fail the same
way every tick.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.linear.linear_settings import linear_settings
from cognee.modules.integrations.linear.sync import PROVIDER, RESUME_KEY, request_sync
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

# Linear's quota is a leaky bucket that refills continuously; this is long
# enough for a useful slice of it to come back before the next run starts.
MIN_AGE = timedelta(minutes=10)
# Let API startup migrations finish before the first credential read.
_STARTUP_DELAY_SECONDS = 60


def is_resumable(credential: IntegrationCredential, now: datetime) -> bool:
    raw = (credential.provider_metadata or {}).get(RESUME_KEY)
    if not raw:
        return False
    try:
        cut_at = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return False
    if cut_at.tzinfo is None:
        cut_at = cut_at.replace(tzinfo=timezone.utc)
    return now - cut_at >= MIN_AGE


async def _resume(credential: IntegrationCredential) -> bool:
    try:
        return await request_sync(credential)
    except asyncio.CancelledError:
        raise
    except Exception:  # one connection must not stop the others
        logger.exception(
            "Resuming the Linear sync of organization %s failed", credential.provider_account_id
        )
        return False


async def resume_cut_short_syncs() -> int:
    """Start a full sync for every connection that is due one. Returns how many started.

    The syncs run side by side: one workspace's long first sync must not hold up
    the others. A request that finds a sync already running is dropped and not
    counted.
    """
    async with get_relational_engine().get_async_session() as db:
        credentials = (
            (
                await db.execute(
                    select(IntegrationCredential).where(
                        IntegrationCredential.provider == PROVIDER,
                        IntegrationCredential.status == "active",
                    )
                )
            )
            .scalars()
            .all()
        )
    now = datetime.now(timezone.utc)
    due = [credential for credential in credentials if is_resumable(credential, now)]
    results = await asyncio.gather(*(_resume(credential) for credential in due))
    return sum(1 for started in results if started)


async def _worker() -> None:
    await asyncio.sleep(_STARTUP_DELAY_SECONDS)
    while True:
        try:
            await resume_cut_short_syncs()
        except asyncio.CancelledError:
            raise
        except Exception:  # keep the scheduler alive across transient failures
            logger.exception("Linear resume tick failed; will retry")
        await asyncio.sleep(max(60, linear_settings.resume_interval_seconds))


@asynccontextmanager
async def linear_resume_lifespan(app):
    task = None
    if linear_settings.client_id and linear_settings.resume_sync_enabled:
        task = asyncio.create_task(_worker())
    try:
        yield
    finally:
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
