"""Scheduled sync on the SDK API server: whose a connection is, how a due one starts,
and the opt-in worker that ticks (SDK-1434).

Run the API that enables it as one process per database (one worker, one replica):
runs of one connection are kept apart only in-process. docs/integration-sync-schedule.md
says what breaks with more.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import and_, or_, select

from cognee.infrastructure.background_tasks import register_background_task
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.background import spawn_background
from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.registry import get_integration
from cognee.modules.integrations.sync_policy import run_due_syncs
from cognee.modules.operations import ORIGIN_BACKGROUND, operation_origin_scope
from cognee.shared.logging_utils import get_logger

logger = get_logger("scheduled_sync")

# A slot starts at most one tick, plus that tick's run time, late; slots are 15+ minutes apart.
TICK_SECONDS = 60


class ScheduledSyncSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INTEGRATIONS_", extra="ignore")

    # Env: INTEGRATIONS_SYNC_SCHEDULER_ENABLED
    sync_scheduler_enabled: bool = False


def owner_principal_id(credential: IntegrationCredential) -> UUID:
    """``workspace_id`` when set, else ``user_id``: the owner ``upsert_credential`` keys on."""
    return credential.workspace_id or credential.user_id


async def owned_credential_ids(principal_id: UUID) -> list[UUID]:
    """The active connections ``owner_principal_id`` assigns to ``principal_id``."""
    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(IntegrationCredential.id).where(
                IntegrationCredential.status == STATUS_ACTIVE,
                or_(
                    IntegrationCredential.workspace_id == principal_id,
                    and_(
                        IntegrationCredential.workspace_id.is_(None),
                        IntegrationCredential.user_id == principal_id,
                    ),
                ),
            )
        )
        return [row.id for row in rows]


async def _sync(credential_id: UUID) -> None:
    async with get_relational_engine().get_async_session() as session:
        credential = await session.get(IntegrationCredential, credential_id)
    if credential.status != STATUS_ACTIVE:
        # Disconnected after the tick listed it: nothing to sync.
        return
    try:
        integration = get_integration(credential.provider)
    except KeyError:
        logger.warning(
            "Connection %s is not synced: no integration is registered for %s",
            credential_id,
            credential.provider,
        )
        return
    await integration.sync_now(credential)


async def start_sync(credential_id: UUID) -> None:
    """Start the connection's ``sync_now`` in the background, as the manual sync route does.

    The credential is read inside the task too, so nothing here raises: a failure is
    logged with the connection, and the tick starts the others.
    """
    spawn_background(
        _sync(credential_id), description=f"Scheduled sync of connection {credential_id}"
    )


async def _tick_until(stop: asyncio.Event) -> None:
    while True:
        # The first tick comes one interval after startup.
        try:
            await asyncio.wait_for(stop.wait(), timeout=TICK_SECONDS)
            logger.info("Scheduled sync worker stopped")
            return
        except asyncio.TimeoutError:
            pass
        try:
            # Recorded as background work; the syncs it starts inherit the scope.
            with operation_origin_scope(ORIGIN_BACKGROUND):
                await run_due_syncs(datetime.now(timezone.utc), owned_credential_ids, start_sync)
        except Exception:
            # No caller receives the worker's errors, and a database restart raises driver
            # errors SQLAlchemy does not wrap (asyncpg's CannotConnectNowError, a refused
            # connection), so a failed tick is logged and the next one tries again.
            logger.exception("Scheduled sync tick failed; retrying in %s s", TICK_SECONDS)


@asynccontextmanager
async def sync_scheduler_lifespan(app):
    if not ScheduledSyncSettings().sync_scheduler_enabled:
        yield
        return
    stop = asyncio.Event()
    # Registered, so the app's bounded shutdown drain waits for a tick in flight: it
    # is not cut between a claim and its starts, and a database that hangs cannot
    # hold shutdown past the drain's budget.
    register_background_task(asyncio.create_task(_tick_until(stop)))
    try:
        yield
    finally:
        stop.set()
