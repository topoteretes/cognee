"""Opt-in Slack source refresh. Configuration/status use the credential store.

No messages or derived memory are stored here. A fixed backfill start is
retained for recovery; each successful refresh reconciles the selected source
through the same native ingestion entry point as a manual import.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.credentials import update_provider_metadata
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.slack.history import authorize_import, import_slack_history
from cognee.modules.integrations.slack.history_models import SlackHistoryRequest, SlackSyncSettings
from cognee.modules.integrations.slack.persistence import get_by_team, is_active
from cognee.modules.integrations.slack.slack_settings import slack_settings
from cognee.modules.users.methods import get_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("slack_history")
_metadata_lock = asyncio.Lock()
_pending_imports: set[asyncio.Task] = set()
_running: set[tuple[str, str]] = set()


def background_import(coroutine):
    task = asyncio.create_task(coroutine)
    _pending_imports.add(task)
    task.add_done_callback(_pending_imports.discard)


async def _patch_entry(team_id: str, section: str, key: str, value: dict, *, user_id):
    async with _metadata_lock:
        credential = await get_by_team(team_id)
        if not is_active(credential) or credential.user_id != user_id:
            return
        entries = dict((credential.provider_metadata or {}).get(section) or {})
        entries[key] = value
        await update_provider_metadata("slack", team_id, {section: entries})


async def configure_slack_sync(team_id, settings: SlackSyncSettings, *, user):
    """Set a refresh selection per destination dataset, without changing its ACLs."""
    selection = settings.selection
    if selection is None:
        raise ValueError("Provide the selection identifying the dataset to configure or disable.")
    if settings.enabled:
        await authorize_import(team_id, user, selection)
    else:
        credential = await get_by_team(team_id)
        if not is_active(credential) or credential.user_id != user.id:
            from cognee.modules.integrations.slack.history_models import SlackHistoryError

            raise SlackHistoryError("Only the Slack connection owner can disable its sync.")
    oldest, _ = selection.bounds()
    # Resolve 'last N days' once. Moving that boundary on every run would
    # miss late replies and conflate source selection with retention.
    fixed = selection.model_copy(update={"days": None, "oldest": oldest})
    saved = settings.model_copy(update={"selection": fixed})
    await _patch_entry(
        team_id,
        "history_sync",
        str(selection.dataset_id),
        saved.model_dump(mode="json"),
        user_id=user.id,
    )
    return saved


async def run_history_import(team_id, selection, *, user, reconcile=False):
    key = str(selection.dataset_id)
    await authorize_import(team_id, user, selection)
    if (team_id, key) in _running:
        from cognee.modules.integrations.slack.history_models import SlackHistoryError

        raise SlackHistoryError("An import for this workspace and dataset is already running.")
    _running.add((team_id, key))
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        await _patch_entry(team_id, "history_reports", key, report, user_id=user.id)
        result = await import_slack_history(team_id, selection, user=user, reconcile=reconcile)
        await _patch_entry(
            team_id,
            "history_reports",
            key,
            {
                **report,
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": result.model_dump(mode="json"),
            },
            user_id=user.id,
        )
        return result
    except BaseException:
        # Persist no raw exception (provider errors may contain sensitive data).
        await _patch_entry(
            team_id,
            "history_reports",
            key,
            {
                **report,
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": "Import did not complete. Retry the selection; check the server logs.",
            },
            user_id=user.id,
        )
        raise
    finally:
        _running.discard((team_id, key))


async def sync_due_slack_history():
    """One scheduler tick; can also be called by an external scheduler."""
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        credentials = (
            (
                await db.execute(
                    select(IntegrationCredential).where(
                        IntegrationCredential.provider == "slack",
                        IntegrationCredential.status == "active",
                    )
                )
            )
            .scalars()
            .all()
        )
    now = datetime.now(timezone.utc)
    for credential in credentials:
        metadata = credential.provider_metadata or {}
        for key, raw in (metadata.get("history_sync") or {}).items():
            try:
                settings = SlackSyncSettings.model_validate(raw)
                if not settings.enabled:
                    continue
                report = (metadata.get("history_reports") or {}).get(key) or {}
                stamp = report.get("finished_at") or report.get("started_at")
                interval = settings.interval_seconds if report.get("status") == "completed" else 600
                if stamp and (now - datetime.fromisoformat(stamp)).total_seconds() < interval:
                    continue
                user = await get_user(credential.user_id)
                await run_history_import(
                    credential.provider_account_id,
                    settings.selection,
                    user=user,
                    reconcile=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one connection must not stop other scheduled refreshes
                # Raw exception messages/content are intentionally not logged.
                logger.error(
                    "Slack history refresh failed for connection %s, dataset %s", credential.id, key
                )


async def _worker():
    while True:
        # Let API startup migrations finish before the first credential read.
        await asyncio.sleep(60)
        try:
            await sync_due_slack_history()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - keep the optional scheduler alive across transient failures
            logger.error("Slack history scheduler tick failed; will retry")


@asynccontextmanager
async def slack_history_lifespan(app):
    task = None
    if slack_settings.client_id and slack_settings.history_sync_enabled:
        task = asyncio.create_task(_worker())
    try:
        yield
    finally:
        tasks = list(_pending_imports)
        if task:
            tasks.append(task)
        for pending in tasks:
            pending.cancel()
        for pending in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await pending
