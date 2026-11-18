"""Sync selected Slack channels through the SDK source's incremental, deletion-aware path.

Copied from cognee.modules.integrations.slack.sync (cognee dev, 2026-11-12).

Core owns OAuth, the channel selection, the authorization guards and the dataset.
``slack_source`` owns the walk, rendering, fingerprints and tombstones. One channel is
one scope, so one dlt table and one sync state; a workspace is one dataset.

Before any message is read the connection must name the Slack user who installed it and
the token must belong to the workspace (one ``auth.test`` per sync). Then each selected
channel gets a verdict (``channel_verdict``) from Slack's answer, following Slack's own
permissions, because the installer is the reader of the workspace dataset:

* synced: the app is in the channel, and for a private one the installer is too;
* archived: Slack keeps the history and the channel takes no new messages, so its data is
  kept without a walk, counted under ``channels_archived``, until it is unarchived;
* retired: the installer can no longer read the private channel; its data is removed like
  a deselected channel's;
* unavailable: the channel was deleted or the app removed from it; it is reported as a
  failure under ``failed_channel_unavailable`` and keeps its data, or is retired when the
  owner turned on ``forget_unavailable_channels``;
* an answer Slack could not give (rate limit, an unreadable member list) keeps the data,
  counts a failure and blocks retirement for this pass.

A token Slack answers with ``account_inactive`` belongs to a deleted workspace or bot user:
the connection is ended like an uninstall, and its data stays until the owner deletes it.

An in-process tick syncs every connection with a selection every six hours, unless
``SLACK_HISTORY_SYNC_ENABLED`` is false.
"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from sqlalchemy import select

# This package's source and client, not ``ingestion.build_service`` / ``source_factory``:
# on a cognee that bundles Slack those resolve to its own copy, whose errors are not the
# classes this module catches.
from cognee_connector_slack import source as slack_source_module
from cognee_connector_slack.settings import slack_settings

logger = logging.getLogger(__name__)

PROVIDER = "slack"
SLACK_DATASET_PREFIX = "slack"
SELECTION_KEY = "selected_channels"
# Written by every run (``credentials.record_sync_result``).
LAST_SYNC_COUNTS_KEY = "last_sync_counts"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"
# The bot scope that renders authors by name. Older installs lack it until they reconnect.
REQUIRED_SCOPE = "users:read"
# The former history worker's refresh interval.
SYNC_INTERVAL = timedelta(hours=6)
# The former history worker's retry after a run that did not complete. A run cut short by
# the request budget, a rate limit or a Slack outage resumes after it; any other failure
# waits the interval.
RESUME_INTERVAL = timedelta(seconds=600)
RESUMABLE_COUNT_KEYS = ("failed_budget", "failed_rate_limit", "failed_unavailable")
# How often the tick looks for a connection that is due, and how long it waits after
# startup so the API's migrations finish before the first credential read.
_TICK_SECONDS = 60
_STARTUP_DELAY_SECONDS = 60

# Slack's Developer Policy: an uninstalled app deletes the workspace's data within 14
# business days (G28). The data waits 14 days, so a workspace connected again by then
# keeps it; the key is set by ``app_uninstalled`` and cleared by a reconnect.
UNINSTALLED_AT_KEY = "uninstalled_at"
# When the uninstall's deletion ran. Kept with ``UNINSTALLED_AT_KEY`` so the owner's status
# still says when the data was deleted, and the deletion runs once; a reconnect clears
# both (G84).
UNINSTALL_DELETED_AT_KEY = "uninstall_data_deleted_at"
# Threads whose change the next sync cannot see on their root (an edited or deleted reply,
# a removed file: G6), by channel: ``{channel_id: {root_ts: stamp}}``. Written by the
# message events, read again and cleared by the sync. Every mark writes a new stamp and a
# sync clears only the stamps it read, so a change that arrives while a sync reads the
# thread stays marked for the next one (G72). A list of roots (before G72) reads as marks
# without a stamp.
CHANGED_THREADS_KEY = "changed_threads"
# Messages taken out of the workspace's memory by their author or the connection's owner
# ("Forget this", ``@cognee forget``), by channel: ``{channel_id: [ts, ...]}``. Every sync
# leaves them out of the documents they would be in, for good (Goran, 2026-11-10).
EXCLUDED_MESSAGES_KEY = "excluded_messages"
UNINSTALL_RETENTION = timedelta(days=14)
# dlt and ingestion each wrap the source's error once: one step per link, no further.
_ERROR_CHAIN_DEPTH = 5
_CHANNEL = re.compile(r"[CG][A-Z0-9]+")
# A selected channel Slack reports as deleted, or the app was removed from. Counted as a
# failure so the owner sees it, but it does not block retirement (``retirement_blocked``).
UNAVAILABLE_KEY = "failed_channel_unavailable"
# The connection no longer matches its workspace (no installer recorded, a token of another
# workspace): only a reconnect helps, and the owner's card says so (G76).
RECONNECT_KEY = "failed_reconnect"
# The owner's setting: retire unavailable channels instead of keeping their data.
FORGET_UNAVAILABLE_KEY = "forget_unavailable_channels"
# A channel's verdict before any message is read (``channel_verdict``).
SYNC, RETIRE, UNAVAILABLE, ARCHIVED = "sync", "retire", "unavailable", "archived"


# Scheduled syncs in flight; the lifespan cancels and awaits them on shutdown.
_scheduled: set[asyncio.Task] = set()


class SlackSyncError(RuntimeError):
    """A sync cannot start for this connection; the message says what to do."""


class SlackChannelAccessError(RuntimeError):
    """A selected channel fails an authorization guard. Carries no channel id."""


def dataset_name_for_credential(credential: IntegrationCredential) -> str:
    """The dataset a workspace syncs into, named from the team id alone (names change)."""
    return ingestion.dataset_name_for_account(
        SLACK_DATASET_PREFIX, "", str(credential.provider_account_id)
    )


def selected_channels(credential: IntegrationCredential) -> list[str]:
    """The selected channel ids, sorted and deduplicated. No key means nothing selected."""
    selected = (credential.provider_metadata or {}).get(SELECTION_KEY)
    if selected is None:
        return []
    if not isinstance(selected, list) or not all(
        isinstance(channel, str) and _CHANNEL.fullmatch(channel) for channel in selected
    ):
        raise ValueError("Slack selection must be a list of channel ids (C... or G...)")
    return sorted(set(selected))


def _classify_error(error: BaseException) -> str:
    """The count key a failed channel is reported under, from the source's typed errors."""
    from cognee_connector_slack.cognee_host.adapter import SlackReconnectError
    from cognee_connector_slack.source import (
        SlackAuthError,
        SlackMissingScopeError,
        SlackRateLimitedError,
        SlackUnavailableError,
    )

    current: BaseException | None = error
    for _ in range(_ERROR_CHAIN_DEPTH):
        if current is None:
            break
        if isinstance(current, SlackReconnectError):
            return RECONNECT_KEY  # a channel's own token refresh refused (G91)
        if isinstance(current, SlackMissingScopeError):
            return "failed_missing_scope"
        if isinstance(current, SlackAuthError):
            return "failed_auth"
        if isinstance(current, SlackRateLimitedError):
            return "failed_rate_limit"
        if isinstance(current, SlackUnavailableError):
            return "failed_unavailable"
        if isinstance(current, SlackChannelAccessError):
            return "failed_channel_access"
        current = current.__cause__ or current.__context__
    return "failed_ingestion"


def _count_failure(counts: dict[str, int], key: str) -> None:
    counts["failed"] += 1
    counts[key] = counts.get(key, 0) + 1


def channel_members(client: Any, channel_id: str) -> set[str]:
    """Every member of a channel, across all pages of ``conversations.members``."""
    members: set[str] = set()
    cursor = ""
    cursors: set[str] = set()
    while True:
        params: dict[str, Any] = {"channel": channel_id, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        page = client.call("conversations.members", **params)
        batch = page.get("members")
        if not isinstance(batch, list):
            raise SlackChannelAccessError("Slack returned an unreadable member list")
        members.update(batch)
        cursor = ((page.get("response_metadata") or {}).get("next_cursor") or "").strip()
        if not cursor:
            return members
        if cursor in cursors:
            raise SlackChannelAccessError("Slack repeated a member list cursor")
        cursors.add(cursor)


def channel_verdict(client: Any, channel_id: str, installer: str) -> tuple[str, dict[str, Any]]:
    """Whether a selected channel is synced, kept as archived, retired or reported unavailable,
    with Slack's description of it (empty when Slack says the channel does not exist).

    The installer reads the workspace dataset, so a channel is read only while Slack lets
    them read it. Any member of the workspace may read a public channel; a private one
    needs the installer as a member, because the app may sit in private channels its
    installer cannot see. An archived channel is not asked for the app's membership:
    Slack does not say whether the app stays a member, and the channel is not read anyway.
    Raises (``SlackAPIError``, ``SlackChannelAccessError``) when Slack gives no clear
    answer, so an uncertain answer never retires anything.
    """
    from cognee_connector_slack.source import SlackAPIError

    try:
        info = client.call("conversations.info", channel=channel_id).get("channel") or {}
    except SlackAPIError as error:
        if error.code == "channel_not_found":
            return UNAVAILABLE, {}
        raise
    if info.get("id") != channel_id:
        # No channel object, or another channel's: not an answer about this one.
        raise SlackChannelAccessError("Slack did not describe the channel")
    if info.get("is_im") or info.get("is_mpim"):
        return RETIRE, info
    archived = bool(info.get("is_archived"))
    if not archived and not info.get("is_member"):
        return UNAVAILABLE, info
    if info.get("is_private") and installer not in channel_members(client, channel_id):
        return RETIRE, info
    return (ARCHIVED if archived else SYNC), info


async def _sort_channels(
    client: Any, credential: IntegrationCredential, channels: list[str], counts: dict[str, int]
) -> tuple[list[str], list[str], list[str]]:
    """The selected channels to sync, those that keep their data without a sync, and those
    whose verdict is to retire them.

    The retired ones are removed, and counted, only when nothing blocks the retirement.
    """
    from cognee.modules.integrations.credentials import CredentialInactiveError

    from cognee_connector_slack.source import (
        SlackAPIError,
        SlackAuthError,
        SlackRateLimitedError,
    )

    metadata = credential.provider_metadata or {}
    installer = metadata.get("installed_by_slack_user_id")
    if not installer:
        _count_failure(counts, RECONNECT_KEY)
        raise SlackSyncError("Reconnect Slack to identify the user authorizing channel access.")
    try:
        identity = await asyncio.to_thread(client.call, "auth.test")
    except SlackAuthError as error:
        if error.code != "account_inactive":
            # Counted, so the owner is told to reconnect, not to retry (G76).
            _count_failure(counts, "failed_auth")
            raise
        await _end_connection(credential)
        raise CredentialInactiveError(
            "Slack deleted the workspace or the app's bot user"
        ) from error
    host = urlparse(identity.get("url") or "").hostname or ""
    if identity.get("team_id") != credential.provider_account_id or not host.endswith(".slack.com"):
        _count_failure(counts, RECONNECT_KEY)
        raise SlackSyncError("The Slack token does not belong to the connected workspace.")
    forget_unavailable = metadata.get(FORGET_UNAVAILABLE_KEY) is True
    synced, kept, retired = [], [], []
    for channel in channels:
        await ingestion.require_active_credential(credential)
        try:
            verdict, _ = await asyncio.to_thread(channel_verdict, client, channel, installer)
        except SlackAuthError:
            # The token is gone: no other channel can be read either. Counted, so the
            # owner is told to reconnect (G85).
            _count_failure(counts, "failed_auth")
            raise
        except (SlackChannelAccessError, SlackAPIError) as error:
            limited = isinstance(error, SlackRateLimitedError)
            _count_failure(counts, "failed_rate_limit" if limited else "failed_channel_access")
            logger.warning(
                "Slack channel skipped for workspace %s: %s", credential.provider_account_id, error
            )
            kept.append(channel)
            continue
        if verdict == SYNC:
            synced.append(channel)
        elif verdict == ARCHIVED:
            counts["channels_archived"] = counts.get("channels_archived", 0) + 1
            kept.append(channel)
        elif verdict == UNAVAILABLE and not forget_unavailable:
            _count_failure(counts, UNAVAILABLE_KEY)
            logger.warning(
                "Slack channel unavailable for workspace %s: deleted, or the app was removed",
                credential.provider_account_id,
            )
            kept.append(channel)
        else:
            retired.append(channel)
    return synced, kept, retired


async def _end_connection(credential: IntegrationCredential) -> None:
    """End a connection whose workspace or bot user Slack deleted, as an uninstall does.

    For a bot token Slack answers ``account_inactive`` only then, so the connection cannot
    come back; retrying it every interval would only hide that. Its data stays until the
    owner deletes it, and the status says Slack ended the connection.
    """
    from cognee_connector_slack.cognee_host.persistence import revoke_by_team

    await revoke_by_team(str(credential.provider_account_id))
    logger.warning(
        "Slack ended workspace %s (account_inactive): connection revoked, data kept",
        credential.provider_account_id,
    )


class _FreshTokenService:
    """A Slack client that takes a fresh token when its channel's walk makes its first call.

    A rotating token lives 12 hours, which a long multi-channel sync can outlast, so each
    channel asks for its own. The walk runs on dlt's worker thread, so the token is fetched
    by bridging to the API loop, like ``ingestion.extraction_checkpoint``.
    """

    def __init__(self, credential: IntegrationCredential, loop: asyncio.AbstractEventLoop):
        self._credential = credential
        self._loop = loop
        self._client: Any = None

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        if self._client is None:
            from cognee_connector_slack.cognee_host.adapter import access_token_for

            token = asyncio.run_coroutine_threadsafe(
                access_token_for(self._credential), self._loop
            ).result()
            self._client = slack_source_module.build_slack_service(token)
        return self._client.call(method, **params)


async def sync_slack(credential: IntegrationCredential) -> None:
    """Run a sync of the connection's selected channels and record its outcome."""
    await ingestion.run_sync(PROVIDER, credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    """Sync with the connection as it is now.

    ``run_sync`` keeps one sync per workspace and runs it once more after a changed
    selection, with the credential it started with, so the selection is read afresh here.
    Workspaces stage apart (one dlt dataset each), so syncs of different workspaces run side
    by side.
    """
    credential = await ingestion.require_active_credential(credential)
    return await _sync_channels(credential, counts)


async def _sync_channels(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee_connector_slack.cognee_host.adapter import access_token_for
    from cognee_connector_slack.source import SlackMissingScopeError

    dataset_name = dataset_name_for_credential(credential)
    channels = selected_channels(credential)
    if not channels:
        await ingestion.retire_resources(PROVIDER, credential, dataset_name, set())
        return SYNC_STATUS_OK, counts
    if REQUIRED_SCOPE not in (credential.scopes or "").split(","):
        # Authors are rendered by name, and there is no fallback to raw ids: stop the
        # whole workspace instead of failing every channel the same way.
        _count_failure(counts, "failed_missing_scope")
        raise SlackMissingScopeError("users.info", REQUIRED_SCOPE)

    from cognee_connector_slack.cognee_host.adapter import SlackReconnectError

    try:
        token = await access_token_for(credential)
    except SlackReconnectError:
        _count_failure(counts, RECONNECT_KEY)  # the card says reconnect, not retry (G85)
        raise
    client = slack_source_module.build_slack_service(token)
    to_sync, to_keep, to_retire = await _sort_channels(client, credential, channels, counts)
    source_factory = slack_source_module.slack_source
    loop = asyncio.get_running_loop()
    marked = thread_marks(credential.provider_metadata or {})
    unread = {channel: set(stamps) for channel, stamps in marked.items()}
    excluded = (credential.provider_metadata or {}).get(EXCLUDED_MESSAGES_KEY) or {}

    sources: dict[str, Any] = {}

    def make_source(channel: str, resource_name: str, check_active):
        sources[channel] = source_factory(
            service=_FreshTokenService(credential, loop),
            team_id=str(credential.provider_account_id),
            channel_id=channel,
            resource_name=resource_name,
            check_active=check_active,
            changed_roots=unread.setdefault(channel, set()),
            excluded_messages=set(excluded.get(channel, [])),
        )
        return sources[channel]

    stored: set[str] = set()
    retained = await ingestion.sync_scopes(
        PROVIDER,
        credential,
        counts,
        scopes=to_sync,
        dataset_name=dataset_name,
        make_source=make_source,
        classify_error=_classify_error,
        on_scope_stored=stored.add,
    )
    # A run that failed stored nothing, so its marks stay for the next sync (G57).
    await _clear_read_threads(
        credential,
        {
            channel: {
                root: stamp
                for root, stamp in stamps.items()
                if root not in unread.get(channel, set())
            }
            for channel, stamps in marked.items()
            if channel in stored
        },
    )
    retained |= {ingestion.resource_name(PROVIDER, credential, channel) for channel in to_keep}
    # A stored channel whose source counts a failure was cut short by it (budget, rate
    # limit, an outage after rows): it resumes next time and holds only its own table.
    cut = sum(1 for channel in stored if ingestion.source_counts(sources[channel]).get("failed"))
    if not ingestion.retirement_blocked(counts, gone_keys=(UNAVAILABLE_KEY,), cut_scopes=cut):
        removed = await ingestion.retire_resources(PROVIDER, credential, dataset_name, retained)
        # Only the channels whose documents this pass removed: a blocked pass keeps them,
        # and a channel removed by an earlier pass has nothing left to remove.
        retired_now = [
            channel
            for channel in to_retire
            if ingestion.resource_name(PROVIDER, credential, channel) in removed
        ]
        if retired_now:
            counts["channels_retired"] = len(retired_now)
    return (SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK), counts


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------
def is_due(credential: IntegrationCredential, now: datetime) -> bool:
    """A connection with channels selected, not syncing, and not synced in its interval
    (``next_sync_at``)."""
    if not (credential.provider_metadata or {}).get(SELECTION_KEY):
        return False
    if ingestion.sync_is_running(PROVIDER, credential.provider_account_id):
        return False
    due = next_sync_at(credential)
    return due is None or now >= due


def next_sync_at(credential: IntegrationCredential) -> datetime | None:
    """When the tick next syncs the connection; ``None`` when it never has (the next tick).

    Six hours after the last sync, or ten minutes after a run that was cut short and
    resumes.
    """
    last = credential.last_synced_at
    if last is None:
        return None
    if last.tzinfo is None:
        # SQLite stores no timezone; the value is written as UTC.
        last = last.replace(tzinfo=timezone.utc)
    counts = (credential.provider_metadata or {}).get(LAST_SYNC_COUNTS_KEY) or {}
    from cognee_connector_slack.source import STALLED_KEY

    # Each cut channel counts once under its cut key; one that could not move past a day
    # (G56) counts under STALLED_KEY too and is not worth resuming early.
    cut_short = credential.sync_status == SYNC_STATUS_DEGRADED and sum(
        counts.get(key, 0) for key in RESUMABLE_COUNT_KEYS
    ) > counts.get(STALLED_KEY, 0)
    return last + (RESUME_INTERVAL if cut_short else SYNC_INTERVAL)


# Why a degraded sync could not finish, one plain sentence per failure kind, in the words
# the Integrations page uses (``degradedReasons`` in SdkIntegrationCard.tsx).
DEGRADED_REASONS = {
    "failed_day_too_large": (
        "A channel has more messages in one day than the provider's rate limit lets a sync "
        "read, so its history stops there."
    ),
    "failed_rate_limit": (
        "Slack rate limit reached. The sync continues on its own in about ten minutes."
    ),
    "failed_budget": "Large history: the sync continues where it stopped in about ten minutes.",
    "failed_unavailable": "Slack was unavailable; the sync continues on its own.",
    "failed_auth": "Slack signed Cognee out. Reconnect to keep syncing.",
    "failed_missing_scope": "Reconnect to grant the permissions Cognee needs.",
    "failed_reconnect": (
        "The connection no longer matches the Slack workspace. Reconnect to keep syncing."
    ),
    "failed_channel_access": (
        "The app can no longer read a channel. Add Cognee to it again, or deselect it."
    ),
    "failed_channel_unavailable": "A selected channel was deleted, or the app was removed from it.",
    "failed_thread_unreadable": (
        "Slack answered a thread in a way Cognee cannot read. Its last stored version is "
        "kept, and the next sync tries it again."
    ),
}
DEGRADED_FALLBACK = "Some content could not be processed. Refresh to retry."


def degraded_reasons(counts: dict[str, int]) -> list[str]:
    """The sentences for the failure kinds ``counts`` records, in ``DEGRADED_REASONS``
    order; the general one when none is known."""
    return [text for key, text in DEGRADED_REASONS.items() if counts.get(key)] or [
        DEGRADED_FALLBACK
    ]


async def workspace_item_count(credential: IntegrationCredential) -> int:
    """How many items the connection's workspace dataset holds, in one query."""
    from cognee.modules.data.models import Data, Dataset
    from sqlalchemy import func

    async with get_relational_engine().get_async_session() as db:
        return (
            await db.execute(
                select(func.count(Data.id))
                .join(Dataset, Data.dataset_id == Dataset.id)
                .where(
                    Dataset.name == dataset_name_for_credential(credential),
                    Dataset.owner_id == credential.user_id,
                )
            )
        ).scalar_one()


async def is_paused(credential: IntegrationCredential) -> bool:
    """Whether a pause holds the connection (SDK-1434), asked quietly: the tick asks every
    minute, and ``ingestion.sync_is_paused`` would log each time (G62)."""
    from cognee.modules.integrations.credentials import owner_principal_id
    from cognee.modules.integrations.sync_policy import get_sync_pause_level

    level = await get_sync_pause_level(
        credential_id=credential.id, principal_id=owner_principal_id(credential)
    )
    return level is not None


async def sync_due_workspaces() -> int:
    """Start a sync for every connection that is due, and return how many were started.

    Each sync runs as its own task, so a workspace that becomes due while another one's
    long first sync runs starts on the next tick; ``is_due`` skips a connection whose sync
    is still running. A failed sync is logged and stops nothing else.
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
    due = [
        credential
        for credential in credentials
        if is_due(credential, now) and not await is_paused(credential)
    ]
    for credential in due:
        task = asyncio.create_task(
            sync_slack(credential), name=f"slack sync of workspace {credential.provider_account_id}"
        )
        _scheduled.add(task)
        task.add_done_callback(_scheduled_finished)
    return len(due)


def _scheduled_finished(task: asyncio.Task) -> None:
    _scheduled.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error("Scheduled %s failed", task.get_name(), exc_info=task.exception())


async def mark_thread_changed(team_id: str, channel_id: str, root_ts: str) -> None:
    """Have the next sync read a thread again whose root looks unchanged (G6).

    Only for a selected channel of an active connection: another channel is not synced.
    """
    from cognee.modules.integrations.credentials import (
        change_provider_metadata,
        get_credential_by_account,
    )

    credential = await get_credential_by_account(PROVIDER, team_id)
    if credential is None or credential.status != "active":
        return
    if channel_id not in selected_channels(credential):
        return

    def add_mark(metadata: dict) -> dict:
        marks = thread_marks(metadata)
        marks.setdefault(channel_id, {})[root_ts] = uuid4().hex
        return {CHANGED_THREADS_KEY: marks}

    await change_provider_metadata(PROVIDER, team_id, add_mark)


async def exclude_message(
    credential: IntegrationCredential, *, channel_id: str, message_ts: str, root_ts: str
) -> bool:
    """Take one message of a synced channel out of the workspace's memory, for good.

    Records it under ``EXCLUDED_MESSAGES_KEY``, marks its thread (``root_ts``, the message
    itself when it is not in a thread) so the next sync renders that thread or day again
    without it, and starts that sync. ``False`` for a channel that is not synced. The
    caller decides who may (``handle_cognee_mention.forget_and_confirm``).
    """
    from cognee.modules.integrations.credentials import change_provider_metadata

    if channel_id not in selected_channels(credential):
        return False
    team_id = str(credential.provider_account_id)

    def add_excluded(metadata: dict) -> dict:
        excluded = dict(metadata.get(EXCLUDED_MESSAGES_KEY) or {})
        excluded[channel_id] = sorted({*excluded.get(channel_id, []), message_ts}, key=Decimal)
        return {EXCLUDED_MESSAGES_KEY: excluded}

    await change_provider_metadata(PROVIDER, team_id, add_excluded)
    await mark_thread_changed(team_id, channel_id, root_ts)
    await start_sync(credential)
    return True


async def start_sync(credential: IntegrationCredential) -> None:
    """Sync the workspace now, or once more after the sync already running; a paused
    connection waits for its resume (G62)."""
    from cognee.modules.integrations.background import spawn_background

    if await is_paused(credential):
        logger.info("Slack sync of %s held: the connection is paused", credential.id)
        return
    if ingestion.request_rerun_if_running(PROVIDER, str(credential.provider_account_id)):
        return
    spawn_background(sync_slack(credential), description="Slack sync after a removed message")


def thread_marks(metadata: dict) -> dict[str, dict[str, str]]:
    """``CHANGED_THREADS_KEY`` as ``{channel_id: {root_ts: stamp}}``, a copy."""
    marks = metadata.get(CHANGED_THREADS_KEY) or {}
    return {
        channel: dict(roots) if isinstance(roots, dict) else dict.fromkeys(roots, "")
        for channel, roots in marks.items()
    }


async def _clear_read_threads(
    credential: IntegrationCredential, read: dict[str, dict[str, str]]
) -> None:
    """Drop the marks this sync read; a mark written again while it ran has a new stamp
    and stays for the next one (G72)."""
    from cognee.modules.integrations.credentials import change_provider_metadata

    if not any(read.values()):
        return

    def drop_read(metadata: dict) -> dict:
        marks = thread_marks(metadata)
        for channel, stamps in read.items():
            left = {
                root: stamp
                for root, stamp in marks.get(channel, {}).items()
                if stamps.get(root) != stamp
            }
            if left:
                marks[channel] = left
            else:
                marks.pop(channel, None)
        return {CHANGED_THREADS_KEY: marks}

    await change_provider_metadata(PROVIDER, str(credential.provider_account_id), drop_read)


async def record_uninstall(team_id: str) -> None:
    """Note when the app was uninstalled from ``team_id``; a retried event keeps the first."""
    from cognee.modules.integrations.credentials import change_provider_metadata

    def first_uninstall(metadata: dict) -> dict | None:
        if metadata.get(UNINSTALLED_AT_KEY):
            return None
        return {UNINSTALLED_AT_KEY: datetime.now(timezone.utc).isoformat()}

    await change_provider_metadata(PROVIDER, team_id, first_uninstall)


async def forget_uninstalled_workspaces(now: datetime | None = None) -> int:
    """Delete the data of every workspace uninstalled at least 14 days ago, and return how many.

    A workspace qualifies while its connection is still revoked and carries
    ``UNINSTALLED_AT_KEY``: a reconnect makes it active again and clears the key. The
    documents the connector synced (a dataset they leave empty goes with them) and every
    member's remembered copies from it are forgotten, then
    ``UNINSTALL_DELETED_AT_KEY`` is stamped, so a workspace is deleted once and its owner is
    told when (G84). A workspace that fails is logged and tried again on the next tick.
    """
    from cognee.modules.integrations.credentials import get_credential_by_account

    now = now or datetime.now(timezone.utc)
    async with get_relational_engine().get_async_session() as db:
        credentials = (
            (
                await db.execute(
                    select(IntegrationCredential).where(
                        IntegrationCredential.provider == PROVIDER,
                        IntegrationCredential.status == "revoked",
                    )
                )
            )
            .scalars()
            .all()
        )
    forgotten = 0
    for credential in credentials:
        uninstalled = _uninstalled_at(credential)
        if uninstalled is None or now - uninstalled < UNINSTALL_RETENTION:
            continue
        if (credential.provider_metadata or {}).get(UNINSTALL_DELETED_AT_KEY):
            continue  # deleted already
        # Read again just before: a workspace connected again while an earlier one was
        # being deleted keeps its data (G75).
        current = await get_credential_by_account(PROVIDER, credential.provider_account_id)
        if (
            current is None
            or current.status != "revoked"
            or _uninstalled_at(current) is None
            or (current.provider_metadata or {}).get(UNINSTALL_DELETED_AT_KEY)
        ):
            continue
        try:
            await _forget_workspace(current)
        except Exception:  # the next tick tries again; one workspace never stops the rest
            logger.exception(
                "Could not delete the data of uninstalled Slack workspace %s; will retry",
                credential.provider_account_id,
            )
            continue
        forgotten += 1
    return forgotten


def _uninstalled_at(credential: IntegrationCredential) -> datetime | None:
    return _metadata_time(credential, UNINSTALLED_AT_KEY)


def _uninstall_deleted_at(credential: IntegrationCredential) -> datetime | None:
    return _metadata_time(credential, UNINSTALL_DELETED_AT_KEY)


def _metadata_time(credential: IntegrationCredential, key: str) -> datetime | None:
    value = (credential.provider_metadata or {}).get(key)
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def _forget_workspace(credential: IntegrationCredential) -> None:
    from cognee.api.v1.datasets.datasets import datasets as datasets_api
    from cognee.modules.data.methods import get_dataset_data
    from cognee.modules.data.models import Dataset
    from cognee.modules.integrations.credentials import change_provider_metadata
    from cognee.modules.users.methods import get_user
    from cognee.tasks.ingestion.ingest_dlt_source import drop_dlt_dataset

    from cognee_connector_slack.cognee_host.handle_cognee_ask import forget_answer_records
    from cognee_connector_slack.cognee_host.remember_message import forget_workspace_remembered

    team_id = str(credential.provider_account_id)
    dataset_name = dataset_name_for_credential(credential)
    # Every owner's copy: a previous owner who disconnected keeping the data holds a
    # dataset of the same name (the name derives from the workspace alone, G44). The name
    # is not proof that the connector made the dataset (anyone can create one by that
    # name), so only the documents the connector wrote go, and a dataset only when they
    # leave it empty.
    async with get_relational_engine().get_async_session() as db:
        datasets = (
            await db.execute(
                select(Dataset.id, Dataset.owner_id).where(Dataset.name == dataset_name)
            )
        ).all()
    for dataset_id, owner_id in datasets:
        synced = [
            data.id
            for data in await get_dataset_data(dataset_id)
            if isinstance(data.system_metadata, dict)
            and data.system_metadata.get("source") == PROVIDER
        ]
        if not synced:
            continue
        owner = await get_user(owner_id)
        for data_id in synced:
            await datasets_api.delete_data(
                dataset_id, data_id, user=owner, delete_dataset_if_empty=True
            )
    # forget leaves the staging copy, the dlt state and the load packages, which hold
    # every message's text (G43).
    await drop_dlt_dataset(dataset_name)
    await forget_workspace_remembered(team_id)
    await forget_answer_records(team_id, credential.user_id, every_owner=True)
    uninstalled = (credential.provider_metadata or {}).get(UNINSTALLED_AT_KEY)

    def stamp_if_still_uninstalled(metadata: dict) -> dict | None:
        # A reconnect during the deletion cleared the key under the same lock: the
        # connection is active again, and a stamp would skip its next uninstall (G90).
        if metadata.get(UNINSTALLED_AT_KEY) != uninstalled:
            return None
        return {UNINSTALL_DELETED_AT_KEY: datetime.now(timezone.utc).isoformat()}

    await change_provider_metadata(PROVIDER, team_id, stamp_if_still_uninstalled)
    logger.info(
        "Deleted the data of Slack workspace %s, uninstalled at least %s days ago",
        team_id,
        UNINSTALL_RETENTION.days,
    )


async def _worker() -> None:
    await asyncio.sleep(_STARTUP_DELAY_SECONDS)
    while True:
        if slack_settings.history_sync_enabled:
            try:
                await sync_due_workspaces()
            except asyncio.CancelledError:
                raise
            except Exception:  # keep the scheduler alive across a failed credential read
                logger.exception("Slack sync tick failed; will retry")
        try:
            await forget_uninstalled_workspaces()
        except asyncio.CancelledError:
            raise
        except Exception:  # keep the scheduler alive across a failed credential read
            logger.exception("Slack uninstall cleanup failed; will retry")
        await asyncio.sleep(_TICK_SECONDS)


@asynccontextmanager
async def slack_sync_lifespan(app):
    # Runs without the history sync too: remembered copies need the uninstall cleanup.
    task = None
    if slack_settings.client_id:
        task = asyncio.create_task(_worker())
    try:
        yield
    finally:
        pending = [*_scheduled, *([task] if task else [])]
        for running in pending:
            running.cancel()
        # A sync's own failure is logged by its done callback; here it only has to end.
        await asyncio.gather(*pending, return_exceptions=True)
