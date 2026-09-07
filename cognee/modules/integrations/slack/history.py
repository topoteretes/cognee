"""Import selected Slack conversations through native Cognee add/update/delete.

Only source selection and source identity live here. Documents, graph state,
ACLs, indexing, and deletion belong to Cognee's existing APIs. A conversation
is one document so replies retain their context and edits replace that same
document. There is no full-dataset replacement or rolling-window cleanup.
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select

from cognee.api.v1.add import add
from cognee.api.v1.cognify import cognify
from cognee.api.v1.datasets import datasets
from cognee.api.v1.update import update
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks.dataset_lock import dataset_lock
from cognee.modules.data.models import Data
from cognee.modules.integrations.credentials import decrypt_token_payload
from cognee.modules.integrations.slack.adapter import SlackIntegration
from cognee.modules.integrations.slack.history_client import SlackAPIError, SlackHistoryClient
from cognee.modules.integrations.slack.history_models import (
    SlackHistoryError,
    SlackHistoryRequest,
    SlackHistoryResult,
    parse_thread_link,
)
from cognee.modules.integrations.slack.persistence import get_by_team, is_active
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import check_permission_on_dataset
from cognee.tasks.ingestion.data_item import DataItem

SOURCE = "slack_history"
_token_refresh_lock = asyncio.Lock()


async def _access_token(credential):
    async with _token_refresh_lock:
        current = await get_by_team(credential.provider_account_id)
        if not is_active(current) or current.user_id != credential.user_id:
            raise SlackHistoryError("The Slack connection is no longer active.")
        expiry = getattr(current, "token_expires_at", None)
        if expiry is not None:
            expiry = expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry
            if (expiry - datetime.now(timezone.utc)).total_seconds() < 120:
                await SlackIntegration().refresh(current)
                current = await get_by_team(credential.provider_account_id)
                if not is_active(current) or current.user_id != credential.user_id:
                    raise SlackHistoryError("The Slack connection changed during token refresh.")
        return decrypt_token_payload(current).get("access_token")


async def authorize_import(team_id: str, user: User, selection: SlackHistoryRequest):
    """Require the connection owner and native dataset permissions, before fetching text."""
    credential = await get_by_team(team_id)
    if not is_active(credential) or credential.user_id != user.id:
        raise SlackHistoryError("Only the owner of an active Slack connection can import history.")
    if not (credential.provider_metadata or {}).get("installed_by_slack_user_id"):
        raise SlackHistoryError("Reconnect Slack to identify the user authorizing history access.")
    # update() may use the native delete/add fallback; fail before fetching
    # source content if that operation would not be authorized.
    for permission in ("read", "write", "delete"):
        await check_permission_on_dataset(user, permission, selection.dataset_id)
    return credential


async def _stored_documents(dataset_id: UUID, team_id: str) -> dict[tuple[str, str], Data]:
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        rows = (await db.execute(select(Data).where(Data.dataset_id == dataset_id))).scalars().all()
    documents = {}
    for row in rows:
        stamp = (row.system_metadata or {}).get(SOURCE) or {}
        if stamp.get("team_id") == team_id and stamp.get("version") == 1:
            documents[(stamp["channel_id"], stamp["thread_ts"])] = row
    return documents


def _root(messages: list[dict], fallback: str) -> str:
    if not messages:
        return fallback
    return messages[0].get("thread_ts") or messages[0]["ts"]


def _document(
    team: str, channel: dict, root: str, messages: list[dict], base_url: str, bot_user: str
):
    lines = []
    for message in messages:
        text = (message.get("text") or "").strip()
        if (
            not text
            or message.get("subtype") in {"tombstone", "channel_join", "channel_leave"}
            or (bot_user and message.get("user") == bot_user)
        ):
            continue
        timestamp = message["ts"]
        author = message.get("user") or message.get("bot_id") or "unknown"
        when = datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
        link = f"{base_url}/archives/{channel['id']}/p{timestamp.replace('.', '')}"
        lines.append(f"[{when}] <@{author}> ({link})\n{text}")
    if not lines:
        return None, 0
    source_uri = f"{base_url}/archives/{channel['id']}/p{root.replace('.', '')}"
    content = (
        f"Slack conversation in #{channel['name']} ({channel['id']}), workspace {team}\n"
        f"Thread: {root}\nSource: {source_uri}\n\n" + "\n\n".join(lines)
    )
    return (content, source_uri), len(lines)


async def import_slack_history(
    team_id: str,
    selection: SlackHistoryRequest,
    *,
    user: User,
    reconcile: bool = False,
) -> SlackHistoryResult:
    """Import a date window or explicit thread links into an existing dataset.

    ``reconcile`` additionally re-fetches previously imported conversations in
    selected channels. It catches edits, late replies and deletions after an
    outage; only an explicit ``thread_not_found``/``message_not_found`` from a
    successfully authorized channel removes an existing source document.
    Changing a lookback never infers deletion from absence in a window.

    All source pages and permissions are checked before mutations. Native
    writes are idempotent, not a multi-document transaction: after a pipeline
    failure retry the same selection. Run one SDK/API process against local
    embedded stores, as required by Cognee's process-local dataset locks.
    """
    selection = SlackHistoryRequest.model_validate(selection)
    oldest, latest = selection.bounds()
    credential = await authorize_import(team_id, user, selection)
    metadata = credential.provider_metadata or {}
    slack_user = metadata["installed_by_slack_user_id"]
    links = [parse_thread_link(link) for link in selection.thread_links]
    links.extend((thread.channel_id, thread.ts) for thread in selection.threads)
    channel_ids = set(selection.channel_ids) | {channel for channel, _ in links}
    allowed = set(metadata.get("allowed_channel_ids") or [])
    if allowed and not channel_ids <= allowed:
        raise SlackHistoryError("A selected channel is outside the configured Slack allowlist.")

    # Hold the native re-entrant dataset lock across fetch and apply: another
    # import must not apply an older snapshot after a newer one in this process.
    async with dataset_lock(selection.dataset_id):
        token = await _access_token(credential)
        stored = await _stored_documents(selection.dataset_id, team_id)
        conversations = {}
        deleted = set()
        channels = {}
        async with SlackHistoryClient(
            token, max_requests=selection.max_requests, max_messages=selection.max_messages
        ) as client:
            identity = await client.call("auth.test")
            base_url = (identity.get("url") or "").rstrip("/")
            if identity.get("team_id") != team_id or not (
                urlparse(base_url).hostname or ""
            ).endswith(".slack.com"):
                raise SlackHistoryError(
                    "The Slack token does not belong to the selected workspace."
                )
            for link in selection.thread_links:
                if urlparse(link).hostname != urlparse(base_url).hostname:
                    raise SlackHistoryError("A thread link belongs to another Slack workspace.")
            for channel in sorted(channel_ids):
                channels[channel] = await client.authorize_channel(channel, slack_user)

            async def fetch_thread(channel: str, ts: str):
                key = (channel, ts)
                if key in conversations or key in deleted:
                    return conversations.get(key, [])
                try:
                    messages = await client.thread(channel, ts)
                except SlackAPIError as error:
                    if error.code not in {"thread_not_found", "message_not_found"}:
                        raise
                    # Missing explicit links are mistakes on first import,
                    # source deletions only when we already own that document.
                    if key not in stored:
                        raise SlackHistoryError(
                            "A selected Slack conversation no longer exists."
                        ) from error
                    deleted.add(key)
                    return []
                if not messages:
                    raise SlackHistoryError(
                        "Slack returned an empty thread without confirming deletion."
                    )
                root = _root(messages, ts)
                conversations[(channel, root)] = messages
                return messages

            for channel in selection.channel_ids:
                roots = await client.history(
                    channel,
                    oldest="0" if selection.thread_mode == "active" else str(oldest.timestamp()),
                    latest=str(latest.timestamp()),
                )
                for root in roots:
                    ts = root.get("thread_ts") or root.get("ts")
                    if not ts:
                        raise SlackHistoryError(
                            "Slack returned a conversation without a timestamp."
                        )
                    stamp = Decimal(ts)
                    in_window = (
                        Decimal(str(oldest.timestamp()))
                        <= stamp
                        <= Decimal(str(latest.timestamp()))
                    )
                    if not in_window and selection.thread_mode == "started":
                        continue
                    if not in_window and not root.get("reply_count"):
                        continue
                    messages = await fetch_thread(channel, ts)
                    if selection.thread_mode == "active" and not any(
                        Decimal(str(oldest.timestamp()))
                        <= Decimal(message["ts"])
                        <= Decimal(str(latest.timestamp()))
                        for message in messages
                    ):
                        conversations.pop((channel, _root(messages, ts)), None)

            for channel, ts in links:
                await fetch_thread(channel, ts)
            if reconcile:
                for channel, ts in stored:
                    if channel in selection.channel_ids:
                        await fetch_thread(channel, ts)

        # A connection or dataset grant may have been revoked during a long fetch.
        current = await authorize_import(team_id, user, selection)
        current_allowed = set((current.provider_metadata or {}).get("allowed_channel_ids") or [])
        if current.id != credential.id or (current_allowed and not channel_ids <= current_allowed):
            raise SlackHistoryError(
                "Slack access changed during the import. Retry with the current selection."
            )

        result = SlackHistoryResult(dataset_id=selection.dataset_id)
        for (channel, root), messages in conversations.items():
            document, count = _document(
                team_id,
                channels[channel],
                root,
                messages,
                base_url,
                metadata.get("bot_user_id", ""),
            )
            key = (channel, root)
            if document is None:
                if key in stored:
                    deleted.add(key)
                continue
            content, source_uri = document
            digest = hashlib.sha256(content.encode()).hexdigest()
            existing = stored.get(key)
            result.conversations += 1
            result.messages += count
            if (
                existing
                and ((existing.system_metadata or {}).get(SOURCE) or {}).get("digest") == digest
            ):
                result.unchanged += 1
                continue
            stamp = {
                "version": 1,
                "team_id": team_id,
                "channel_id": channel,
                "thread_ts": root,
                "digest": digest,
            }
            item = DataItem(
                data=content,
                data_id=existing.id
                if existing
                else uuid5(
                    NAMESPACE_URL, f"cognee:slack:{selection.dataset_id}:{team_id}:{channel}:{root}"
                ),
                label=f"Slack #{channels[channel]['name']} · {root}",
                external_metadata={"source_uri": source_uri, "slack": stamp},
                system_metadata={SOURCE: stamp},
            )
            node_set = ["slack", f"slack:{team_id}", f"slack:channel:{channel}"]
            if existing:
                native_result = await update(
                    existing.id,
                    item,
                    dataset_id=selection.dataset_id,
                    user=user,
                    node_set=node_set,
                    chunk_level_diff=False,
                )
                result.updated += 1
            else:
                native_result = await add(
                    item, dataset_id=selection.dataset_id, user=user, node_set=node_set
                )
                result.added += 1
            if get_errored_run_info(native_result) is not None:
                raise SlackHistoryError("Cognee could not ingest a conversation. Retry the import.")
        for key in deleted:
            if key in stored:
                await datasets.delete_data(
                    dataset_id=selection.dataset_id,
                    data_id=stored[key].id,
                    user=user,
                )
                result.deleted += 1
        # Always retry pending native indexing, even if a previous run saved
        # the raw document successfully and then failed while building its graph.
        if result.conversations:
            await cognify(datasets=[selection.dataset_id], user=user, raise_on_error=True)
        return result
