"""Slack import dialog and full-thread shortcut. Replies stay private."""

import asyncio
import json
from urllib.parse import parse_qs

import aiohttp
from pydantic import ValidationError

from cognee.modules.integrations.credentials import decrypt_token_payload
from cognee.modules.integrations.slack.history_models import (
    SlackHistoryError,
    SlackHistoryRequest,
    SlackSyncSettings,
)
from cognee.modules.integrations.slack.history_sync import (
    background_import,
    configure_slack_sync,
    run_history_import,
)
from cognee.modules.integrations.slack.persistence import get_by_team, is_active
from cognee.modules.integrations.slack.response_url import post_to_response_url
from cognee.modules.integrations.slack.slack_settings import slack_settings
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.shared.logging_utils import get_logger

logger = get_logger("slack_history")
HISTORY_VIEW = "cognee_history_import"
HISTORY_SHORTCUT = "import_history"
THREAD_SHORTCUT = "remember_thread"


async def _owner(team_id, slack_user_id):
    credential = await get_by_team(team_id)
    installed_by = (
        (credential.provider_metadata or {}).get("installed_by_slack_user_id")
        if is_active(credential)
        else None
    )
    if not installed_by or installed_by != slack_user_id:
        raise SlackHistoryError(
            "Only the person who connected this Slack workspace can import history."
        )
    return credential


def _input(name, label, element, *, optional=False):
    return {
        "type": "input",
        "block_id": name,
        "optional": optional,
        "label": {"type": "plain_text", "text": label},
        "element": {**element, "action_id": "value"},
    }


def history_view(*, team_id, response_url="", channel_id="", days=7, thread=None):
    channels = {
        "type": "multi_conversations_select",
        "filter": {
            "include": ["public", "private"],
            "exclude_bot_users": True,
        },
    }
    if channel_id.startswith(("C", "G")) and not thread:
        channels["initial_conversations"] = [channel_id]
    options = [
        {
            "text": {
                "type": "plain_text",
                "text": "Include older threads active during the period (slower)",
            },
            "value": "active",
        },
        {
            "text": {"type": "plain_text", "text": "Keep this selection synced every 6 hours"},
            "value": "sync",
        },
    ]
    return {
        "type": "modal",
        "callback_id": HISTORY_VIEW,
        "title": {"type": "plain_text", "text": "Import Slack history"},
        "submit": {"type": "plain_text", "text": "Import"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "private_metadata": json.dumps(
            {"team_id": team_id, "response_url": response_url, "thread": thread}
        ),
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        (
                            "The selected message's *entire thread* will be imported.\n"
                            if thread
                            else ""
                        )
                        + "Choose channels, paste thread links, or both. Selected threads include all available "
                        "replies, including context outside the date range. The destination dataset controls "
                        "who can read this information; use a restricted dataset for private conversations."
                    ),
                },
            },
            _input(
                "dataset",
                "Destination dataset",
                {
                    "type": "external_select",
                    "min_query_length": 0,
                    "placeholder": {"type": "plain_text", "text": "Choose a Cognee dataset"},
                },
            ),
            _input("channels", "Channels", channels, optional=True),
            _input(
                "days",
                "Past days (for channel selection)",
                {"type": "plain_text_input", "initial_value": str(days)},
            ),
            _input(
                "links",
                "Thread links (one per line)",
                {"type": "plain_text_input", "multiline": True},
                optional=True,
            ),
            _input(
                "options",
                "Import options",
                {"type": "checkboxes", "options": options},
                optional=True,
            ),
        ],
    }


async def _open_view(credential, trigger_id, view):
    token = decrypt_token_payload(credential).get("access_token")
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session,
            session.post(
                "https://slack.com/api/views.open",
                headers={"Authorization": f"Bearer {token}"},
                json={"trigger_id": trigger_id, "view": view},
                allow_redirects=False,
            ) as response,
        ):
            payload = await response.json()
            if response.status != 200 or not payload.get("ok"):
                raise SlackHistoryError("Could not open the import dialog. Try the command again.")
    except (aiohttp.ClientError, asyncio.TimeoutError):
        raise SlackHistoryError(
            "Could not open the import dialog. Try the command again."
        ) from None


async def handle_cognee_import(raw_body: bytes):
    form = {key: value[0] for key, value in parse_qs(raw_body.decode()).items()}
    try:
        credential = await _owner(form.get("team_id", ""), form.get("user_id", ""))
        try:
            days = int(form.get("text", "").strip() or "7")
        except ValueError:
            raise SlackHistoryError("Use /cognee-import or /cognee-import 14.") from None
        if not 1 <= days <= 3650:
            raise SlackHistoryError("Choose between 1 and 3650 days.")
        await _open_view(
            credential,
            form.get("trigger_id", ""),
            history_view(
                team_id=form["team_id"],
                response_url=form.get("response_url", ""),
                channel_id=form.get("channel_id", ""),
                days=days,
            ),
        )
    except SlackHistoryError as error:
        return {"response_type": "ephemeral", "text": str(error)}
    return {
        "response_type": "ephemeral",
        "text": "Choose the conversations and destination in the import dialog.",
    }


def _submission(payload):
    view = payload["view"]
    metadata = json.loads(view.get("private_metadata") or "{}")
    if metadata.get("team_id") != (payload.get("team") or {}).get("id"):
        raise ValueError("Slack workspace mismatch.")
    values = view["state"]["values"]
    value = lambda name: values.get(name, {}).get("value", {})
    options = {option["value"] for option in value("options").get("selected_options") or []}
    channels = value("channels").get("selected_conversations") or []
    selection = SlackHistoryRequest(
        dataset_id=(value("dataset").get("selected_option") or {}).get("value", ""),
        channel_ids=channels,
        thread_links=(value("links").get("value") or "").split(),
        threads=[metadata["thread"]] if metadata.get("thread") else [],
        days=int(value("days").get("value") or "7") if channels else None,
        thread_mode="active" if "active" in options else "started",
    )
    return metadata, selection, "sync" in options


async def _import_and_confirm(metadata, selection, *, user, keep_synced):
    response_url = metadata.get("response_url", "")
    try:
        result = await run_history_import(metadata["team_id"], selection, user=user)
        if keep_synced:
            await configure_slack_sync(
                metadata["team_id"],
                SlackSyncSettings(enabled=True, selection=selection),
                user=user,
            )
        text = f"Imported {result.conversations} conversations ({result.messages} messages) into Cognee."
        if keep_synced:
            if slack_settings.history_sync_enabled and slack_settings.client_id:
                text += " This selection will refresh every 6 hours."
            else:
                text += " Sync selection saved; this server's refresh worker is disabled."
    except SlackHistoryError as error:
        text = str(error)
    except Exception:  # noqa: BLE001 - detached work reports a private, sanitized failure
        logger.error("Slack history import failed for team %s", metadata["team_id"])
        text = "Import did not complete. Check destination permissions and server logs, then retry."
    await post_to_response_url(response_url, {"response_type": "ephemeral", "text": text})


async def handle_history_interactive(payload):
    team_id = (payload.get("team") or {}).get("id", "")
    slack_user = (payload.get("user") or {}).get("id", "")
    try:
        credential = await _owner(team_id, slack_user)
        if payload.get("type") == "block_suggestion":
            user = await get_user(credential.user_id)
            permitted = None
            for permission in ("read", "write", "delete"):
                available = {
                    str(dataset.id): dataset
                    for dataset in await get_all_user_permission_datasets(user, permission)
                }
                permitted = (
                    available
                    if permitted is None
                    else {key: value for key, value in permitted.items() if key in available}
                )
            query = (payload.get("value") or "").casefold()
            choices = sorted((permitted or {}).values(), key=lambda dataset: dataset.name)
            return {
                "options": [
                    {
                        "text": {"type": "plain_text", "text": dataset.name[:75]},
                        "value": str(dataset.id),
                    }
                    for dataset in choices
                    if query in dataset.name.casefold()
                ][:100]
            }
        if payload.get("type") == "view_submission":
            try:
                metadata, selection, keep_synced = _submission(payload)
            except (ValidationError, ValueError, KeyError):
                return {
                    "response_action": "errors",
                    "errors": {
                        "dataset": "Choose a dataset, channels or Slack thread links, and a valid number of days."
                    },
                }
            user = await get_user(credential.user_id)
            background_import(
                _import_and_confirm(metadata, selection, user=user, keep_synced=keep_synced)
            )
            return {"response_action": "clear"}
        thread = None
        if payload.get("callback_id") == THREAD_SHORTCUT:
            message = payload.get("message") or {}
            thread = {
                "channel_id": (payload.get("channel") or {}).get("id", ""),
                "ts": message.get("thread_ts") or message.get("ts", ""),
            }
        await _open_view(
            credential,
            payload.get("trigger_id", ""),
            history_view(
                team_id=team_id,
                response_url=payload.get("response_url", ""),
                thread=thread,
            ),
        )
    except SlackHistoryError as error:
        if payload.get("type") == "block_suggestion":
            return {"options": []}
        if payload.get("type") == "view_submission":
            return {"response_action": "errors", "errors": {"dataset": str(error)}}
        await post_to_response_url(
            payload.get("response_url", ""), {"response_type": "ephemeral", "text": str(error)}
        )
    return {}
