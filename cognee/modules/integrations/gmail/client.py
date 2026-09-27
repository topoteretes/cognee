"""Minimal Gmail REST client used by the core integration."""

from typing import Any

from cognee.modules.integrations.google.client import _authorized_get

GMAIL_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


async def get_profile(access_token: str) -> dict[str, Any]:
    return await _authorized_get("Gmail profile", f"{GMAIL_BASE_URL}/profile", access_token)


async def list_labels(access_token: str) -> dict[str, Any]:
    return await _authorized_get("Gmail label listing", f"{GMAIL_BASE_URL}/labels", access_token)


async def list_messages(
    access_token: str,
    *,
    page_token: str | None = None,
    label_ids: list[str] | None = None,
) -> dict[str, Any]:
    params: dict[str, str] = {"maxResults": "100"}
    if page_token:
        params["pageToken"] = page_token
    if label_ids:
        params["labelIds"] = ",".join(label_ids)
    return await _authorized_get(
        "Gmail message listing", f"{GMAIL_BASE_URL}/messages", access_token, params
    )


async def get_message(access_token: str, message_id: str) -> dict[str, Any]:
    return await _authorized_get(
        "Gmail message fetch",
        f"{GMAIL_BASE_URL}/messages/{message_id}",
        access_token,
        {"format": "full"},
    )


async def list_history(
    access_token: str,
    start_history_id: str,
    *,
    page_token: str | None = None,
) -> dict[str, Any]:
    params = {
        "startHistoryId": start_history_id,
        "historyTypes": "messageAdded,messageDeleted,labelAdded,labelRemoved",
    }
    if page_token:
        params["pageToken"] = page_token
    return await _authorized_get(
        "Gmail history listing", f"{GMAIL_BASE_URL}/history", access_token, params
    )
