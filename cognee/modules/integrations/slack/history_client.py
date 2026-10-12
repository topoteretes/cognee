"""Bounded Slack API reads, with complete pagination and no arbitrary URL fetching."""

import asyncio
from decimal import Decimal
from typing import Any

import aiohttp

from cognee.modules.integrations.slack.history_models import TIMESTAMP, SlackHistoryError


class SlackAPIError(SlackHistoryError):
    def __init__(self, method: str, code: str):
        self.code = code
        super().__init__(f"Slack {method}: {code}. Check app scopes and channel membership.")


class SlackHistoryClient:
    def __init__(self, token: str, *, max_requests: int = 1000, max_messages: int = 50000):
        if not token:
            raise SlackHistoryError("The Slack connection has no access token. Reconnect Slack.")
        self._token = token
        self.max_requests = max_requests
        self.max_messages = max_messages
        self.requests = 0
        self.messages = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def call(self, method: str, **params) -> dict[str, Any]:
        for attempt in range(4):
            self.requests += 1
            if self.requests > self.max_requests:
                raise SlackHistoryError(
                    "Slack request limit reached. Narrow the selection or raise max_requests."
                )
            try:
                async with self.session.get(
                    f"https://slack.com/api/{method}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    params=params,
                    allow_redirects=False,
                ) as response:
                    status = response.status
                    retry_after = response.headers.get("Retry-After", "1")
                    payload = await response.json() if status < 500 and status != 429 else {}
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                if attempt == 3:
                    raise SlackHistoryError(
                        "Slack could not be reached. Retry the import."
                    ) from None
                await asyncio.sleep(2**attempt)
                continue
            if status == 429 or payload.get("error") == "ratelimited" or status >= 500:
                if attempt == 3:
                    raise SlackHistoryError(
                        "Slack is temporarily rate limited or unavailable. Retry later."
                    )
                try:
                    delay = max(1, int(retry_after)) if status == 429 else 2**attempt
                except ValueError:
                    delay = 2**attempt
                # Never retry earlier than Retry-After; excessive waits fail explicitly.
                if delay > 300:
                    raise SlackHistoryError(f"Slack requested a {delay}s wait. Retry later.")
                await asyncio.sleep(delay)
                continue
            if status != 200 or not payload.get("ok"):
                code = payload.get("error", "request_failed")
                # Slack error identifiers, never response bodies or echoed credentials.
                code = (
                    code
                    if isinstance(code, str) and code.replace("_", "").isalnum()
                    else "request_failed"
                )
                raise SlackAPIError(method, code)
            return payload
        raise SlackHistoryError("Slack request failed.")

    async def pages(self, method: str, key: str, **params) -> list[dict]:
        items = []
        cursor = ""
        seen = set()
        while True:
            page = await self.call(
                method, limit=200, **params, **({"cursor": cursor} if cursor else {})
            )
            batch = page.get(key)
            if not isinstance(batch, list):
                raise SlackHistoryError("Slack returned an incomplete page. Nothing was imported.")
            items.extend(batch)
            if key == "messages":
                self.messages += len(batch)
                if self.messages > self.max_messages:
                    raise SlackHistoryError(
                        "Slack message limit reached. Narrow the selection or raise max_messages."
                    )
            cursor = ((page.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                if page.get("has_more"):
                    raise SlackHistoryError("Slack omitted a pagination cursor. Retry the import.")
                return items
            if cursor in seen:
                raise SlackHistoryError("Slack repeated a pagination cursor. Retry the import.")
            seen.add(cursor)

    async def authorize_channel(self, channel: str, slack_user: str) -> dict:
        info = (await self.call("conversations.info", channel=channel)).get("channel") or {}
        if info.get("id") != channel or info.get("is_im") or info.get("is_mpim"):
            raise SlackHistoryError(
                "History imports currently support public and private channels only."
            )
        if not info.get("is_member"):
            raise SlackHistoryError(
                "Invite the Cognee app to each selected channel before importing."
            )
        # The installation's bot may see private channels its installer cannot.
        # Account linking alone is therefore insufficient authorization to export.
        members = await self.pages("conversations.members", "members", channel=channel)
        if not slack_user or slack_user not in members:
            raise SlackHistoryError(
                "The connecting Slack user must be a member of every selected channel."
            )
        return info

    async def history(self, channel: str, *, oldest: str, latest: str) -> list[dict]:
        return await self.pages(
            "conversations.history",
            "messages",
            channel=channel,
            oldest=oldest,
            latest=latest,
            inclusive="true",
        )

    async def thread(self, channel: str, ts: str) -> list[dict]:
        # Resolve a permalink to a reply to its actual root (no reliance on the
        # presence of thread_ts in the URL). conversations.history can get one
        # message even when the caller copied a reply's own permalink.
        found = await self.call(
            "conversations.history",
            channel=channel,
            oldest=ts,
            latest=ts,
            inclusive="true",
            limit=1,
        )
        messages = found.get("messages") or []
        if messages and messages[0].get("ts") == ts:
            ts = messages[0].get("thread_ts") or ts
        replies = await self.pages("conversations.replies", "messages", channel=channel, ts=ts)
        normalized = {}
        for message in replies:
            stamp = message.get("ts", "")
            if not isinstance(stamp, str) or not TIMESTAMP.fullmatch(stamp):
                raise SlackHistoryError("Slack returned a message without a valid timestamp.")
            normalized[stamp] = message
        return sorted(normalized.values(), key=lambda message: Decimal(message["ts"]))
