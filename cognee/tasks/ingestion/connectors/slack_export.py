"""Slack workspace export → dlt source for Cognee ingestion.

Parses the standard Slack export layout (channels.json, users.json, and
per-channel daily JSON message files) and yields flat message rows suitable
for DLT ingestion with ``primary_key="id"``.

Each message row uses ``id = "{channel_id}:{ts}"`` so re-syncs and orphan
cleanup can track individual messages across export snapshots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    import dlt
except ImportError:
    dlt = None


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_user_lookup(users: List[dict]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for user in users:
        user_id = user.get("id")
        if not user_id:
            continue
        display = (
            user.get("real_name")
            or user.get("profile", {}).get("real_name")
            or user.get("name")
            or user_id
        )
        lookup[user_id] = display
    return lookup


def _build_channel_lookup(channels: List[dict]) -> Dict[str, dict]:
    """Map channel folder name (``name`` field) → channel metadata."""
    lookup: Dict[str, dict] = {}
    for channel in channels:
        name = channel.get("name")
        if name:
            lookup[name] = channel
    return lookup


def _message_text(message: dict, user_name: Optional[str], channel_name: str) -> str:
    """Build searchable text, folding thread context when present."""
    raw_text = message.get("text") or ""
    parts: List[str] = []

    thread_ts = message.get("thread_ts")
    msg_ts = message.get("ts")
    if thread_ts and msg_ts and thread_ts != msg_ts:
        parts.append(f"[thread reply in #{channel_name}]")

    if user_name:
        parts.append(f"{user_name}:")
    parts.append(raw_text)
    return " ".join(part for part in parts if part).strip()


def iter_slack_export_messages(export_path: str | os.PathLike) -> Iterator[dict]:
    """Yield flat Slack message dicts from a workspace export directory.

    Skips non-message events (joins, topic changes, etc.) and messages
    without a ``ts`` timestamp.
    """
    root = Path(export_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Slack export directory not found: {root}")

    channels_path = root / "channels.json"
    users_path = root / "users.json"
    if not channels_path.is_file():
        raise FileNotFoundError(f"Missing channels.json in Slack export: {root}")

    channels = _load_json(channels_path)
    users = _load_json(users_path) if users_path.is_file() else []
    channel_lookup = _build_channel_lookup(channels)
    user_lookup = _build_user_lookup(users)

    for channel_dir in sorted(root.iterdir()):
        if not channel_dir.is_dir():
            continue

        channel_meta = channel_lookup.get(channel_dir.name)
        if channel_meta is None:
            continue

        channel_id = channel_meta.get("id", channel_dir.name)
        channel_name = channel_meta.get("name", channel_dir.name)

        for message_file in sorted(channel_dir.glob("*.json")):
            day_messages = _load_json(message_file)
            if not isinstance(day_messages, list):
                continue

            for message in day_messages:
                if not isinstance(message, dict):
                    continue
                if message.get("type") != "message":
                    continue

                ts = message.get("ts")
                if not ts:
                    continue

                user_id = message.get("user") or message.get("bot_id")
                user_name = user_lookup.get(user_id) if user_id else None
                text = _message_text(message, user_name, channel_name)
                ts_no_dot = ts.replace(".", "")

                yield {
                    "id": f"{channel_id}:{ts}",
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "ts": ts,
                    "thread_ts": message.get("thread_ts"),
                    "user_id": user_id,
                    "user_name": user_name,
                    "text": text,
                    "subtype": message.get("subtype"),
                    # Reconstructable permalink for citable recall (#3604).
                    "slack_permalink": f"https://slack.com/archives/{channel_id}/p{ts_no_dot}",
                }


def slack_export_source(export_path: str | os.PathLike):
    """Return a dlt resource over a Slack workspace export directory."""
    if dlt is None:
        raise ImportError(
            "The 'dlt' package is required for slack_export_source. "
            "Install cognee with dlt extras or `pip install dlt`."
        )

    path = str(export_path)

    @dlt.resource(name="slack_messages", write_disposition="replace")
    def slack_messages():
        yield from iter_slack_export_messages(path)

    return slack_messages()
