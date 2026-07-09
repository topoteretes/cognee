"""Slack workspace export connector for cognee — a ``dlt`` source over a Slack
export archive.

Parses the standard Slack export layout (``channels.json``, ``users.json``, and
per-channel daily JSON message files) into flat message rows and hands them
straight to :func:`cognee.remember`, reusing the existing DLT ingestion path
(``resolve_dlt_sources`` -> ``ingest_dlt_source`` -> ``orphan_cleanup``)::

    import cognee
    from cognee.tasks.ingestion.connectors import slack_export_source

    await cognee.remember(
        slack_export_source("/path/to/slack-export"),
        dataset_name="team-slack-export",
        max_rows_per_table=0,   # ingest the whole export (see .. note:: below)
    )

Design
------
* **Primary key** — ``id = "{channel_id}:{ts}"``. Slack's ``ts`` is the stable,
  per-channel message id; prefixing the channel id disambiguates it across
  channels. A message keeps its id across export snapshots, so re-syncs and
  orphan cleanup can track it.
* **Snapshot sync (``replace``)** — a Slack export is a *full snapshot*, not a
  live incremental feed, so the resource uses ``write_disposition="replace"``
  (also cognee's dlt default). Each sync drops + reloads the table, so the rows
  read back reflect only the current snapshot; a message removed upstream falls
  out of that set and cognee's ``orphan_cleanup`` purges it from the graph +
  vector + relational stores. (Contrast the Gmail connector, which is a live
  feed and uses ``merge`` + a hard-delete marker.)
* **Incremental cognify** — message ids/content are stable, so re-ingesting a
  later snapshot only re-cognifies new/changed messages; keep the default
  ``incremental_loading=True``. ``merge`` would be wrong here: it never removes
  rows absent from the new load, so deletions would not propagate.

.. note::
   cognee's ``ingest_dlt_source`` reads at most ``max_rows_per_table`` rows from
   the dlt destination (default 50). For a real export pass
   ``max_rows_per_table=0`` (unlimited) so orphan cleanup compares against the
   *whole* snapshot rather than a truncated window. Use a dedicated
   ``dataset_name`` per workspace so cleanup only touches this export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("slack_export_connector")

# Slack membership/config notices arrive as ``type == "message"`` events
# distinguished only by their ``subtype`` (e.g. "<@U> has joined the channel").
# They are not conversation content, so they are skipped during parsing.
_SYSTEM_MESSAGE_SUBTYPES = frozenset(
    {
        "channel_join",
        "channel_leave",
        "channel_topic",
        "channel_purpose",
        "channel_name",
        "channel_archive",
        "channel_unarchive",
        "group_join",
        "group_leave",
        "group_topic",
        "group_purpose",
        "group_name",
        "group_archive",
        "group_unarchive",
        "pinned_item",
        "unpinned_item",
    }
)


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

    Skips non-message events, system/membership notices (joins, topic
    changes, etc. — see ``_SYSTEM_MESSAGE_SUBTYPES``), and messages without
    a ``ts`` timestamp.
    """
    root = Path(export_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Slack export directory not found: {root}")

    channels_path = root / "channels.json"
    users_path = root / "users.json"
    if not channels_path.is_file():
        raise FileNotFoundError(f"Missing channels.json in Slack export: {root}")

    channels = _load_json(channels_path)
    if not isinstance(channels, list):
        raise ValueError(f"channels.json must be a JSON array of channels: {channels_path}")
    users = _load_json(users_path) if users_path.is_file() else []
    if not isinstance(users, list):
        users = []
    channel_lookup = _build_channel_lookup(channels)
    user_lookup = _build_user_lookup(users)

    yielded = 0
    for channel_dir in sorted(root.iterdir()):
        if not channel_dir.is_dir():
            continue

        channel_meta = channel_lookup.get(channel_dir.name)
        if channel_meta is None:
            # A directory with no matching entry in channels.json (e.g. a
            # private channel from groups.json, a DM, or a non-channel folder).
            logger.debug(
                "Slack export: skipping directory not in channels.json: %s", channel_dir.name
            )
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
                if message.get("subtype") in _SYSTEM_MESSAGE_SUBTYPES:
                    continue

                ts = message.get("ts")
                if not ts:
                    continue

                user_id = message.get("user") or message.get("bot_id")
                user_name = user_lookup.get(user_id) if user_id else None
                text = _message_text(message, user_name, channel_name)

                yielded += 1
                yield {
                    "id": f"{channel_id}:{ts}",
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "ts": ts,
                    "thread_ts": message.get("thread_ts"),
                    "user_id": user_id,
                    "user_name": user_name,
                    "text": text,
                }

    logger.info("Slack export %s: parsed %d message(s).", root, yielded)


def slack_export_source(export_path: str | os.PathLike):
    """Return a ``dlt`` resource over a Slack workspace export for ``remember``.

    Args:
        export_path: Path to the unpacked Slack export directory (the one that
            contains ``channels.json`` and the per-channel folders).

    Returns:
        A ``dlt`` resource (``slack_messages``) with ``write_disposition="replace"``.
        Hand it to ``cognee.remember(...)``; see the module docstring for the
        recommended ``max_rows_per_table=0`` / dedicated-dataset settings.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            'The Slack export connector requires the dlt extra: pip install "cognee[dlt]".'
        ) from exc

    path = str(export_path)

    @dlt.resource(name="slack_messages", write_disposition="replace")
    def slack_messages():
        yield from iter_slack_export_messages(path)

    return slack_messages
