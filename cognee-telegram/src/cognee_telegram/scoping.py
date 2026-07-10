"""Map a Telegram conversation to a cognee dataset.

Each Telegram chat is one memory boundary: the **dataset** is the durable
store and the unit ``/forget`` clears.

Convention::

    DM (private)   dataset telegram_dm_<user_id>
    group / super  dataset telegram_group_<chat_id>
    forum topic    dataset telegram_group_<chat_id>_<thread>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PRIVATE = "private"
GROUP = "group"
SUPERGROUP = "supergroup"

_SANITIZE = re.compile(r"[^0-9a-zA-Z]+")


def _sanitize(value: object) -> str:
    """Make a Telegram id safe for a cognee dataset name.

    Telegram group ids are negative (e.g. ``-1001234567890``); the leading
    ``-`` is encoded as ``n`` so positive and negative ids never collide and
    the name stays alphanumeric + underscore.
    """
    text = str(value)
    text = text.replace("-", "n")
    return _SANITIZE.sub("_", text)


@dataclass(frozen=True)
class Scope:
    """The resolved memory boundary for one Telegram conversation."""

    dataset_name: str
    chat_id: int
    thread_id: int | None = None


def resolve_scope(
    *,
    chat_type: str,
    chat_id: int,
    user_id: int,
    thread_id: int | None = None,
) -> Scope:
    """Resolve a Telegram chat into its ``Scope`` (dataset).

    Args:
        chat_type: Telegram ``chat.type`` (``private`` / ``group`` / ``supergroup``).
        chat_id: Telegram ``chat.id``.
        user_id: Sender ``user.id`` (used to scope DMs to the user).
        thread_id: Forum-topic ``message_thread_id`` when present.

    Returns:
        The ``Scope`` describing this conversation's dataset.
    """
    if chat_type == PRIVATE:
        return Scope(
            dataset_name=f"telegram_dm_{_sanitize(user_id)}",
            chat_id=chat_id,
            thread_id=None,
        )

    dataset_name = f"telegram_group_{_sanitize(chat_id)}"
    if thread_id is not None:
        dataset_name += f"_{_sanitize(thread_id)}"

    return Scope(dataset_name=dataset_name, chat_id=chat_id, thread_id=thread_id)
