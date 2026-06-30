"""Map a Telegram conversation to a cognee dataset + session.

Each Telegram chat is one memory boundary. The **dataset** is the durable
store and the unit ``/forget`` clears; the **session_id** is an optional
freshness cache so a just-sent message is answerable before the background
graph sync finishes.

Convention (per-chat by default)::

    DM (private)   dataset telegram_dm_<user_id>             session telegram:dm:<user_id>
    group / super  dataset telegram_group_<chat_id>          session telegram:group:<chat_id>
    forum topic    dataset telegram_group_<chat_id>_<thread> session telegram:group:<chat_id>:<thread>

With ``per_user_in_group=True`` a group is split per sender
(``telegram_group_<chat_id>_user_<user_id>``) so hard per-user deletion works.
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
    session_id: str
    chat_id: int
    thread_id: int | None = None
    is_private: bool = False


def resolve_scope(
    *,
    chat_type: str,
    chat_id: int,
    user_id: int,
    thread_id: int | None = None,
    per_user_in_group: bool = False,
) -> Scope:
    """Resolve a Telegram chat into its ``Scope`` (dataset + session).

    Args:
        chat_type: Telegram ``chat.type`` (``private`` / ``group`` / ``supergroup``).
        chat_id: Telegram ``chat.id``.
        user_id: Sender ``user.id`` (used for DMs and per-user group scoping).
        thread_id: Forum-topic ``message_thread_id`` when present.
        per_user_in_group: Split group memory per sender when True.

    Returns:
        The ``Scope`` describing this conversation's dataset and session id.
    """
    is_private = chat_type == PRIVATE
    if is_private:
        return Scope(
            dataset_name=f"telegram_dm_{_sanitize(user_id)}",
            session_id=f"telegram:dm:{user_id}",
            chat_id=chat_id,
            thread_id=None,
            is_private=True,
        )

    base_ds = f"telegram_group_{_sanitize(chat_id)}"
    base_sid = f"telegram:group:{chat_id}"
    if thread_id is not None:
        base_ds += f"_{_sanitize(thread_id)}"
        base_sid += f":{thread_id}"
    if per_user_in_group:
        base_ds += f"_user_{_sanitize(user_id)}"
        base_sid += f":user:{user_id}"

    return Scope(
        dataset_name=base_ds,
        session_id=base_sid,
        chat_id=chat_id,
        thread_id=thread_id,
        is_private=False,
    )
