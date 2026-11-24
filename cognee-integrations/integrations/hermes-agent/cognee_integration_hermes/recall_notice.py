"""A one-time note that recall stays on the configured dataset, and that auto exists (SDK-1393).

A configured dataset other than the default keeps recall pinned to it while no
recall mode is set (``config.effective_recall_mode``). Such a user is told once
per machine and dataset that Cognee can also pick the dataset for each prompt.
The note is appended to the first reply of a session, by the same
``transform_llm_output`` hook that draws the memory credit, and logged once.

The provider queues it (:func:`queue`) once its session is connected; the hook
takes it (:func:`append_pending`) and writes the marker, so a session that never
answers does not use it up. It is cut from the answer before the turn is stored
(:func:`strip_notice`).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading

from .config import SHARED_PLUGIN_STATE_DIR

logger = logging.getLogger(__name__)

#: One marker file per dataset the note was shown for.
NOTICE_DIR = SHARED_PLUGIN_STATE_DIR / "hermes" / "notices"

_lock = threading.Lock()
#: Session id -> the dataset whose note waits for that session's first reply.
_pending: dict[str, str] = {}

_NOTICE_RE = re.compile(
    r"\n*^Cognee: recall stays on `[^`\n]*`\. Cognee can also pick the best-matching "
    r"dataset[^\n]*for this conversation\.[ \t]*$",
    re.M,
)


def notice_text(dataset: str) -> str:
    return (
        f"Cognee: recall stays on `{dataset}`. Cognee can also pick the best-matching dataset "
        "for each prompt from all you can read: set `recall_mode: auto` in cognee.json (or "
        "`COGNEE_PLUGIN_RECALL_MODE=auto`, or rerun `hermes memory setup cognee`), or ask to "
        "set recall to auto for this conversation."
    )


def _marker(dataset: str):
    digest = hashlib.sha1(str(dataset).encode("utf-8")).hexdigest()[:16]
    return NOTICE_DIR / f"recall-pinned-{digest}"


def due(dataset: str) -> bool:
    """True until the note was shown for ``dataset`` on this machine."""
    try:
        return not _marker(dataset).exists()
    except OSError:
        return False


def _mark_shown(dataset: str) -> None:
    try:
        NOTICE_DIR.mkdir(parents=True, exist_ok=True)
        _marker(dataset).write_text(f"{dataset}\n", encoding="utf-8")
    except OSError:
        pass


def queue(session_id: str, dataset: str) -> None:
    """Hold the note for ``session_id``'s next reply."""
    with _lock:
        _pending[str(session_id or "")] = dataset


def _take(session_id: str) -> str:
    with _lock:
        key = str(session_id or "")
        if key in _pending:
            return _pending.pop(key)
        # The hook's session id may not match the provider's (or be missing):
        # the note is per machine, so any waiting one will do.
        if _pending:
            return _pending.pop(next(iter(_pending)))
        return ""


def append_pending(response_text: str, session_id: str = "") -> str | None:
    """``response_text`` with a waiting note appended once, or None when none is due.

    Append-only, after a blank line: the terminal chats keep the streamed text
    and show only what is added after it (see memory_credit.render_credit).
    """
    if not response_text or not response_text.strip():
        return None
    dataset = _take(session_id)
    if not dataset or not due(dataset):
        return None
    _mark_shown(dataset)
    text = notice_text(dataset)
    logger.info("%s", text)
    return f"{response_text.rstrip()}\n\n{text}"


def strip_notice(answer: str) -> str:
    """The answer without the note, for storing."""
    if not answer or "Cognee: recall stays on" not in answer:
        return answer
    return _NOTICE_RE.sub("", answer).rstrip()
