"""Typed lesson documents of the coding distillation (``DISTILLATION_MODE=coding``).

A coding lesson is one document with two header lines, a title and the statement::

    Lesson-Type: pitfall
    State-Key: pitfall:Edit:no-changes-to-make:py      (state and pitfall lessons only)
    # Session learning (session <session_id>)

    <statement> (seen in N sessions)                   (pitfalls only)

Its ``Data`` row carries the same facts in ``external_metadata`` (the column the
dataset listing exposes, which is how the Claude Code plugin reads them, SDK-1336):
``lesson_type``, ``state_key``, ``session_id``, ``session_date``, ``git_commit``,
``git_branch``, ``files``, ``project_node_sets``, ``seen_sessions`` and ``seen_in_sessions``.

A state key names one current fact *per project*: the lessons of one dataset are
scoped by the project node sets their rows carry (the session's pinned node set, which
the plugin sets per repository), and a key is replaced only within that scope. Among the
lessons with one key the newest session date wins, so a session distilled late never
displaces what a later session stored; without dates on both sides the lesson stored
last wins.
"""

import re
from dataclasses import dataclass
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.data.models import Data
from cognee.modules.engine.models.node_set import parse_node_set_names
from cognee.modules.improve.constants import SESSION_LEARNINGS_NODE_SET
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_distillation.lesson_store")

LESSON_TYPE_HEADER = "Lesson-Type:"
STATE_KEY_HEADER = "State-Key:"

# A path-like token with an extension: ``src/app.py``, ``README.md``, ``.env.template``.
_FILE_TOKEN_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*\.?[\w-]+\.[A-Za-z][A-Za-z0-9]{0,5}(?![\w/])"
)
_NOT_FILES = {"e.g", "i.e", "etc", "vs"}


@dataclass(frozen=True, slots=True)
class LessonDocument:
    """One rendered coding lesson with the metadata its ``Data`` row carries."""

    text: str
    metadata: dict
    data_id: UUID


def lesson_data_id(dataset_id, text: str) -> UUID:
    """Deterministic row id: the same lesson text in one dataset is one row."""
    return uuid5(NAMESPACE_OID, f"session-lesson:{dataset_id}:{text}")


def mentioned_files(statement: str) -> list[str]:
    """File names the statement names, in their original casing, in order, deduplicated."""
    found: list[str] = []
    for token in _FILE_TOKEN_RE.findall(statement or ""):
        stem = token.rsplit(".", 1)[0].rsplit("/", 1)[-1]
        if token.lower().rstrip(".") in _NOT_FILES or not re.search(r"[A-Za-z]", stem):
            continue
        if token not in found:
            found.append(token)
    return found


def render_coding_lesson_document(
    statement: str,
    *,
    session_id: str,
    lesson_type: str,
    state_key: str = "",
    seen_in_sessions: int | None = None,
) -> str:
    headers = [f"{LESSON_TYPE_HEADER} {lesson_type}"]
    if state_key:
        headers.append(f"{STATE_KEY_HEADER} {state_key}")
    body = statement.strip()
    if lesson_type == "pitfall" and seen_in_sessions:
        body = f"{body} (seen in {seen_in_sessions} sessions)"
    return "\n".join(headers) + f"\n# Session learning (session {session_id})\n\n{body}\n"


# -- Scope ---------------------------------------------------------------------------


def project_node_sets(node_set) -> tuple[str, ...]:
    """The node sets that scope a lesson to a project: all but the distillation's own.

    ``node_set`` is a stored ``Data.node_set`` value or a list of names. The
    ``session_learnings`` set and the per-session ``session_learnings:<id>`` sets are
    on every lesson, so they say nothing about the project; what remains is the
    session's pinned node set (the plugin's project), empty when there is none.
    """
    names = node_set if isinstance(node_set, (list, tuple)) else parse_node_set_names(node_set)
    return tuple(
        sorted(
            {
                name
                for name in names
                if name != SESSION_LEARNINGS_NODE_SET
                and not name.startswith(f"{SESSION_LEARNINGS_NODE_SET}:")
            }
        )
    )


def _metadata(row) -> dict:
    metadata = getattr(row, "external_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def lessons_in_scope(rows: list, node_set) -> list:
    """The coding lessons among ``rows`` whose project scope is ``node_set``'s.

    A row is a coding lesson when its metadata carries ``lesson_type``; a document a
    user added with a ``state_key`` of their own is never touched.
    """
    scope = project_node_sets(node_set)
    return [
        row
        for row in rows
        if _metadata(row).get("lesson_type")
        and project_node_sets(getattr(row, "node_set", None)) == scope
    ]


async def find_lessons_by_state_key(dataset_id: UUID, state_key: str, node_set) -> list[Data]:
    """The dataset's coding lessons with ``state_key`` in the project scope of ``node_set``."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine

    query = select(Data).where(
        Data.dataset_id == dataset_id,
        Data.external_metadata["state_key"].as_string() == state_key,
    )
    async with get_relational_engine().get_async_session() as session:
        rows = list((await session.execute(query)).scalars().all())
    return lessons_in_scope(rows, node_set)


# -- Pitfall counts ------------------------------------------------------------------


def pitfall_sessions(rows: list, session_id: str) -> list[str]:
    """Every session that stored the key, in first-seen order, ``session_id`` included once.

    Each row carries the sessions it counted (``seen_sessions``; a row written before
    the list existed counts as its own ``session_id``), so the count survives the
    replacement of older rows and re-distilling a session never changes it.
    """
    sessions: list[str] = []
    for row in rows:
        metadata = _metadata(row)
        seen = metadata.get("seen_sessions")
        if not isinstance(seen, list):
            seen = [metadata.get("session_id")]
        for seen_session in seen:
            if isinstance(seen_session, str) and seen_session and seen_session not in sessions:
                sessions.append(seen_session)
    if session_id not in sessions:
        sessions.append(session_id)
    return sessions


# -- Replacement ---------------------------------------------------------------------


def _session_date(row) -> str:
    return str(_metadata(row).get("session_date") or "")


def rows_to_delete(existing: list, stored: dict[UUID, str]) -> list:
    """The rows a state key loses once ``stored`` (``{row id: session date}``) is in.

    The newest session date wins: when an existing row of another session is dated
    later than the lesson just stored, that row stays and the just-stored rows go,
    together with every other row; otherwise the just-stored rows stay and every
    other row goes. Without a date on both sides the just-stored rows win.
    """
    stored_date = max(stored.values(), default="")
    others = [row for row in existing if row.id not in stored]
    newer = [row for row in others if stored_date and _session_date(row) > stored_date]
    if newer:
        newest_date = max(_session_date(row) for row in newer)
        keep = {row.id for row in newer if _session_date(row) == newest_date}
        return [row for row in existing if row.id not in keep]
    return others


async def replace_superseded_lessons(
    *, dataset_id: UUID, state_key: str, node_set, stored: dict[UUID, str], user
) -> int:
    """Leave the project scope one lesson with ``state_key``; returns how many were deleted.

    ``stored`` maps the ids just stored for the key to their session date. The rows
    just stored are part of the dataset by now, so they are among the rows found and
    ``rows_to_delete`` decides between them and the older ones.
    """
    from cognee.api.v1.datasets.datasets import datasets

    existing = await find_lessons_by_state_key(dataset_id, state_key, node_set)
    losers = rows_to_delete(existing, stored)
    removed = 0
    for row in losers:
        receipt = await datasets.delete_data(
            dataset_id=dataset_id, data_id=row.id, user=user, delete_dataset_if_empty=False
        )
        if receipt.get("status") == "success":
            removed += 1
    discarded_new = [row for row in losers if row.id in stored]
    if discarded_new:
        logger.info(
            "Coding distillation: a newer lesson with state key '%s' is already stored; "
            "the lesson of this older session was discarded",
            state_key,
        )
    elif removed:
        logger.info(
            "Coding distillation: replaced %d older lesson(s) with state key '%s'",
            removed,
            state_key,
        )
    return removed
