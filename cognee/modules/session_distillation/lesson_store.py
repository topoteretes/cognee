"""Typed lesson documents of the coding distillation (``DISTILLATION_MODE=coding``).

A coding lesson is one document with two header lines, a title and the statement::

    Lesson-Type: pitfall
    State-Key: pitfall:Edit:no-changes-to-make:py      (state and pitfall lessons only)
    # Session learning (session <session_id>)

    <statement> (seen in N sessions)                   (pitfalls only)

Its ``Data`` row carries the same facts in ``external_metadata`` (the column the
dataset listing exposes, which is how the Claude Code plugin reads them, SDK-1336):
``lesson_type``, ``state_key``, ``session_id``, ``session_date``, ``git_commit``,
``git_branch``, ``files`` and ``seen_in_sessions``. A state key names one current fact,
so once a new lesson with it is stored, every older row with the same key in the
dataset is deleted.
"""

import re
from dataclasses import dataclass
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.data.models import Data
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


async def find_lessons_by_state_key(dataset_id: UUID, state_key: str) -> list[Data]:
    """Every ``Data`` row of the dataset whose lesson metadata carries ``state_key``."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine

    query = select(Data).where(
        Data.dataset_id == dataset_id,
        Data.external_metadata["state_key"].as_string() == state_key,
    )
    async with get_relational_engine().get_async_session() as session:
        return list((await session.execute(query)).scalars().all())


def seen_in_sessions(rows: list[Data], session_id: str) -> int:
    """One more than the highest count another session stored for the key; 1 when none.

    The same session distilled twice computes the same number, since its own row is
    left out, so re-distilling never inflates the count.
    """
    highest = 0
    for row in rows:
        metadata = row.external_metadata if isinstance(row.external_metadata, dict) else {}
        if metadata.get("session_id") == session_id:
            continue
        try:
            highest = max(highest, int(metadata.get("seen_in_sessions") or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1


async def delete_superseded_lessons(
    *, dataset_id: UUID, state_key: str, keep: set[UUID], user
) -> int:
    """Delete the dataset's rows with ``state_key`` except ``keep``; returns how many."""
    from cognee.api.v1.datasets.datasets import datasets

    removed = 0
    for row in await find_lessons_by_state_key(dataset_id, state_key):
        if row.id in keep:
            continue
        receipt = await datasets.delete_data(
            dataset_id=dataset_id, data_id=row.id, user=user, delete_dataset_if_empty=False
        )
        if receipt.get("status") == "success":
            removed += 1
    if removed:
        logger.info(
            "Coding distillation: replaced %d older lesson(s) with state key '%s'",
            removed,
            state_key,
        )
    return removed
