"""Turn a LoCoMo conversation into the documents cognee ingests.

Every session is cut into windows of consecutive turns (6 by default). Each window starts with a
header carrying the session number, its date and the two speakers, so the date survives into the
chunk the window becomes: temporal questions ("when did X happen?") are only answerable when the
session date is next to the event text. A short overview document lists the speakers and the
session timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognee.eval_framework.benchmark_adapters.locomo_adapter import (
    LocomoConversation,
    LocomoSession,
    format_turn,
)

DEFAULT_WINDOW_TURNS = 6


@dataclass(frozen=True)
class SessionWindow:
    session_index: int
    date_time: str
    part: int
    parts: int
    first_dia_id: str
    last_dia_id: str
    text: str
    word_count: int


def window_header(
    conversation: LocomoConversation, session: LocomoSession, part: int, parts: int
) -> str:
    header = (
        f"Session {session.index} of the conversation between {conversation.speaker_a} and "
        f"{conversation.speaker_b}, which took place at {session.date_time}"
    )
    if parts > 1:
        header += f" (part {part} of {parts})"
    return header + "."


def build_session_windows(
    conversation: LocomoConversation,
    session: LocomoSession,
    window_turns: int = DEFAULT_WINDOW_TURNS,
) -> list[SessionWindow]:
    if window_turns < 1:
        raise ValueError("window_turns must be at least 1")
    turns = session.turns
    if not turns:
        return []

    groups = [turns[i : i + window_turns] for i in range(0, len(turns), window_turns)]
    # Avoid a dangling one- or two-turn tail: fold it into the previous window.
    if len(groups) > 1 and len(groups[-1]) < max(2, window_turns // 3):
        tail = groups.pop()
        groups[-1] = groups[-1] + tail

    windows: list[SessionWindow] = []
    parts = len(groups)
    for part, group in enumerate(groups, start=1):
        lines = [window_header(conversation, session, part, parts)]
        lines.extend(format_turn(turn) for turn in group)
        text = "\n".join(lines)
        windows.append(
            SessionWindow(
                session_index=session.index,
                date_time=session.date_time,
                part=part,
                parts=parts,
                first_dia_id=group[0].dia_id,
                last_dia_id=group[-1].dia_id,
                text=text,
                word_count=len(text.split()),
            )
        )
    return windows


def build_conversation_windows(
    conversation: LocomoConversation, window_turns: int = DEFAULT_WINDOW_TURNS
) -> list[SessionWindow]:
    """The windows of every session, in session order."""
    return [
        window
        for session in conversation.sessions
        for window in build_session_windows(conversation, session, window_turns)
    ]


def conversation_overview_text(conversation: LocomoConversation) -> str:
    """Short document: who talked, and when each session happened."""
    lines = [
        (
            f"{conversation.speaker_a} and {conversation.speaker_b} are friends who talked in "
            f"{len(conversation.sessions)} conversation sessions over time "
            f"(conversation id {conversation.sample_id})."
        ),
        "Session timeline:",
    ]
    for session in conversation.sessions:
        lines.append(
            f"- Session {session.index} took place at {session.date_time} "
            f"({len(session.turns)} turns)."
        )
    return "\n".join(lines)


def dataset_name_for(conversation: LocomoConversation) -> str:
    return f"locomo_{conversation.sample_id.replace('-', '_')}"
