"""Turn a LoCoMo conversation into ingestion-ready session windows.

LoCoMo sessions are dated chats between two people. The memory pipeline stores one
*window* of consecutive turns per session-memory entry. Every window starts with a header
carrying the session number, its date, and the two speakers, so the date survives into
whichever chunk the window lands in — temporal questions ("when did X happen?") are only
answerable when the session date is next to the event text.

Also writes the same windows as JSON-list files (one per session), the representation
``JsonListChunker`` reads, for anyone who wants the BEAM-style ``add``/``cognify`` path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognee.eval_framework.benchmark_adapters.locomo_adapter import (
    LocomoAdapter,
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


def conversation_overview_text(conversation: LocomoConversation) -> str:
    """Short permanent-memory document: who talked, and when each session happened."""
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


def session_id_for(conversation: LocomoConversation, session: LocomoSession) -> str:
    return f"locomo_{conversation.sample_id.replace('-', '_')}_s{session.index:02d}"


def dataset_name_for(conversation: LocomoConversation) -> str:
    return f"locomo_{conversation.sample_id.replace('-', '_')}"


def build_conversation_bundle(
    conversation: LocomoConversation, window_turns: int = DEFAULT_WINDOW_TURNS
) -> dict[str, Any]:
    """Everything ingestion needs for one conversation, as plain data."""
    sessions = []
    for session in conversation.sessions:
        windows = build_session_windows(conversation, session, window_turns)
        sessions.append(
            {
                "session_index": session.index,
                "session_id": session_id_for(conversation, session),
                "date_time": session.date_time,
                "turn_count": len(session.turns),
                "windows": [window.__dict__ for window in windows],
            }
        )
    return {
        "conversation_index": conversation.conversation_index,
        "sample_id": conversation.sample_id,
        "dataset_name": dataset_name_for(conversation),
        "speaker_a": conversation.speaker_a,
        "speaker_b": conversation.speaker_b,
        "overview": conversation_overview_text(conversation),
        "window_turns": window_turns,
        "sessions": sessions,
    }


def write_conversation_files(bundle: dict[str, Any], output_dir: Path) -> Path:
    """Write ``session_NN.json`` JSON-list files plus a manifest; returns the folder."""
    folder = output_dir / f"conv_{bundle['conversation_index']:02d}_{bundle['sample_id']}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_sessions = []
    for session in bundle["sessions"]:
        path = folder / f"session_{session['session_index']:02d}.json"
        items = [window["text"] for window in session["windows"]]
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_sessions.append(
            {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "date_time": session["date_time"],
                "turn_count": session["turn_count"],
                "window_count": len(items),
                "max_window_words": max((w["word_count"] for w in session["windows"]), default=0),
                "file": path.name,
            }
        )
    (folder / "overview.txt").write_text(bundle["overview"] + "\n", encoding="utf-8")
    manifest = {key: value for key, value in bundle.items() if key != "sessions"}
    manifest["sessions"] = manifest_sessions
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return folder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None, help="Path to locomo10.json")
    parser.add_argument("--conversation-index", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--window-turns", type=int, default=DEFAULT_WINDOW_TURNS)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("temp") / "locomo_preprocessed_documents"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    adapter = LocomoAdapter(
        conversation_index=args.conversation_index,
        max_sessions=args.max_sessions,
        data_path=args.data_path,
    )
    for conversation in adapter.iter_conversations():
        bundle = build_conversation_bundle(conversation, args.window_turns)
        folder = write_conversation_files(bundle, args.output_dir)
        windows = sum(len(session["windows"]) for session in bundle["sessions"])
        print(
            f"{bundle['sample_id']}: {len(bundle['sessions'])} sessions, "
            f"{conversation.turn_count} turns, {windows} windows -> {folder}"
        )


if __name__ == "__main__":
    main()
