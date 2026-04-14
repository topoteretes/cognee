"""Chunker that preserves BEAM conversation structure.

The input is expected to look like the flattened BEAM transcripts:
    --- Session 1 ---
    [time_anchor] User: ...
    [time_anchor] Assistant: ...

Chunks are built around conversation turns instead of paragraphs. Complete
user-assistant pairs are preferred, but incomplete standalone turns are still
kept with explicit metadata so they can be handled downstream.
"""

import re
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

_TURN_START = re.compile(r"^(?:\[.*?\]\s*)?(?P<role>User|Assistant):\s?")
_ROLE_PREFIX = re.compile(r"^(?P<prefix>(?:\[.*?\]\s*)?(?:User|Assistant):\s?)(?P<content>.*)$")
_SESSION_MARKER = re.compile(r"^--- Session \d+.*---$")
_PLAN_MARKER = re.compile(r"^===\s*PLAN-\d+\s*===$", re.IGNORECASE)
_BEAM_METADATA = re.compile(r"\s*->->.*$")
_INLINE_CODE_CONTENT = re.compile(
    r"^(def |class |import |from \S+ import |if __name__|"
    r"@\w+|return |raise |try:|except\b|finally:|with\b|yield\b|"
    r"print\(|self\.|assert\b|```|\s{4,}\S)"
)
_HTML_LINE = re.compile(r"^\s*</?[A-Za-z][^>]*>\s*$")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate used consistently within this chunker."""
    return max(1, len(text) // 4) if text else 0


def _split_role_prefix(line: str) -> tuple[str, str]:
    match = _ROLE_PREFIX.match(line)
    if not match:
        return "", line
    return match.group("prefix"), match.group("content")


def _append_if_new(lines: list[str], value: str) -> None:
    if not value:
        return
    if lines and lines[-1] == value:
        return
    lines.append(value)


def _clean_text(text: str) -> str:
    """Remove BEAM metadata and collapse code / HTML-heavy lines."""
    cleaned_lines: list[str] = []
    in_code_block = False

    for raw_line in text.splitlines():
        line = _BEAM_METADATA.sub("", raw_line).rstrip()
        prefix, content = _split_role_prefix(line)
        stripped_content = content.strip()

        if stripped_content.startswith("```"):
            if not in_code_block:
                _append_if_new(cleaned_lines, f"{prefix}[code example omitted]".strip())
                in_code_block = True
            else:
                in_code_block = False
            continue

        if in_code_block:
            continue

        if not stripped_content:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if _INLINE_CODE_CONTENT.match(stripped_content):
            _append_if_new(cleaned_lines, f"{prefix}[code line omitted]".strip())
            continue

        if _HTML_LINE.match(stripped_content):
            _append_if_new(cleaned_lines, f"{prefix}[HTML example omitted]".strip())
            continue

        cleaned_lines.append(f"{prefix}{content}" if prefix else line)

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines).strip()


def _flush_chunk(
    chunks: list[tuple[str, int, int, str]],
    lines: list[str],
    current_session: int,
    turn_in_session: int,
    saw_user: bool,
    saw_assistant: bool,
) -> int:
    if not lines or not (saw_user or saw_assistant):
        return turn_in_session

    chunk_text = "\n".join(lines).strip()
    if not chunk_text:
        return turn_in_session

    turn_in_session += 1
    if saw_user and saw_assistant:
        turn_status = "complete"
    elif saw_user:
        turn_status = "user_only"
    else:
        turn_status = "assistant_only"

    chunks.append((chunk_text, current_session, turn_in_session, turn_status))
    return turn_in_session


def _split_into_turns(text: str) -> list[tuple[str, int, int, str]]:
    """Parse flattened conversation text into turn-aware chunks.

    Preamble noise before the first recognized turn is ignored. Complete
    user-assistant pairs are emitted when possible. Standalone turns are kept
    and marked via turn_status metadata.
    """

    chunks: list[tuple[str, int, int, str]] = []
    current_lines: list[str] = []
    current_session = 1
    turn_in_session = 0
    saw_user = False
    saw_assistant = False

    def flush_current() -> None:
        nonlocal current_lines, turn_in_session, saw_user, saw_assistant
        turn_in_session = _flush_chunk(
            chunks,
            current_lines,
            current_session,
            turn_in_session,
            saw_user,
            saw_assistant,
        )
        current_lines = []
        saw_user = False
        saw_assistant = False

    for line in text.splitlines():
        stripped = line.strip()

        if _PLAN_MARKER.match(stripped):
            flush_current()
            continue

        if _SESSION_MARKER.match(stripped):
            flush_current()
            session_number = re.search(r"\d+", stripped)
            if session_number:
                current_session = int(session_number.group())
            turn_in_session = 0
            continue

        match = _TURN_START.match(line)
        if match:
            role = match.group("role").lower()

            if role == "user":
                if current_lines:
                    flush_current()
                current_lines.append(line)
                saw_user = True
                continue

            if role == "assistant":
                if saw_user and saw_assistant:
                    flush_current()
                elif saw_assistant and not saw_user:
                    flush_current()
                current_lines.append(line)
                saw_assistant = True
                continue

        if current_lines:
            current_lines.append(line)

    flush_current()
    return chunks


def _format_chunk_prefix(session: int, turn: int) -> str:
    return f"[Session {session}, Turn {turn}]"


def _format_chunk_text(prefix: str, body: str) -> str:
    return prefix if not body else f"{prefix}\n{body}"


def _split_on_separator(text: str, sep: str, max_chunk_tokens: int) -> list[str]:
    """Split text on a separator while keeping each part under the token budget."""
    segments = [segment for segment in text.split(sep) if segment]
    if not segments:
        return [text] if text else []

    parts: list[str] = []
    current: list[str] = []

    for segment in segments:
        candidate_parts = current + [segment]
        candidate = sep.join(candidate_parts).strip()
        if current and _estimate_tokens(candidate) > max_chunk_tokens:
            parts.append(sep.join(current).strip())
            current = [segment]
        else:
            current = candidate_parts

    if current:
        parts.append(sep.join(current).strip())

    return [part for part in parts if part]


def _split_hard(text: str, max_chunk_tokens: int) -> list[str]:
    if not text:
        return []

    hard_limit_chars = max(1, max_chunk_tokens * 4)
    parts: list[str] = []
    remaining = text.strip()

    while remaining and _estimate_tokens(remaining) > max_chunk_tokens:
        candidate = remaining[:hard_limit_chars]
        split_idx = max(candidate.rfind("\n"), candidate.rfind(" "))
        if split_idx > hard_limit_chars // 2:
            candidate = candidate[:split_idx]

        candidate = candidate.rstrip()
        if not candidate:
            candidate = remaining[:hard_limit_chars].rstrip() or remaining[:hard_limit_chars]

        parts.append(candidate)
        remaining = remaining[len(candidate) :].lstrip()

    if remaining:
        parts.append(remaining)

    return parts


def _split_oversized(body: str, prefix: str, max_chunk_tokens: int) -> list[str]:
    """Split a chunk body so the final prefixed chunks fit the requested size."""
    formatted = _format_chunk_text(prefix, body)
    if _estimate_tokens(formatted) <= max_chunk_tokens:
        return [formatted]

    prefix_budget = _estimate_tokens(prefix) + 1
    body_budget = max(1, max_chunk_tokens - prefix_budget)

    parts = _split_on_separator(body, "\n\n", body_budget)
    final_parts: list[str] = []
    for part in parts:
        if _estimate_tokens(part) > body_budget:
            final_parts.extend(_split_on_separator(part, "\n", body_budget))
        else:
            final_parts.append(part)

    result: list[str] = []
    for part in final_parts:
        if _estimate_tokens(part) > body_budget:
            result.extend(_split_hard(part, body_budget))
        else:
            result.append(part)

    return [_format_chunk_text(prefix, part) for part in result if part]


class ConversationChunker(Chunker):
    """Chunk conversation transcripts with BEAM-aware parsing and cleanup."""

    async def read(self):
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        turns = _split_into_turns(full_text)

        idx = 0
        for turn_body, session_num, turn_num, turn_status in turns:
            cleaned_body = _clean_text(turn_body)
            if not cleaned_body:
                continue

            prefix = _format_chunk_prefix(session_num, turn_num)
            sub_chunks = _split_oversized(cleaned_body, prefix, self.max_chunk_size)
            pair_complete = turn_status == "complete"

            for sub_text in sub_chunks:
                chunk_size = _estimate_tokens(sub_text)
                yield DocumentChunk(
                    id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{idx}"),
                    text=sub_text,
                    chunk_size=chunk_size,
                    is_part_of=self.document,
                    chunk_index=idx,
                    cut_type="conversation_turn_pair" if pair_complete else "conversation_turn",
                    contains=[],
                    metadata={
                        "index_fields": ["text"],
                        "session": session_num,
                        "turn": turn_num,
                        "turn_status": turn_status,
                        "pair_complete": pair_complete,
                    },
                )
                idx += 1

        self.chunk_index = idx
