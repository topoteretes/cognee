"""Chunker that preserves BEAM conversation structure and factual fidelity."""

from dataclasses import dataclass
import re
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

_TURN_START = re.compile(
    r"^(?:\[(?P<time_anchor>.*?)\]\s*)?(?P<role>User|Assistant):\s?(?P<content>.*)$"
)
_SESSION_MARKER = re.compile(r"^--- Session \d+.*---$")
_PLAN_MARKER = re.compile(r"^===\s*PLAN-\d+\s*===$", re.IGNORECASE)
_BEAM_METADATA = re.compile(r"\s*->->.*$")


@dataclass
class ConversationMessage:
    role: str
    body: str
    time_anchor: str | None = None


@dataclass
class ConversationTurn:
    session: int
    turn: int
    user: ConversationMessage | None = None
    assistant: ConversationMessage | None = None


@dataclass
class ChunkFragment:
    body: str
    fragment_kind: str


def _estimate_tokens(text: str) -> int:
    """Rough token estimate used consistently within this chunker."""
    return max(1, len(text) // 4) if text else 0


def _normalize_text(text: str) -> str:
    """Preserve original content while removing BEAM-only suffix markers."""
    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = _BEAM_METADATA.sub("", raw_line).rstrip()
        is_blank = line == ""

        if is_blank and previous_blank:
            continue

        normalized_lines.append(line)
        previous_blank = is_blank

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines).strip()


def _split_on_separator(text: str, sep: str, max_chunk_tokens: int) -> list[str]:
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


def _split_text(text: str, max_chunk_tokens: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    if _estimate_tokens(normalized) <= max_chunk_tokens:
        return [normalized]

    parts = _split_on_separator(normalized, "\n\n", max_chunk_tokens)
    final_parts: list[str] = []

    for part in parts:
        if _estimate_tokens(part) > max_chunk_tokens:
            final_parts.extend(_split_on_separator(part, "\n", max_chunk_tokens))
        else:
            final_parts.append(part)

    result: list[str] = []
    for part in final_parts:
        if _estimate_tokens(part) > max_chunk_tokens:
            result.extend(_split_hard(part, max_chunk_tokens))
        else:
            result.append(part)

    return [part for part in result if part]


def _message_label(message: ConversationMessage) -> str:
    time_anchor = f"[{message.time_anchor}] " if message.time_anchor else ""
    return f"{time_anchor}{message.role.capitalize()}:"


def _render_message(message: ConversationMessage, body: str | None = None) -> str:
    message_body = _normalize_text(message.body if body is None else body)
    label = _message_label(message)
    if not message_body:
        return label

    lines = message_body.splitlines()
    rendered = f"{label} {lines[0]}".rstrip()
    if len(lines) > 1:
        rendered += "\n" + "\n".join(lines[1:])
    return rendered


def _split_message(message: ConversationMessage, max_chunk_tokens: int) -> list[str]:
    rendered = _render_message(message)
    if _estimate_tokens(rendered) <= max_chunk_tokens:
        return [rendered]

    label = _message_label(message)
    body_budget = max(1, max_chunk_tokens - _estimate_tokens(label) - 1)
    body_parts = _split_text(message.body, body_budget)

    if not body_parts:
        return [label]

    return [_render_message(message, part) for part in body_parts]


def _truncate_text(text: str, max_chunk_tokens: int) -> str:
    normalized = _normalize_text(text)
    if not normalized or _estimate_tokens(normalized) <= max_chunk_tokens:
        return normalized

    hard_limit_chars = max(4, max_chunk_tokens * 4)
    candidate = normalized[: hard_limit_chars - 3]
    split_idx = max(candidate.rfind("\n"), candidate.rfind(" "))
    if split_idx > hard_limit_chars // 3:
        candidate = candidate[:split_idx]

    candidate = candidate.rstrip()
    if not candidate:
        candidate = (
            normalized[: hard_limit_chars - 3].rstrip() or normalized[: hard_limit_chars - 3]
        )

    return f"{candidate}..."


def _format_chunk_prefix(
    session: int,
    turn: int,
    part: int | None = None,
    part_count: int | None = None,
) -> str:
    if part is None or part_count is None or part_count <= 1:
        return f"[Session {session}, Turn {turn}]"
    return f"[Session {session}, Turn {turn}, Part {part}/{part_count}]"


def _format_chunk_text(prefix: str, body: str) -> str:
    return prefix if not body else f"{prefix}\n{body}"


def _get_body_budget(session: int, turn: int, max_chunk_tokens: int, split: bool) -> int:
    prefix = (
        _format_chunk_prefix(session, turn, 99, 99)
        if split
        else _format_chunk_prefix(session, turn)
    )
    return max(1, max_chunk_tokens - _estimate_tokens(prefix) - 1)


def _assistant_anchor_budget(
    user_text: str, assistant: ConversationMessage, body_budget: int
) -> tuple[str, int]:
    assistant_min = _estimate_tokens(_render_message(assistant, "..."))
    if _estimate_tokens(user_text) + assistant_min + 1 <= body_budget:
        anchor = user_text
    else:
        max_anchor_tokens = max(1, min(body_budget // 3, body_budget - assistant_min - 1))
        anchor = _truncate_text(user_text, max_anchor_tokens)
    assistant_budget = max(1, body_budget - _estimate_tokens(anchor) - 1)
    return anchor, assistant_budget


def _build_fragments_for_complete_turn(
    turn: ConversationTurn,
    max_chunk_tokens: int,
) -> list[ChunkFragment]:
    user_text = _render_message(turn.user)
    assistant_text = _render_message(turn.assistant)
    full_body = f"{user_text}\n{assistant_text}"
    single_body_budget = _get_body_budget(turn.session, turn.turn, max_chunk_tokens, split=False)

    if _estimate_tokens(full_body) <= single_body_budget:
        return [ChunkFragment(full_body, "full_pair")]

    split_body_budget = _get_body_budget(turn.session, turn.turn, max_chunk_tokens, split=True)
    if _estimate_tokens(user_text) > split_body_budget:
        user_parts = _split_message(turn.user, split_body_budget)
        anchor = _truncate_text(user_text, max(1, split_body_budget // 3))
        assistant_budget = max(1, split_body_budget - _estimate_tokens(anchor) - 1)
        assistant_parts = _split_message(turn.assistant, assistant_budget)

        fragments = [ChunkFragment(part, "user_fragment") for part in user_parts]
        fragments.extend(
            ChunkFragment(f"{anchor}\n{part}", "assistant_continuation") for part in assistant_parts
        )
        return fragments

    anchor, assistant_budget = _assistant_anchor_budget(
        user_text, turn.assistant, split_body_budget
    )
    assistant_parts = _split_message(turn.assistant, assistant_budget)
    return [
        ChunkFragment(f"{anchor}\n{part}", "assistant_continuation") for part in assistant_parts
    ]


def _build_fragments_for_incomplete_turn(
    turn: ConversationTurn,
    max_chunk_tokens: int,
) -> list[ChunkFragment]:
    split_body_budget = _get_body_budget(turn.session, turn.turn, max_chunk_tokens, split=True)

    if turn.user:
        parts = _split_message(turn.user, split_body_budget)
        kind = "user_only" if len(parts) == 1 else "user_fragment"
        return [ChunkFragment(part, kind) for part in parts]

    parts = _split_message(turn.assistant, split_body_budget)
    kind = "assistant_only" if len(parts) == 1 else "assistant_fragment"
    return [ChunkFragment(part, kind) for part in parts]


def _build_fragments_for_turn(turn: ConversationTurn, max_chunk_tokens: int) -> list[ChunkFragment]:
    if turn.user and turn.assistant:
        return _build_fragments_for_complete_turn(turn, max_chunk_tokens)
    return _build_fragments_for_incomplete_turn(turn, max_chunk_tokens)


def _parse_turns(text: str) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    current_session = 1
    turn_in_session = 0
    current_turn: ConversationTurn | None = None
    current_role: str | None = None
    current_time_anchor: str | None = None
    current_message_lines: list[str] = []
    in_code_block = False

    def clear_message_state() -> None:
        nonlocal current_role, current_time_anchor, current_message_lines, in_code_block
        current_role = None
        current_time_anchor = None
        current_message_lines = []
        in_code_block = False

    def update_code_block_state(line_text: str) -> None:
        nonlocal in_code_block
        if line_text.strip().startswith("```"):
            in_code_block = not in_code_block

    def finalize_message() -> None:
        nonlocal current_turn
        if current_role is None:
            return

        message = ConversationMessage(
            role=current_role,
            body=_normalize_text("\n".join(current_message_lines)),
            time_anchor=current_time_anchor,
        )
        if current_turn is None:
            current_turn = ConversationTurn(session=current_session, turn=0)

        if message.role == "user":
            current_turn.user = message
        else:
            current_turn.assistant = message

        clear_message_state()

    def flush_turn() -> None:
        nonlocal current_turn, turn_in_session
        finalize_message()
        if current_turn and (current_turn.user or current_turn.assistant):
            turn_in_session += 1
            current_turn.session = current_session
            current_turn.turn = turn_in_session
            turns.append(current_turn)
        current_turn = None

    def start_message(role: str, content: str, time_anchor: str | None) -> None:
        nonlocal current_turn, current_role, current_time_anchor, current_message_lines
        if current_turn is None:
            current_turn = ConversationTurn(session=current_session, turn=0)
        current_role = role
        current_time_anchor = time_anchor
        current_message_lines = [content] if content else []
        update_code_block_state(content)

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if _PLAN_MARKER.match(stripped):
            flush_turn()
            continue

        if _SESSION_MARKER.match(stripped):
            flush_turn()
            session_number = re.search(r"\d+", stripped)
            if session_number:
                current_session = int(session_number.group())
            turn_in_session = 0
            continue

        match = None if in_code_block else _TURN_START.match(raw_line)
        if match:
            role = match.group("role").lower()
            time_anchor = match.group("time_anchor")
            content = match.group("content")

            if role == "user":
                flush_turn()
                start_message(role, content, time_anchor)
                continue

            finalize_message()
            if current_turn and current_turn.assistant is not None:
                flush_turn()
            start_message(role, content, time_anchor)
            continue

        if current_role is not None:
            current_message_lines.append(raw_line)
            update_code_block_state(raw_line)

    flush_turn()
    return turns


class ConversationChunker(Chunker):
    """Chunk conversation transcripts with BEAM-aware turn pairing."""

    async def read(self):
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        turns = _parse_turns(full_text)
        idx = 0

        for turn in turns:
            fragments = _build_fragments_for_turn(turn, self.max_chunk_size)
            part_count = len(fragments)
            turn_status = (
                "complete"
                if turn.user and turn.assistant
                else "user_only"
                if turn.user
                else "assistant_only"
            )
            pair_complete = turn_status == "complete"

            for part, fragment in enumerate(fragments, start=1):
                prefix = _format_chunk_prefix(turn.session, turn.turn, part, part_count)
                chunk_text = _format_chunk_text(prefix, fragment.body)
                chunk_size = _estimate_tokens(chunk_text)

                yield DocumentChunk(
                    id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{idx}"),
                    text=chunk_text,
                    chunk_size=chunk_size,
                    is_part_of=self.document,
                    chunk_index=idx,
                    cut_type="conversation_turn_pair" if pair_complete else "conversation_turn",
                    contains=[],
                    metadata={
                        "index_fields": ["text"],
                        "session": turn.session,
                        "turn": turn.turn,
                        "turn_status": turn_status,
                        "pair_complete": pair_complete,
                        "part": part,
                        "part_count": part_count,
                        "fragment_kind": fragment.fragment_kind,
                    },
                )
                idx += 1

        self.chunk_index = idx
