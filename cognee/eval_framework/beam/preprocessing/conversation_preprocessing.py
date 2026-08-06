"""Shared BEAM-aware conversation preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

_TURN_START = re.compile(
    r"^(?:\[(?P<time_anchor>.*?)\]\s*)?(?P<role>User|Assistant):\s?(?P<content>.*)$"
)
_SESSION_MARKER = re.compile(r"^--- Session \d+.*---$")
_PLAN_MARKER = re.compile(r"^===\s*PLAN-\d+\s*===$", re.IGNORECASE)
_BEAM_METADATA = re.compile(r"\s*->->.*$")
_LONG_UNBROKEN_TOKEN = re.compile(r"\S{513,}")
_MAX_UNBROKEN_TOKEN_CHARS = 512


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    body: str
    time_anchor: str | None = None


@dataclass(frozen=True)
class ConversationTurn:
    session: int
    turn: int
    plan: str | None = None
    user: ConversationMessage | None = None
    assistant: ConversationMessage | None = None


@dataclass(frozen=True)
class ChunkFragment:
    body: str
    fragment_kind: str


@dataclass(frozen=True)
class PreprocessedFragment:
    text: str
    chunk_size: int
    session: int
    turn: int
    plan: str | None
    turn_status: str
    pair_complete: bool
    part: int
    part_count: int
    fragment_kind: str
    time_anchor: str | None = None


@lru_cache(maxsize=4096)
def estimate_tokens(text: str) -> int:
    """Estimate tokens with the configured embedding tokenizer when available."""
    if not text:
        return 0

    try:
        from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

        tokenizer = getattr(get_embedding_engine(), "tokenizer", None)
        if tokenizer:
            return max(1, tokenizer.count_tokens(text) * 2)
    except Exception:
        pass

    return max(1, len(text) // 4, len(text.split()) * 2)


def _wrap_long_unbroken_tokens(text: str) -> str:
    """Prevent downstream chunkers from receiving a single oversized token."""

    def wrap_match(match: re.Match[str]) -> str:
        value = match.group(0)
        return " ".join(
            value[index : index + _MAX_UNBROKEN_TOKEN_CHARS]
            for index in range(0, len(value), _MAX_UNBROKEN_TOKEN_CHARS)
        )

    return _LONG_UNBROKEN_TOKEN.sub(wrap_match, text)


def normalize_text(text: str) -> str:
    """Preserve original content while removing BEAM-only suffix markers."""
    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = _wrap_long_unbroken_tokens(_BEAM_METADATA.sub("", raw_line).rstrip())
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

        if current and estimate_tokens(candidate) > max_chunk_tokens:
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

    parts: list[str] = []
    remaining = text.strip()

    while remaining and estimate_tokens(remaining) > max_chunk_tokens:
        hard_limit_chars = min(len(remaining), max(1, max_chunk_tokens * 4))
        candidate = remaining[:hard_limit_chars]
        split_idx = max(candidate.rfind("\n"), candidate.rfind(" "))
        if split_idx > 0:
            candidate = candidate[:split_idx]

        candidate = candidate.rstrip()
        while candidate and estimate_tokens(candidate) > max_chunk_tokens:
            if len(candidate) <= 1:
                break

            shorter_candidate = candidate[: max(1, len(candidate) // 2)].rstrip()
            if len(shorter_candidate) >= len(candidate):
                break

            split_idx = max(shorter_candidate.rfind("\n"), shorter_candidate.rfind(" "))
            if split_idx > 0:
                shorter_candidate = shorter_candidate[:split_idx].rstrip()
            candidate = shorter_candidate

        if not candidate:
            candidate = (
                remaining[: max(1, max_chunk_tokens)].rstrip()
                or remaining[: max(1, max_chunk_tokens)]
            )

        parts.append(candidate)
        remaining = remaining[len(candidate) :].lstrip()

    if remaining:
        parts.append(remaining)

    return parts


def _split_text(text: str, max_chunk_tokens: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    if estimate_tokens(normalized) <= max_chunk_tokens:
        return [normalized]

    parts = _split_on_separator(normalized, "\n\n", max_chunk_tokens)
    final_parts: list[str] = []

    for part in parts:
        if estimate_tokens(part) > max_chunk_tokens:
            final_parts.extend(_split_on_separator(part, "\n", max_chunk_tokens))
        else:
            final_parts.append(part)

    result: list[str] = []
    for part in final_parts:
        if estimate_tokens(part) > max_chunk_tokens:
            result.extend(_split_hard(part, max_chunk_tokens))
        else:
            result.append(part)

    return [part for part in result if part]


def _tail_text(text: str, max_chunk_tokens: int) -> str:
    normalized = normalize_text(text)
    if not normalized or max_chunk_tokens <= 0:
        return ""

    hard_limit_chars = max(1, max_chunk_tokens * 4)
    if estimate_tokens(normalized) <= max_chunk_tokens:
        return normalized

    candidate = normalized[-hard_limit_chars:]
    split_idx = (
        min(idx for idx in (candidate.find("\n"), candidate.find(" ")) if idx >= 0)
        if "\n" in candidate or " " in candidate
        else -1
    )
    if 0 <= split_idx < hard_limit_chars // 2:
        candidate = candidate[split_idx + 1 :]

    return candidate.lstrip()


def _add_previous_overlap(
    parts: list[str],
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if overlap_tokens <= 0 or len(parts) <= 1:
        return parts

    overlapped_parts = [parts[0]]
    for previous, current in zip(parts, parts[1:]):
        available_tokens = max_chunk_tokens - estimate_tokens(current) - 1
        overlap = _tail_text(previous, min(overlap_tokens, available_tokens))
        if overlap:
            overlapped_parts.append(f"{overlap}\n{current}")
        else:
            overlapped_parts.append(current)

    return overlapped_parts


def _message_label(
    message: ConversationMessage,
    include_time_anchor: bool = True,
    role_suffix: str = "",
) -> str:
    time_anchor = f"[{message.time_anchor}] " if include_time_anchor and message.time_anchor else ""
    return f"{time_anchor}{message.role.capitalize()}{role_suffix}:"


def _render_message(
    message: ConversationMessage,
    body: str | None = None,
    *,
    include_time_anchor: bool = True,
    role_suffix: str = "",
) -> str:
    message_body = normalize_text(message.body if body is None else body)
    label = _message_label(
        message,
        include_time_anchor=include_time_anchor,
        role_suffix=role_suffix,
    )
    if not message_body:
        return label

    lines = message_body.splitlines()
    rendered = f"{label} {lines[0]}".rstrip()
    if len(lines) > 1:
        rendered += "\n" + "\n".join(lines[1:])
    return rendered


def _replace_role_label(rendered_message: str, role: str, replacement_label: str) -> str:
    match = re.match(
        rf"^(?P<time_anchor>\[[^\]]+\]\s*)?{role.capitalize()}:",
        rendered_message,
    )
    if not match:
        return rendered_message

    time_anchor = match.group("time_anchor") or ""
    return f"{time_anchor}{replacement_label}:{rendered_message[match.end() :]}"


def _repeated_user_anchor(text: str) -> str:
    return _replace_role_label(text, "user", "User (repeated)")


def _split_message(
    message: ConversationMessage,
    max_chunk_tokens: int,
    *,
    include_time_anchor: bool = True,
    overlap_tokens: int = 0,
    role_suffix: str = "",
) -> list[str]:
    rendered = _render_message(
        message,
        include_time_anchor=include_time_anchor,
        role_suffix=role_suffix,
    )
    if estimate_tokens(rendered) <= max_chunk_tokens:
        return [rendered]

    label = _message_label(
        message,
        include_time_anchor=include_time_anchor,
        role_suffix=role_suffix,
    )
    body_budget = max(1, max_chunk_tokens - estimate_tokens(label) - 1)
    body_parts = _split_text(message.body, body_budget)

    if not body_parts:
        return [label]

    body_parts = _add_previous_overlap(body_parts, body_budget, overlap_tokens)
    return [
        _render_message(
            message,
            part,
            include_time_anchor=include_time_anchor,
            role_suffix=role_suffix,
        )
        for part in body_parts
    ]


def _truncate_text(text: str, max_chunk_tokens: int) -> str:
    normalized = normalize_text(text)
    if not normalized or estimate_tokens(normalized) <= max_chunk_tokens:
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
    plan: str | None = None,
    time_anchor: str | None = None,
) -> str:
    prefix_parts: list[str] = []
    if plan:
        prefix_parts.append(plan.upper())
    prefix_parts.append(f"Session {session}")
    prefix_parts.append(f"Turn {turn}")
    if time_anchor:
        prefix_parts.append(f"date {time_anchor}")
    if part is not None and part_count is not None and part_count > 1:
        prefix_parts.append(f"Part {part}/{part_count}")
    return f"[{', '.join(prefix_parts)}]"


def _format_chunk_text(prefix: str, body: str) -> str:
    return prefix if not body else f"{prefix}\n{body}"


def _format_split_prefix(
    session: int,
    turn: int,
    part: int,
    part_count: int,
    time_anchor: str | None = None,
) -> str:
    prefix_parts = [
        "Conversation continues",
        f"session {session}",
        f"turn {turn}",
    ]
    if time_anchor:
        prefix_parts.append(f"date {time_anchor}")
    prefix_parts.append(f"part {part} of {part_count}")
    return f"[{', '.join(prefix_parts)}]"


def _turn_time_anchor(turn: ConversationTurn) -> str | None:
    if turn.user and turn.user.time_anchor:
        return turn.user.time_anchor
    if turn.assistant and turn.assistant.time_anchor:
        return turn.assistant.time_anchor
    return None


def _format_time_metadata_prefix(time_anchor: str) -> str:
    return f"[Conversation metadata: date {time_anchor}]"


def _get_body_budget(
    session: int,
    turn: int,
    max_chunk_tokens: int,
    split: bool,
    plan: str | None = None,
    include_location_prefix: bool = True,
) -> int:
    if not include_location_prefix:
        return max_chunk_tokens

    prefix = (
        _format_chunk_prefix(session, turn, 99, 99, plan=plan)
        if split
        else _format_chunk_prefix(session, turn, plan=plan)
    )
    return max(1, max_chunk_tokens - estimate_tokens(prefix) - 1)


def _assistant_anchor_budget(
    user_text: str,
    assistant: ConversationMessage,
    body_budget: int,
    *,
    include_time_anchor: bool = True,
) -> tuple[str, int]:
    assistant_min = estimate_tokens(
        _render_message(
            assistant,
            "...",
            include_time_anchor=include_time_anchor,
            role_suffix=" (cont.)",
        )
    )
    repeated_user_text = _repeated_user_anchor(user_text)
    if estimate_tokens(repeated_user_text) + assistant_min + 1 <= body_budget:
        anchor = repeated_user_text
    else:
        max_anchor_tokens = max(1, min(body_budget // 3, body_budget - assistant_min - 1))
        anchor = _repeated_user_anchor(_truncate_text(user_text, max_anchor_tokens))
    assistant_budget = max(1, body_budget - estimate_tokens(anchor) - 1)
    return anchor, assistant_budget


def _build_fragments_for_complete_turn(
    turn: ConversationTurn,
    max_chunk_tokens: int,
    *,
    include_location_prefix: bool = True,
    include_time_anchor: bool = True,
    split_overlap_tokens: int = 0,
) -> list[ChunkFragment]:
    user_text = _render_message(turn.user, include_time_anchor=include_time_anchor)
    assistant_text = _render_message(turn.assistant, include_time_anchor=include_time_anchor)
    full_body = f"{user_text}\n{assistant_text}"
    single_body_budget = _get_body_budget(
        turn.session,
        turn.turn,
        max_chunk_tokens,
        split=False,
        plan=turn.plan,
        include_location_prefix=include_location_prefix,
    )

    if estimate_tokens(full_body) <= single_body_budget:
        return [ChunkFragment(full_body, "full_pair")]

    split_body_budget = _get_body_budget(
        turn.session,
        turn.turn,
        max_chunk_tokens,
        split=True,
        plan=turn.plan,
        include_location_prefix=include_location_prefix,
    )
    if estimate_tokens(user_text) > split_body_budget:
        user_parts = _split_message(
            turn.user,
            split_body_budget,
            include_time_anchor=include_time_anchor,
            overlap_tokens=split_overlap_tokens,
        )
        anchor = _repeated_user_anchor(_truncate_text(user_text, max(1, split_body_budget // 3)))
        assistant_budget = max(1, split_body_budget - estimate_tokens(anchor) - 1)
        assistant_parts = _split_message(
            turn.assistant,
            assistant_budget,
            include_time_anchor=include_time_anchor,
            overlap_tokens=split_overlap_tokens,
            role_suffix=" (cont.)",
        )

        fragments = [ChunkFragment(part, "user_fragment") for part in user_parts]
        fragments.extend(
            ChunkFragment(
                f"{anchor}\n{part}",
                "assistant_continuation",
            )
            for part in assistant_parts
        )
        return fragments

    anchor, assistant_budget = _assistant_anchor_budget(
        user_text,
        turn.assistant,
        split_body_budget,
        include_time_anchor=include_time_anchor,
    )
    assistant_parts = _split_message(
        turn.assistant,
        assistant_budget,
        include_time_anchor=include_time_anchor,
        overlap_tokens=split_overlap_tokens,
        role_suffix=" (cont.)",
    )
    return [
        ChunkFragment(
            f"{anchor}\n{part}",
            "assistant_continuation",
        )
        for part in assistant_parts
    ]


def _build_fragments_for_incomplete_turn(
    turn: ConversationTurn,
    max_chunk_tokens: int,
    *,
    include_location_prefix: bool = True,
    include_time_anchor: bool = True,
    split_overlap_tokens: int = 0,
) -> list[ChunkFragment]:
    split_body_budget = _get_body_budget(
        turn.session,
        turn.turn,
        max_chunk_tokens,
        split=True,
        plan=turn.plan,
        include_location_prefix=include_location_prefix,
    )

    if turn.user:
        parts = _split_message(
            turn.user,
            split_body_budget,
            include_time_anchor=include_time_anchor,
            overlap_tokens=split_overlap_tokens,
        )
        kind = "user_only" if len(parts) == 1 else "user_fragment"
        return [ChunkFragment(part, kind) for part in parts]

    parts = _split_message(
        turn.assistant,
        split_body_budget,
        include_time_anchor=include_time_anchor,
        overlap_tokens=split_overlap_tokens,
    )
    kind = "assistant_only" if len(parts) == 1 else "assistant_fragment"
    return [ChunkFragment(part, kind) for part in parts]


def _build_fragments_for_turn(
    turn: ConversationTurn,
    max_chunk_tokens: int,
    *,
    include_location_prefix: bool = True,
    include_time_anchor: bool = True,
    split_overlap_tokens: int = 0,
) -> list[ChunkFragment]:
    if turn.user and turn.assistant:
        return _build_fragments_for_complete_turn(
            turn,
            max_chunk_tokens,
            include_location_prefix=include_location_prefix,
            include_time_anchor=include_time_anchor,
            split_overlap_tokens=split_overlap_tokens,
        )
    return _build_fragments_for_incomplete_turn(
        turn,
        max_chunk_tokens,
        include_location_prefix=include_location_prefix,
        include_time_anchor=include_time_anchor,
        split_overlap_tokens=split_overlap_tokens,
    )


def _append_turn(
    turns: list[ConversationTurn],
    current_turn: ConversationTurn | None,
    session: int,
    turn_in_session: int,
    plan: str | None = None,
) -> tuple[ConversationTurn | None, int]:
    if current_turn and (current_turn.user or current_turn.assistant):
        turn_in_session += 1
        turns.append(
            ConversationTurn(
                session=session,
                turn=turn_in_session,
                plan=plan if current_turn.plan is None else current_turn.plan,
                user=current_turn.user,
                assistant=current_turn.assistant,
            )
        )
    return None, turn_in_session


def parse_turns_from_beam_batches(
    chat_batches: list[list[dict]],
    *,
    plan: str | None = None,
    session_numbers: list[int] | None = None,
) -> list[ConversationTurn]:
    """Build turn records directly from structured BEAM chat batches."""
    turns: list[ConversationTurn] = []

    for batch_index, batch in enumerate(chat_batches):
        session = (
            session_numbers[batch_index]
            if session_numbers is not None and batch_index < len(session_numbers)
            else batch_index + 1
        )
        turn_in_session = 0
        current_turn: ConversationTurn | None = None

        for raw_message in batch:
            role = str(raw_message.get("role", "unknown")).lower()
            if role not in {"user", "assistant"}:
                continue

            message = ConversationMessage(
                role=role,
                body=normalize_text(str(raw_message.get("content", "") or "")),
                time_anchor=raw_message.get("time_anchor"),
            )

            if role == "user":
                current_turn, turn_in_session = _append_turn(
                    turns, current_turn, session, turn_in_session, plan=plan
                )
                current_turn = ConversationTurn(session=session, turn=0, plan=plan, user=message)
                continue

            if current_turn is None:
                current_turn = ConversationTurn(
                    session=session, turn=0, plan=plan, assistant=message
                )
                continue

            if current_turn.assistant is None:
                current_turn = ConversationTurn(
                    session=session,
                    turn=0,
                    plan=plan,
                    user=current_turn.user,
                    assistant=message,
                )
                continue

            current_turn, turn_in_session = _append_turn(
                turns, current_turn, session, turn_in_session, plan=plan
            )
            current_turn = ConversationTurn(session=session, turn=0, plan=plan, assistant=message)

        _append_turn(turns, current_turn, session, turn_in_session, plan=plan)

    return turns


def parse_turns_from_beam_10m_plan_batches(
    plan_batches: list[dict],
    *,
    plan: str,
    session_offset: int = 0,
) -> list[ConversationTurn]:
    """Build turn records directly from structured BEAM-10M plan batches."""
    normalized_batches: list[list[dict]] = []
    session_numbers: list[int] = []

    for batch_index, batch in enumerate(plan_batches, start=1):
        turns = batch.get("turns") or []
        normalized_messages: list[dict] = []
        batch_time_anchor = batch.get("time_anchor")
        for turn_group in turns:
            if isinstance(turn_group, list):
                for raw_message in turn_group:
                    if isinstance(raw_message, dict):
                        normalized_message = dict(raw_message)
                        if normalized_message.get("time_anchor") is None and batch_time_anchor:
                            normalized_message["time_anchor"] = batch_time_anchor
                        normalized_messages.append(normalized_message)
            elif isinstance(turn_group, dict):
                normalized_message = dict(turn_group)
                if normalized_message.get("time_anchor") is None and batch_time_anchor:
                    normalized_message["time_anchor"] = batch_time_anchor
                normalized_messages.append(normalized_message)

        if not batch_time_anchor:
            batch_time_anchor = next(
                (
                    str(message["time_anchor"])
                    for message in normalized_messages
                    if message.get("time_anchor")
                ),
                None,
            )
            if batch_time_anchor:
                normalized_messages = [
                    {
                        **message,
                        "time_anchor": message.get("time_anchor") or batch_time_anchor,
                    }
                    for message in normalized_messages
                ]

        normalized_batches.append(normalized_messages)
        session_numbers.append(session_offset + batch.get("batch_number", batch_index))

    return parse_turns_from_beam_batches(
        normalized_batches,
        plan=plan,
        session_numbers=session_numbers,
    )


def parse_turns_from_text(text: str) -> list[ConversationTurn]:
    """Parse a flattened conversation transcript into turn records."""
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
            body=normalize_text("\n".join(current_message_lines)),
            time_anchor=current_time_anchor,
        )
        if current_turn is None:
            current_turn = ConversationTurn(session=current_session, turn=0)

        if message.role == "user":
            current_turn = ConversationTurn(
                session=current_turn.session,
                turn=current_turn.turn,
                plan=current_turn.plan,
                user=message,
                assistant=current_turn.assistant,
            )
        else:
            current_turn = ConversationTurn(
                session=current_turn.session,
                turn=current_turn.turn,
                plan=current_turn.plan,
                user=current_turn.user,
                assistant=message,
            )

        clear_message_state()

    def flush_turn() -> None:
        nonlocal current_turn, turn_in_session
        finalize_message()
        current_turn, turn_in_session = _append_turn(
            turns, current_turn, current_session, turn_in_session
        )

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


def build_preprocessed_fragments_from_turns(
    turns: list[ConversationTurn],
    max_chunk_tokens: int,
    *,
    include_location_prefix: bool = True,
    include_time_anchor: bool = True,
    split_overlap_tokens: int = 0,
    include_time_metadata_header: bool = False,
) -> list[PreprocessedFragment]:
    """Render turn records into bounded, self-anchored text fragments."""
    fragments: list[PreprocessedFragment] = []

    for turn in turns:
        turn_fragments = _build_fragments_for_turn(
            turn,
            max_chunk_tokens,
            include_location_prefix=include_location_prefix,
            include_time_anchor=include_time_anchor,
            split_overlap_tokens=split_overlap_tokens,
        )
        part_count = len(turn_fragments)
        turn_status = (
            "complete"
            if turn.user and turn.assistant
            else "user_only"
            if turn.user
            else "assistant_only"
        )
        pair_complete = turn_status == "complete"
        time_anchor = _turn_time_anchor(turn)

        for part, fragment in enumerate(turn_fragments, start=1):
            prefix_time_anchor = (
                time_anchor if include_time_anchor or include_time_metadata_header else None
            )
            prefix = _format_chunk_prefix(
                turn.session,
                turn.turn,
                part,
                part_count,
                plan=turn.plan,
                time_anchor=prefix_time_anchor,
            )
            if include_location_prefix:
                text = _format_chunk_text(prefix, fragment.body)
            elif part_count > 1:
                split_prefix = _format_split_prefix(
                    turn.session,
                    turn.turn,
                    part,
                    part_count,
                    time_anchor=prefix_time_anchor,
                )
                text = _format_chunk_text(split_prefix, fragment.body)
            else:
                text = fragment.body
            if include_time_metadata_header and time_anchor:
                if not include_location_prefix and part_count == 1:
                    text = _format_chunk_text(_format_time_metadata_prefix(time_anchor), text)
            fragments.append(
                PreprocessedFragment(
                    text=text,
                    chunk_size=estimate_tokens(text),
                    session=turn.session,
                    turn=turn.turn,
                    plan=turn.plan,
                    turn_status=turn_status,
                    pair_complete=pair_complete,
                    part=part,
                    part_count=part_count,
                    fragment_kind=fragment.fragment_kind,
                    time_anchor=time_anchor,
                )
            )

    return fragments


def build_preprocessed_fragments_from_text(
    text: str, max_chunk_tokens: int
) -> list[PreprocessedFragment]:
    return build_preprocessed_fragments_from_turns(parse_turns_from_text(text), max_chunk_tokens)


def build_preprocessed_fragments_from_beam_batches(
    chat_batches: list[list[dict]], max_chunk_tokens: int
) -> list[PreprocessedFragment]:
    return build_preprocessed_fragments_from_turns(
        parse_turns_from_beam_batches(chat_batches), max_chunk_tokens
    )


def build_preprocessed_fragments_from_beam_10m_plan_batches(
    plan_batches: list[dict],
    max_chunk_tokens: int,
    *,
    plan: str,
    session_offset: int = 0,
    include_location_prefix: bool = False,
    include_time_anchor: bool = False,
    split_overlap_tokens: int = 40,
    include_time_metadata_header: bool = True,
) -> list[PreprocessedFragment]:
    return build_preprocessed_fragments_from_turns(
        parse_turns_from_beam_10m_plan_batches(
            plan_batches,
            plan=plan,
            session_offset=session_offset,
        ),
        max_chunk_tokens,
        include_location_prefix=include_location_prefix,
        include_time_anchor=include_time_anchor,
        split_overlap_tokens=split_overlap_tokens,
        include_time_metadata_header=include_time_metadata_header,
    )
