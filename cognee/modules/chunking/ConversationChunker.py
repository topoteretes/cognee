"""Chunker that splits conversation text by turn pairs (User + Assistant).

Each chunk is one user message followed by its assistant response,
keeping semantic units intact.

Expects text formatted by BEAM's _flatten_chat:
    --- Session 1 ---
    [time_anchor] User: ...
    [time_anchor] Assistant: ...
"""

import re
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

# Matches lines like "User: ...", "[March-15-2024] User: ...", "Assistant: ..."
_TURN_START = re.compile(r"^(?:\[.*?\]\s*)?(?:User|Assistant):\s", re.MULTILINE)
_SESSION_MARKER = re.compile(r"^--- Session \d+.*---$", re.MULTILINE)

# Matches BEAM-10M plan headers like "=== PLAN-1 ==="
_PLAN_MARKER = re.compile(r"^===\s*PLAN-\d+\s*===$", re.MULTILINE | re.IGNORECASE)

# Matches fenced code blocks (```...```) including the language tag
_CODE_BLOCK = re.compile(r"```[\w]*\n.*?```", re.DOTALL)

# Matches inline HTML blocks (multiple consecutive lines with tags)
_HTML_BLOCK = re.compile(r"(?:<[^>]+>[\s\S]*?</[^>]+>\s*){3,}", re.DOTALL)

# Matches BEAM metadata annotations like "->-> 1,1" at end of lines
_BEAM_METADATA = re.compile(r"\s*->->.*$", re.MULTILINE)

# Matches inline code lines (def, class, import, from...import, indented code)
_INLINE_CODE_LINE = re.compile(
    r"^[ \t]*(def |class |import |from \S+ import |if __name__|"
    r"@\w+|return |raise |try:|except |finally:|with |yield |"
    r"print\(|self\.|assert ).*$",
    re.MULTILINE,
)


def _clean_text(text: str) -> str:
    """Remove BEAM metadata, code blocks, HTML blocks, and inline code."""
    text = _CODE_BLOCK.sub("[code example omitted]", text)
    text = _HTML_BLOCK.sub("[HTML example omitted]", text)
    text = _BEAM_METADATA.sub("", text)
    text = _INLINE_CODE_LINE.sub("[code line omitted]", text)
    text = re.sub(r"(\[code line omitted\]\n?){2,}", "[code example omitted]\n", text)
    return text


def _split_into_turn_pairs(text: str) -> list[tuple[str, int, int]]:
    """Split conversation text into turn-pair chunks with position metadata.

    Each chunk contains one User message and its following Assistant response.
    Session markers are prepended to the first turn pair in that session.

    Returns:
        List of (chunk_text, session_number, turn_number) tuples.
    """
    lines = text.split("\n")
    chunks: list[tuple[str, int, int]] = []
    current_turn_lines = []
    turns_in_pair = 0
    current_session = 1
    turn_in_session = 0

    def _flush():
        nonlocal current_turn_lines, turns_in_pair, turn_in_session
        if current_turn_lines:
            turn_in_session += 1
            prefix = f"[Session {current_session}, Turn {turn_in_session}]\n"
            chunk_text = prefix + "\n".join(current_turn_lines).strip()
            if chunk_text:
                chunks.append((chunk_text, current_session, turn_in_session))
            current_turn_lines = []
            turns_in_pair = 0

    for line in lines:
        # Skip BEAM-10M plan headers (=== PLAN-X ===)
        if _PLAN_MARKER.match(line.strip()):
            _flush()
            continue

        # Detect session boundaries
        session_match = _SESSION_MARKER.match(line.strip())
        if session_match:
            _flush()
            # Extract session number from "--- Session N ---"
            num_match = re.search(r"\d+", line)
            if num_match:
                current_session = int(num_match.group())
            turn_in_session = 0
            continue

        # Detect turn starts
        match = _TURN_START.match(line.strip())
        if match:
            matched_text = match.group(0).lower()
            if "user:" in matched_text:
                # New user turn — if we already have a complete pair, flush it
                if turns_in_pair >= 2:
                    _flush()
                turns_in_pair += 1
                current_turn_lines.append(line)
            elif "assistant:" in matched_text:
                turns_in_pair += 1
                current_turn_lines.append(line)
        else:
            # Continuation line — append to current turn
            current_turn_lines.append(line)

    _flush()
    return chunks


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# Max chars per chunk — keeps chunks within embedding model token limits.
# 30000 chars ≈ 7500 tokens, safely under the 8191 token limit.
_MAX_CHUNK_CHARS = 30000


def _split_oversized(text: str, session: int, turn: int) -> list[tuple[str, int, int]]:
    """Split a chunk that exceeds _MAX_CHUNK_CHARS into smaller pieces.

    Tries paragraph boundaries first, then falls back to newline boundaries.
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [(text, session, turn)]

    # Try splitting on paragraph boundaries first
    parts = _split_on_separator(text, "\n\n")

    # If any part is still too large, split on single newlines
    final = []
    for part in parts:
        if len(part) > _MAX_CHUNK_CHARS:
            final.extend(_split_on_separator(part, "\n"))
        else:
            final.append(part)

    # Last resort: hard split at max size
    result = []
    for part in final:
        while len(part) > _MAX_CHUNK_CHARS:
            result.append(part[:_MAX_CHUNK_CHARS])
            part = part[_MAX_CHUNK_CHARS:]
        if part:
            result.append(part)

    return [(r, session, turn) for r in result]


def _split_on_separator(text: str, sep: str) -> list[str]:
    """Split text on separator, grouping pieces to stay under _MAX_CHUNK_CHARS."""
    segments = text.split(sep)
    parts = []
    current = []
    current_len = 0
    sep_len = len(sep)

    for seg in segments:
        if current_len + len(seg) + sep_len > _MAX_CHUNK_CHARS and current:
            parts.append(sep.join(current))
            current = []
            current_len = 0
        current.append(seg)
        current_len += len(seg) + sep_len

    if current:
        parts.append(sep.join(current))

    return parts


class ConversationChunker(Chunker):
    """Chunks conversation text by user-assistant turn pairs.

    Code blocks, HTML, and inline code are stripped to reduce noise
    for graph extraction. BEAM metadata annotations are also removed.
    Chunks exceeding the embedding token limit are split further.
    """

    async def read(self):
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        turn_pairs = _split_into_turn_pairs(full_text)

        idx = 0
        for pair_text, session_num, turn_num in turn_pairs:
            cleaned = _clean_text(pair_text)
            sub_chunks = _split_oversized(cleaned, session_num, turn_num)

            for sub_text, sess, turn in sub_chunks:
                chunk_size = _estimate_tokens(sub_text)

                yield DocumentChunk(
                    id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{idx}"),
                    text=sub_text,
                    chunk_size=chunk_size,
                    is_part_of=self.document,
                    chunk_index=idx,
                    cut_type="conversation_turn_pair",
                    contains=[],
                    metadata={
                        "index_fields": ["text"],
                        "session": sess,
                        "turn": turn,
                    },
                )
                idx += 1

        self.chunk_index = idx
