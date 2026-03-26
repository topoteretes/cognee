"""Chunker that splits conversation text by turn pairs (User + Assistant).

Each chunk is one user message followed by its assistant response,
keeping semantic units intact. Code blocks are stripped and replaced
with a brief marker to reduce noise for graph extraction.

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
_SESSION_MARKER = re.compile(r"^--- Session \d+ ---$", re.MULTILINE)

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


def _strip_code(text: str) -> str:
    """Replace code blocks, HTML blocks, inline code, and BEAM metadata."""
    text = _CODE_BLOCK.sub("[code example omitted]", text)
    text = _HTML_BLOCK.sub("[HTML example omitted]", text)
    text = _BEAM_METADATA.sub("", text)
    # Strip inline code lines (consecutive lines become a single marker)
    text = _INLINE_CODE_LINE.sub("[code line omitted]", text)
    # Collapse consecutive "[code line omitted]" into one marker
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
    current_session_header = ""
    current_turn_lines = []
    turns_in_pair = 0
    current_session = 1
    turn_in_session = 0

    def _flush():
        nonlocal current_turn_lines, turns_in_pair, current_session_header, turn_in_session
        if current_turn_lines:
            turn_in_session += 1
            prefix = f"[Session {current_session}, Turn {turn_in_session}]\n"
            chunk_text = prefix + "\n".join(current_turn_lines).strip()
            if chunk_text:
                chunks.append((chunk_text, current_session, turn_in_session))
            current_turn_lines = []
            current_session_header = ""
            turns_in_pair = 0

    for line in lines:
        # Detect session boundaries
        session_match = _SESSION_MARKER.match(line.strip())
        if session_match:
            _flush()
            # Extract session number from "--- Session N ---"
            num_match = re.search(r"\d+", line)
            if num_match:
                current_session = int(num_match.group())
            turn_in_session = 0
            current_session_header = line.strip() + "\n"
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


class ConversationChunker(Chunker):
    """Chunks conversation text by user-assistant turn pairs.

    Code blocks and HTML are stripped before yielding chunks to reduce
    noise for downstream graph extraction.
    """

    async def read(self):
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        turn_pairs = _split_into_turn_pairs(full_text)

        for idx, (pair_text, session_num, turn_num) in enumerate(turn_pairs):
            cleaned = _strip_code(pair_text)
            chunk_size = _estimate_tokens(cleaned)

            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{idx}"),
                text=cleaned,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=idx,
                cut_type="conversation_turn_pair",
                contains=[],
                metadata={
                    "index_fields": ["text"],
                    "session": session_num,
                    "turn": turn_num,
                },
            )

        self.chunk_index = len(turn_pairs)
