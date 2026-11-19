"""Trim an ``only_context`` recall item down to the retrieved context (SDK-904).

On cognee >= 1.6.0 a completion search with ``only_context=true`` returns the
whole prompt its own LLM would have received: the conversation history, the
question rendered through the retriever's template, the retrieved context, and
the session guidance block. Hermes already holds the conversation and needs no
answering instructions, so injecting that string whole re-sends what is already
in its context window. This module keeps only the retrieved context and shrinks
it deterministically, without an LLM and without touching what the server
retrieved.

Two anchors, both ours to know:

* the question marker ``The question is: `<prompt>```, rendered from the very
  prompt the provider sent, so it matches exactly once however many backticks
  the prompt or the context carry; the context slot opens at the first backtick
  after it, in every one of the server's question templates;
* the end: the final character when the text ends with a backtick, otherwise
  the last ``\\n\\n## Active session guidance`` (the guidance block is the only
  layer the server renders after the template).

Anything that does not fit is left untouched and reported as unparsed: a
template change can only cost tokens, never memory.

A port of the claude-code/codex ``_recall_text.py`` (steps 2 and 3); the shared
cases live in ``integrations/conformance/recall_text_cases.json``. Defaults are
the reference ones openclaw uses (2000 / 12000). :func:`cap_total` is the port
of openclaw's ``capTotal``: one cap on everything a prompt injects.
"""

from __future__ import annotations

import os
import re

#: Every cognee question template starts with this line.
QUESTION_PREFIX = "The question is: `"
#: Title of the server-rendered guidance block (session_context_builder.BLOCK_TITLE).
GUIDANCE_TITLE = "## Active session guidance"

PASSAGES_HEADING = "## Relevant passages"
ENTITIES_HEADING = "## Relevant entities"
FACTS_HEADING = "## Related facts"
#: The hybrid retriever joins passages with this, no blank lines around it;
#: a markdown rule inside an answer has them, which ``split_passages`` uses.
PASSAGE_SEPARATOR = "\n---\n"

ENV_PASSAGE_CHARS = "COGNEE_RECALL_PASSAGE_CHARS"
ENV_CONTEXT_CHARS = "COGNEE_RECALL_CONTEXT_CHARS"
DEFAULT_PASSAGE_CHARS = 2000
DEFAULT_CONTEXT_CHARS = 12000

_SECTION_RE = re.compile(
    r"(?m)^(## Relevant passages|## Relevant entities|## Related facts)[ \t]*\n"
)
#: The header a bridged session chunk opens with. ``Session ID:`` is the
#: transcript document; ``# Session learning … (session <id>)`` the distilled note.
_SESSION_ID_RE = re.compile(r"^Session ID:\s*(\S+)", re.M)
_SESSION_LEARNING_RE = re.compile(r"^# Session learning\b[^\n]*?\(session ([^)\s]+)\)", re.M)
_DOC_HEADER_RE = re.compile(r"^(Session ID:|# Session learning\b)")
#: Entity bullets the graph builder emits for every node: a bare type statement
#: (``X is a concept.``) and node-set membership (``X belongs to set …``).
_TYPE_BULLET_RE = re.compile(r"^- .+? is an? [a-z]+(?:[ _-][a-z]+)?\.$")
_SET_BULLET_RE = re.compile(r"^- .+ belongs to set \S+\.?$")


def _int_env(name: str, default: int) -> int:
    """Non-negative int from the environment; ``0`` disables, junk means default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def passage_chars_limit() -> int:
    return _int_env(ENV_PASSAGE_CHARS, DEFAULT_PASSAGE_CHARS)


def context_chars_limit() -> int:
    return _int_env(ENV_CONTEXT_CHARS, DEFAULT_CONTEXT_CHARS)


# ---------------------------------------------------------------------------
# Step 2: cut the template away
# ---------------------------------------------------------------------------


def split_only_context(text: str, prompt: str) -> dict:
    """Split a 1.6.0 ``only_context`` item into its layers.

    Returns ``{"history", "context", "guidance", "parsed"}``. When the text
    cannot be read as the server's prompt (an older server's bare context, a
    rewritten question, a template this module does not know) ``parsed`` is
    False and the whole text sits in ``context`` — fail-open.
    """
    text = str(text or "")
    unparsed = {"history": "", "context": text, "guidance": "", "parsed": False}
    marker = f"{QUESTION_PREFIX}{prompt}`"
    question_at = text.find(marker)
    if question_at < 0:
        return unparsed
    start = text.find("`", question_at + len(marker))
    if start < 0:
        return unparsed
    start += 1
    if text.endswith("`"):
        end = len(text) - 1
        guidance = ""
    else:
        end = text.rfind("`\n\n" + GUIDANCE_TITLE)
        if end < start:
            return unparsed
        guidance = text[end + 1 :].strip()
    return {
        "history": text[:question_at].rstrip(),
        "context": text[start:end],
        "guidance": guidance,
        "parsed": True,
    }


# ---------------------------------------------------------------------------
# Step 3: shrink the context
# ---------------------------------------------------------------------------


def split_sections(context: str) -> dict:
    """``{"preamble", "passages", "entities", "facts"}`` — a key is None when absent.

    ``preamble`` is whatever precedes the first heading (the hybrid retriever's
    optional global-context section, or an unsectioned context from another
    retriever), returned as-is.
    """
    parts = _SECTION_RE.split(context or "")
    sections = {"preamble": parts[0], "passages": None, "entities": None, "facts": None}
    names = {PASSAGES_HEADING: "passages", ENTITIES_HEADING: "entities", FACTS_HEADING: "facts"}
    for i in range(1, len(parts) - 1, 2):
        sections[names[parts[i]]] = parts[i + 1]
    return sections


def split_passages(block: str) -> list[str]:
    """Passages of a ``## Relevant passages`` block.

    The server joins passages with exactly ``\\n---\\n`` — the next chunk's text
    starts right after the newline. A markdown horizontal rule inside an
    assistant answer is the same three characters, but it sits between blank
    lines (``\\n\\n---\\n\\n``), so after the split that piece starts with a
    newline. Such a piece is glued back onto the one before it, unless it
    opens with a document header. Documents without a header (remembered text,
    blog posts) start with their own text and stay separate.
    """
    pieces = (block or "").strip("\n").split(PASSAGE_SEPARATOR)
    passages: list[str] = []
    for piece in pieces:
        continuation = piece.startswith("\n") and not _DOC_HEADER_RE.match(piece.lstrip())
        if passages and continuation:
            passages[-1] = passages[-1] + PASSAGE_SEPARATOR + piece
        else:
            passages.append(piece)
    return [p for p in passages if p.strip()]


def is_session_transcript(passage: str) -> bool:
    """A bridged raw turn from another session, as opposed to a distilled
    ``# Session learning`` or a remembered document."""
    return passage.lstrip().startswith("Session ID:")


def passage_session_id(passage: str) -> str:
    """The session a bridged chunk came from, or "" for any other document."""
    head = passage.lstrip()[:400]
    match = _SESSION_ID_RE.search(head) or _SESSION_LEARNING_RE.search(head)
    return match.group(1) if match else ""


def cap_passage(passage: str, limit: int) -> str:
    """``passage`` cut to about ``limit`` chars at a paragraph break, marked."""
    if limit <= 0 or len(passage) <= limit:
        return passage
    cut = passage.rfind("\n\n", 0, limit)
    if cut < limit // 2:
        cut = passage.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = limit
    head = passage[:cut].rstrip()
    return f"{head}\n…[+{len(passage) - len(head)} chars]"


def prune_entities(block: str) -> tuple[str, int]:
    """Drop type and node-set bullets, then headings left without bullets.

    Returns ``(text, removed_lines)``; ``text`` is "" when nothing with content
    survives.
    """
    lines = (block or "").strip("\n").split("\n")
    kept: list[str] = []
    removed = 0
    for line in lines:
        if _TYPE_BULLET_RE.match(line) or _SET_BULLET_RE.match(line):
            removed += 1
            continue
        kept.append(line)
    pruned: list[str] = []
    for i, line in enumerate(kept):
        if line.startswith("### "):
            nxt = kept[i + 1] if i + 1 < len(kept) else ""
            if not nxt.startswith("- "):
                removed += 1
                continue
        pruned.append(line)
    text = "\n".join(pruned).strip("\n")
    if not any(line.startswith("- ") for line in pruned):
        return "", removed
    return text, removed


def minimize_context(
    context: str,
    session_id: str = "",
    *,
    passage_chars: int | None = None,
    context_chars: int | None = None,
) -> tuple[str, dict]:
    """Shrink a parsed context. Returns ``(text, stats)``.

    In order: drop the passages that are this session's own bridged turns, cap
    each remaining passage, prune the entity bullets that carry nothing, keep
    the facts, and as a last resort trim passages from the end until the whole
    thing fits ``context_chars``. Entities and facts are never trimmed by the
    budget. An unsectioned context (no headings) passes through untouched.
    """
    if passage_chars is None:
        passage_chars = passage_chars_limit()
    if context_chars is None:
        context_chars = context_chars_limit()
    stats = {
        "passages_in": 0,
        "passages_dropped": 0,
        "passages_capped": 0,
        "entity_lines_removed": 0,
        "budget_trimmed": 0,
    }
    sections = split_sections(context)
    if sections["passages"] is None and sections["entities"] is None and sections["facts"] is None:
        return (context or "").strip(), stats

    out: list[str] = []
    preamble = (sections["preamble"] or "").strip()
    if preamble:
        out.append(preamble)

    passages: list[str] = []
    if sections["passages"] is not None:
        raw = split_passages(sections["passages"])
        stats["passages_in"] = len(raw)
        for passage in raw:
            if session_id and passage_session_id(passage) == session_id:
                stats["passages_dropped"] += 1
                continue
            capped = cap_passage(passage, passage_chars)
            if capped is not passage:
                stats["passages_capped"] += 1
            passages.append(capped)

    entities = ""
    if sections["entities"] is not None:
        entities, removed = prune_entities(sections["entities"])
        stats["entity_lines_removed"] = removed

    facts = (sections["facts"] or "").strip("\n") if sections["facts"] is not None else ""

    def render(kept_passages: list[str]) -> str:
        blocks = list(out)
        if kept_passages:
            blocks.append(PASSAGES_HEADING + "\n" + PASSAGE_SEPARATOR.join(kept_passages))
        if entities:
            blocks.append(ENTITIES_HEADING + "\n" + entities)
        if facts.strip():
            blocks.append(FACTS_HEADING + "\n" + facts)
        return "\n\n".join(blocks).strip()

    text = render(passages)
    while context_chars > 0 and len(text) > context_chars and passages:
        passages.pop()
        stats["budget_trimmed"] += 1
        text = render(passages)
    return text, stats


def trim_recall_text(
    text: str,
    prompt: str,
    session_id: str = "",
    *,
    passage_chars: int | None = None,
    context_chars: int | None = None,
) -> tuple[str, str, dict]:
    """Steps 2 and 3 together. Returns ``(context, guidance, stats)``.

    ``context`` is what to inject under the memory heading; ``guidance`` the
    server's trailing guidance block when one came with the item (empty when
    the request carried no session id and no durable preferences exist).
    ``stats`` carries ``parsed``, ``chars_in``, ``chars_out``, ``history_chars``
    and the minimizer's counters. Unparsed text is returned whole.
    """
    text = str(text or "")
    layers = split_only_context(text, prompt)
    stats = {
        "parsed": layers["parsed"],
        "chars_in": len(text),
        "history_chars": len(layers["history"]),
    }
    if not layers["parsed"]:
        stats["chars_out"] = len(text)
        return text, "", stats
    context, more = minimize_context(
        layers["context"],
        session_id,
        passage_chars=passage_chars,
        context_chars=context_chars,
    )
    stats.update(more)
    stats["chars_out"] = len(context)
    return context, layers["guidance"], stats


# ---------------------------------------------------------------------------
# One cap on the whole injection
# ---------------------------------------------------------------------------

TRUNCATION_MARK = "\n…[truncated]"


def cap_total(parts: list[str], cap: int, atomic=lambda index: False) -> list[str]:
    """``parts`` in priority order, each kept whole, cut, or "" to fit ``cap``.

    :func:`trim_recall_text` bounds each dataset's item on its own, so the
    guidance block, an unparsed item and the code lane could still add up past
    the context budget. Parts are kept in order until the cap is reached; the
    part that crosses it is cut at the last line break that fits and marked,
    and every later part is dropped. An ``atomic`` part (one carrying its own
    markup) is kept whole or dropped, never cut. ``cap`` 0 (or less) disables it.
    """
    if cap <= 0:
        return list(parts)
    left = cap
    out: list[str] = []
    for index, part in enumerate(parts):
        if not part:
            out.append(part)
            continue
        if len(part) <= left:
            left -= len(part)
            out.append(part)
            continue
        if atomic(index):
            out.append("")
            continue
        room = left - len(TRUNCATION_MARK)
        left = 0
        if room <= 0:
            out.append("")
            continue
        head = part[:room]
        cut = head.rfind("\n")
        kept = (head[:cut] if cut > 0 else head).rstrip()
        out.append(kept + TRUNCATION_MARK if kept else "")
    return out
