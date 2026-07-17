"""Slack Block Kit renderer for cited answers (issue #3609, #3604).

Turns an :class:`Answer` (answer text + ordered ``Citation`` list from the
Commit-2 adapter) into Slack Block Kit blocks: a ``section`` block with the
answer, followed by a ``context`` block listing each source as a clickable
``<permalink|#channel · author · time>`` link (Slack mrkdwn link syntax).

Block Kit reference — blocks used:
* ``section`` with a ``mrkdwn`` text object (the answer);
* ``context`` with ``mrkdwn`` elements (the Sources list / no-sources note).

Dedupe: the adapter already collapses multiple chunks of one message to a single
Citation (keyed by ``document_id`` in ``cognee_memory._build_citations``). This
renderer additionally does a *defensive, display-level* dedupe keyed on
``(channel_id, ts)`` — idempotent when the input is already unique, and it never
merges the empty-``ts`` fallback citations. Citation order (relevance order from
the adapter) is preserved.

Graceful degradation (the #3604 "actionable, never broken output" bar):
* missing/blank permalink → plain-text source entry, never a broken ``<|>`` link;
* zero citations → the answer plus a subtle "no sources" note, not an empty block;
* blank answer text → a calm fallback message, never an empty reply.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.memory_adapter import Answer, Citation

# Slack section text objects are capped at 3000 chars; keep a safe margin.
_MAX_SECTION_CHARS = 2900
# Cap the number of rendered source links; note the remainder as "+N more".
DEFAULT_MAX_SOURCES = 5

_ANSWER_FALLBACK = "I couldn't find anything about that in this channel's memory yet."
_NO_SOURCES_NOTE = "_No sources found for this answer._"


def notification_text(answer: Answer) -> str:
    """Plain fallback text for the Slack notification / accessibility layer."""
    return answer.text.strip() if answer.text and answer.text.strip() else _ANSWER_FALLBACK


def _format_time(ts: str) -> str:
    """Render a Slack ts ("1700000000.000100") as a short UTC time, robustly."""
    try:
        seconds = int(float(ts))
        return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError, OverflowError, OSError):
        return ts or ""


def _source_label(cite: Citation) -> str:
    parts = [f"#{cite.channel_id}"] if cite.channel_id else []
    if cite.author:
        parts.append(cite.author)
    if cite.ts:
        parts.append(_format_time(cite.ts))
    return " · ".join(parts)


def _format_source(cite: Citation) -> str:
    """One Sources bullet — a link when possible, else safe plain text."""
    label = _source_label(cite)
    if cite.ok and cite.permalink:
        return f"• <{cite.permalink}|{label or 'message'}>"
    # No usable link: never emit "<|...>". Prefer the descriptive label, then
    # fall back to the message snippet.
    if label:
        return f"• {label}"
    return f"• {cite.snippet or 'source'}"


def _dedupe_for_display(citations: list[Citation]) -> list[Citation]:
    """Drop repeat (channel_id, ts) citations, preserving order.

    Defensive only — the adapter already dedupes by document_id. Citations with
    an empty ``ts`` (the missing-metadata fallback) are never merged.
    """
    seen: set[tuple[str, str]] = set()
    result: list[Citation] = []
    for cite in citations:
        key = (cite.channel_id, cite.ts)
        if cite.ts and key in seen:
            continue
        if cite.ts:
            seen.add(key)
        result.append(cite)
    return result


def _answer_section_text(answer: Answer) -> str:
    text = answer.text.strip() if answer.text else ""
    if not text:
        return _ANSWER_FALLBACK
    if len(text) > _MAX_SECTION_CHARS:
        return text[: _MAX_SECTION_CHARS - 1].rstrip() + "…"
    return text


def render_answer(answer: Answer, *, max_sources: int = DEFAULT_MAX_SOURCES) -> list[dict]:
    """Render an :class:`Answer` to Slack Block Kit blocks."""
    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": _answer_section_text(answer)}}
    ]

    citations = _dedupe_for_display(answer.citations)
    if not citations:
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": _NO_SOURCES_NOTE}]}
        )
        return blocks

    shown = citations[:max_sources]
    bullets = [_format_source(cite) for cite in shown]
    text = "*Sources:*\n" + "\n".join(bullets)
    extra = len(citations) - len(shown)
    if extra > 0:
        text += f"\n_+{extra} more_"

    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": text}]})
    return blocks
