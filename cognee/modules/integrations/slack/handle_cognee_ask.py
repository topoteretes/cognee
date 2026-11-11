"""The answer review of ``@cognee <question>``: Share and Discard, and the answer's text.

Nothing about a question becomes visible to the channel until the asker has seen the
answer and chosen to share it. ``handle_cognee_mention.ask`` shows the answer privately
with Share and Discard; Share is the only thing that ever posts ``in_channel`` (question
and answer together, since nothing was public before it), Discard drops it unseen.
"""

import logging
from typing import Any

from cognee.modules.integrations.slack.response_url import post_to_response_url

logger = logging.getLogger(__name__)

# Referenced by handle_slack_interactive.py to dispatch the answer review
# prompt's button clicks back here.
ASK_SHARE_ACTION_ID = "cognee_ask_share"
ASK_DISCARD_ACTION_ID = "cognee_ask_discard"


# A ready-to-share (question, answer) pair, keyed by a short id (never the
# text itself — Slack button values cap at 2000 chars, easily blown by a
# real question+answer). Popped on Share or Discard; an id missing here
# (server restart, double-click) means "no longer available", not a crash.
_pending_answers: dict[str, tuple[str, str, list[dict[str, Any]]]] = {}

# Keep the placeholder/confirm text readable even if someone pastes an essay.
_QUERY_PREVIEW_MAX_CHARS = 200


def _ephemeral(
    text: str,
    *,
    replace_original: bool = False,
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a reply visible only to the person who ran the command.

    Used for the placeholder ack and every error path — none of those are
    useful to anyone else in the channel/DM.
    """
    return _reply("ephemeral", text, replace_original=replace_original, blocks=blocks)


def _in_channel(
    text: str,
    *,
    replace_original: bool = False,
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a reply visible to everyone in the channel/DM the command ran in.

    Used only for a successful answer — the point of asking in a shared
    channel or a DM with a colleague is usually for both people to see it.
    """
    return _reply("in_channel", text, replace_original=replace_original, blocks=blocks)


def _reply(
    response_type: str,
    text: str,
    *,
    replace_original: bool,
    blocks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    # `text` is always set, even when `blocks` is present — Slack renders it
    # in notifications, screen readers, and any client that doesn't support
    # Block Kit, so it must never be an empty fallback.
    payload: dict[str, Any] = {"response_type": response_type, "text": text}
    if blocks:
        payload["blocks"] = blocks
    if replace_original:
        payload["replace_original"] = True
    return payload


def _escaped(text: str) -> str:
    """``text`` shown as written: Slack's control sequences (``<!channel>``, a disguised
    ``<url|label>`` link, a mention) become plain characters (G48)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _preview(text: str) -> str:
    return text if len(text) <= _QUERY_PREVIEW_MAX_CHARS else text[:_QUERY_PREVIEW_MAX_CHARS] + "…"


async def handle_cognee_ask_share(response_url: str, answer_id: str) -> dict[str, Any]:
    """Handle the answer prompt's Share click — posts the question and its
    answer in_channel together, replacing the ephemeral review prompt. This
    is the first and only point either one becomes visible to anyone else.
    """
    pending = _pending_answers.pop(answer_id, None)
    if pending is None:
        return _ephemeral("This answer is no longer available.", replace_original=True)

    question, fallback_text, blocks = pending
    question_block = {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f'*Asked:* "{_escaped(_preview(question))}"'},
    }
    return _in_channel(
        f'Asked: "{_escaped(_preview(question))}"\n\n{fallback_text}',
        replace_original=True,
        blocks=[question_block, {"type": "divider"}, *blocks],
    )


async def handle_cognee_ask_discard(answer_id: str) -> dict[str, Any]:
    """Handle the answer prompt's Discard click — drops the answer for good."""
    _pending_answers.pop(answer_id, None)
    return _ephemeral("Discarded.", replace_original=True)


# The header of the Evidence block the engine appends to an answer asked with
# ``include_references`` (``append_chunk_evidence``): its bullets carry data and chunk
# ids, so the answer is shown without it and its documents as links (``_sources``).
_EVIDENCE_HEADER = "\n\nEvidence:"


def _extract_fact(result: Any) -> str | None:
    """Pull the answer text out of one ``cognee.search(..., verbose=True)`` result.

    Each result is a dict per dataset searched, with the completion in ``"text_result"``:
    a string or a list of strings (``HYBRID_COMPLETION`` returns ``["<answer>"]``), the
    first non-empty one is used, without the engine's Evidence block. Anything else is
    skipped rather than raising, since a shape this code doesn't recognize shouldn't
    crash the whole answer.
    """
    value = result.get("text_result") if isinstance(result, dict) else None
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item), None)
    if not isinstance(value, str):
        return None
    return value.split(_EVIDENCE_HEADER, 1)[0].strip() or None


# Slack's Block Kit limits: a text object holds at most 3,000 characters and a context
# block at most 10 elements. Over them Slack refuses the whole message, answer included.
_SLACK_TEXT_LIMIT = 3000
_CONTEXT_ELEMENTS = 10
_SOURCES_LEAD = "Sources: "


def _cited_chunks(result: Any) -> list[dict[str, Any]]:
    """The payloads of the chunks one answer read, in the order the search ranked them.

    ``objects_result`` holds them as a list (RAG) or under ``"chunks"`` (hybrid, temporal);
    anything else reads as no chunks.
    """
    objects = result.get("objects_result") if isinstance(result, dict) else None
    chunks = objects.get("chunks") if isinstance(objects, dict) else objects
    payloads = []
    for chunk in chunks if isinstance(chunks, list) else []:
        chunk_payload = (
            chunk.get("payload", chunk)
            if isinstance(chunk, dict)
            else getattr(chunk, "payload", None)
        )
        if isinstance(chunk_payload, dict):
            payloads.append(chunk_payload)
    return payloads


def _sources(result: Any) -> list[str]:
    """The documents one answer read, as Slack links, in element texts within Slack's limit.

    Read from the payloads of the chunks it read: a connector chunk carries its document's
    title (``document_title``) and link (``document_url``, a Slack permalink for a synced
    message), SDK-1500. Each document is named once, in the order the search ranked them
    (G10). A document without a title is left out: its stored file name tells a member
    nothing. Only an https link becomes one, and the title is escaped like an answer. Every
    source is shown: the links fill one element after another (G86); only a title that alone
    would pass Slack's limit is shortened.
    """
    links: list[str] = []
    seen: set[str] = set()
    for chunk_payload in _cited_chunks(result):
        title, url = chunk_payload.get("document_title"), chunk_payload.get("document_url")
        if not isinstance(title, str) or not title.strip():
            continue
        key = str(chunk_payload.get("document_id") or title)
        if key in seen:
            continue
        seen.add(key)
        # Slack's link syntax is ``<url|label>``: a URL carrying a space, ``<``, ``>`` or ``|``
        # would close it early (a ``<!channel>`` after it, a label of the writer's), so
        # such a source is shown by its title alone (G79).
        valid_url = (
            isinstance(url, str)
            and url.startswith("https://")
            and not any(character in url for character in " <>|")
        )
        frame = len(f"<{url}|>") if valid_url else 0
        label = _fitted(title.strip(), _SLACK_TEXT_LIMIT - len(_SOURCES_LEAD) - frame)
        links.append(f"<{url}|{label}>" if valid_url else label)

    elements: list[str] = []
    current = _SOURCES_LEAD
    for link in links:
        joined = current + link if current == _SOURCES_LEAD else f"{current} · {link}"
        if len(joined) <= _SLACK_TEXT_LIMIT:
            current = joined
        else:
            elements.append(current)
            current = link
    if current != _SOURCES_LEAD:
        elements.append(current)
    return elements


def _fitted(title: str, room: int) -> str:
    """``title`` escaped, shortened with an ellipsis when escaped it would pass ``room``."""
    label = _escaped(title)
    while len(label) > room and title:
        title = title[: len(title) - (len(label) - room) - 1].rstrip()
        label = _escaped(title) + "…"
    return label


# Substrings that mark a fact as "I can't answer this from the given
# context" rather than an actual answer. HYBRID_COMPLETION can return a
# separate per-chunk completion for each matched chunk group, and a chunk
# irrelevant to the question still produces its own such completion —
# _extract_fact has no way to tell that apart from a real answer (it's just
# as valid a non-empty string), so it's filtered out here instead.
_REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i'm unable to",
    "can't answer",
    "cannot answer",
    "no information about",
    "no relevant information",
    "doesn't contain",
    "does not contain",
    "contains no information",
    # A dataset without the answer talks about its context instead (live, 2026-11-10:
    # "The capital city is not specified in the provided context.").
    "not specified",
    "not mentioned",
    "does not mention",
    "doesn't mention",
    "provided context",
    "given context",
)


def _is_refusal(fact: str) -> bool:
    lowered = fact.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _format_answer(results: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Build an ``@cognee`` answer as ``(fallback_text, blocks)``.

    ``fallback_text`` is a plain-text summary for notifications/screen
    readers; ``blocks`` (Block Kit) is what actually renders when the client
    supports it — a header and one section per fact, each followed by the documents it
    read (``_sources``), with dividers between them. Presentation only: Share and
    Discard are added by the caller.
    """
    answers = [
        (_escaped(fact), _sources(result))
        for result in results
        if (fact := _extract_fact(result)) and not _is_refusal(fact)
    ][:3]
    if not answers:
        return "No relevant information found.", []

    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Answer", "emoji": True}},
    ]
    for fact, sources in answers:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": fact}})
        for first in range(0, len(sources), _CONTEXT_ELEMENTS):
            texts = sources[first : first + _CONTEXT_ELEMENTS]
            blocks.append(
                {"type": "context", "elements": [{"type": "mrkdwn", "text": t} for t in texts]}
            )
        blocks.append({"type": "divider"})
    blocks.pop()  # no trailing divider after the last fact

    facts = [fact for fact, _ in answers]
    fallback_text = "*Answer:*\n" + "\n\n".join(f"*{i}.* {fact}" for i, fact in enumerate(facts, 1))
    return fallback_text, blocks
