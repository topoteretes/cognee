"""The answer review of ``@cognee <question>``: Share and Discard, and the answer's text.

Nothing about a question becomes visible to the channel until the asker has seen the
answer and chosen to share it. ``handle_cognee_mention.ask`` shows the answer privately
with Share and Discard; Share is the only thing that ever posts ``in_channel`` (question
and answer together, since nothing was public before it), Discard drops it unseen.
"""

import logging
import re
from typing import Any
from uuid import UUID

from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.integrations.slack.response_url import post_to_response_url
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload

logger = logging.getLogger(__name__)

# Referenced by handle_slack_interactive.py to dispatch the answer review
# prompt's button clicks back here.
ASK_SHARE_ACTION_ID = "cognee_ask_share"
ASK_DISCARD_ACTION_ID = "cognee_ask_discard"


# The button value that names an answer kept in the session cache: the owner's user id,
# the asker's session and the QA id, never the text (Slack caps a value at 2000 chars).
_ANSWER_REF_SEPARATOR = "|"

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


def ask_session_id(team_id: str, member: str) -> str:
    """The session an ``@cognee`` member's answers are kept in, one per Slack member."""
    return f"slack:{team_id}:{member}"


def _answer_text(blocks: list[dict[str, Any]]) -> str:
    """What the member was shown, answer and sources, as one mrkdwn text."""
    texts = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    texts += [
        element["text"]
        for block in blocks
        if block["type"] == "context"
        for element in block["elements"]
    ]
    return "\n\n".join(texts)


async def keep_answer(
    owner_user_id: UUID, session_id: str, question: str, blocks: list[dict[str, Any]]
) -> str | None:
    """Record a shown answer as session Q&A and return the reference Share and Discard
    carry, or ``None`` when the session cache is off (``CACHING=false``). The cache's own
    expiry applies; nothing here adds one. Any process can read it back, so Share works
    after a restart or on another worker."""
    qa_id = await get_session_manager().add_qa(
        user_id=str(owner_user_id),
        question=question,
        context="",
        answer=_answer_text(blocks),
        session_id=session_id,
    )
    if qa_id is None:
        return None
    return _ANSWER_REF_SEPARATOR.join((str(owner_user_id), session_id, qa_id))


def _answer_ref(value: str) -> tuple[str, str, str] | None:
    parts = value.split(_ANSWER_REF_SEPARATOR)
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


async def handle_cognee_ask_share(response_url: str, answer_ref: str) -> dict[str, Any]:
    """Handle the answer prompt's Share click — posts the question and its
    answer in_channel together, replacing the ephemeral review prompt. This
    is the first and only point either one becomes visible to anyone else.
    The kept answer is removed, so a second click shares nothing twice.
    """
    ref = _answer_ref(answer_ref)
    entries = []
    if ref is not None:
        user_id, session_id, qa_id = ref
        manager = get_session_manager()
        entries = await manager.get_session_entries_by_ids(
            user_id=user_id, qa_ids=[qa_id], session_id=session_id
        )
        await manager.delete_qa(user_id=user_id, qa_id=qa_id, session_id=session_id)
    if not entries:
        return _ephemeral("This answer is no longer available.", replace_original=True)

    question, answer = entries[0].question, entries[0].answer
    question_block = {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f'*Asked:* "{_escaped(_preview(question))}"'},
    }
    return _in_channel(
        f'Asked: "{_escaped(_preview(question))}"\n\n{answer}',
        replace_original=True,
        blocks=[
            question_block,
            {"type": "divider"},
            *(
                {"type": "section", "text": {"type": "mrkdwn", "text": piece}}
                for piece in _sections(answer)
            ),
        ],
    )


async def handle_cognee_ask_discard(answer_ref: str) -> dict[str, Any]:
    """Handle the answer prompt's Discard click — drops the answer for good."""
    ref = _answer_ref(answer_ref)
    if ref is not None:
        user_id, session_id, qa_id = ref
        await get_session_manager().delete_qa(user_id=user_id, qa_id=qa_id, session_id=session_id)
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


def _retrieved_something(result: dict[str, Any]) -> bool:
    """Whether the search of one dataset found anything to answer from.

    A completion is written even over an empty context ("the provided context does not
    mention..."), so only the retrieval context tells an answer from a miss, whatever the
    answer says. The rule is the search's own (``SearchResultPayload.retrieved_nothing``),
    applied to the verbose result's fields.
    """
    return not SearchResultPayload.model_construct(
        context=result.get("context_result"),
        completion=result.get("text_result"),
        result_object=result.get("objects_result"),
        error=result.get("error"),
    ).retrieved_nothing()


_WHITESPACE = re.compile(r"\s")


def _sections(text: str) -> list[str]:
    """``text`` as consecutive pieces within Slack's text limit, nothing left out.

    Cut at the last whitespace that fits; a run of more characters than the limit without
    any is cut at the limit, never inside an escape such as ``&amp;``.
    """
    pieces = []
    while len(text) > _SLACK_TEXT_LIMIT:
        spaces = [match.start() for match in _WHITESPACE.finditer(text, 1, _SLACK_TEXT_LIMIT + 1)]
        if spaces:
            cut = spaces[-1]
        else:
            cut = _SLACK_TEXT_LIMIT
            escape = text.rfind("&", cut - len("&amp;"), cut)
            if escape > 0 and ";" not in text[escape:cut]:
                cut = escape
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [*pieces, text] if text else pieces


def _format_answer(results: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Build an ``@cognee`` answer as ``(fallback_text, blocks)``.

    ``fallback_text`` is a plain-text summary for notifications/screen
    readers; ``blocks`` (Block Kit) is what actually renders when the client
    supports it — a header and the sections of each fact (``_sections``), each followed by
    the documents it read (``_sources``), with dividers between them. Only the results of a
    search that retrieved something are shown (``_retrieved_something``). Presentation
    only: Share and Discard are added by the caller.
    """
    answers = [
        (_escaped(fact), _sources(result))
        for result in results
        if isinstance(result, dict)
        and _retrieved_something(result)
        and (fact := _extract_fact(result))
    ][:3]
    if not answers:
        return "No relevant information found.", []

    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Answer", "emoji": True}},
    ]
    for fact, sources in answers:
        # Slack refuses the whole message over its text limit, so a long answer takes as
        # many sections as it needs (at most 50 blocks a message: about 140,000 characters
        # over three answers before Slack would refuse it).
        blocks += [
            {"type": "section", "text": {"type": "mrkdwn", "text": piece}}
            for piece in _sections(fact)
        ]
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
