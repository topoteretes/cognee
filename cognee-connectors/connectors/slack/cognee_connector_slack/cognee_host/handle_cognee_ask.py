"""An ``@cognee`` answer: its text in Slack's Block Kit, and its rating.

Copied from cognee.modules.integrations.slack.handle_cognee_ask (cognee dev, 2026-11-12).

The LLM writes Markdown and Slack reads mrkdwn, so every answer goes through one renderer
(``to_mrkdwn``, and ``to_plain`` for the notification text). The answer is the thread reply
under the question: its paragraphs, one numbered "Sources · N" line per fact, a line saying
what it read, and 👍 / 👎 (``record_answer``, ``rate_answer``). ``handle_cognee_mention.ask``
posts it.
"""

import logging
import re
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload

from cognee_connector_slack.source import split_title

logger = logging.getLogger(__name__)

# Referenced by handle_slack_interactive.py; 👍 and 👎 on the 1..5 scale feedback uses.
RATE_UP_ACTION_ID = "cognee_answer_up"
RATE_DOWN_ACTION_ID = "cognee_answer_down"
RATINGS = {RATE_UP_ACTION_ID: 5, RATE_DOWN_ACTION_ID: 1}


def _escaped(text: str) -> str:
    """``text`` shown as written: Slack's control sequences (``<!channel>``, a disguised
    ``<url|label>`` link, a mention) become plain characters (G48)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _linkable(url: str) -> bool:
    """Only an https URL becomes a link. Slack's link syntax is ``<url|label>``: a URL
    carrying a space, ``<``, ``>`` or ``|`` (escaped or not) would close it early (G79)."""
    return url.startswith("https://") and not any(
        mark in url for mark in (" ", "<", ">", "|", "&lt;", "&gt;")
    )


# -- Markdown to mrkdwn --------------------------------------------------------------------

_ANSWER_HEADING = re.compile(r"\A\s*#{1,6}[ \t]*answer[ \t]*:?[ \t]*(?:\n+|\Z)", re.IGNORECASE)
_CODE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.DOTALL)
_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_BULLET = re.compile(r"^([ \t]*)[-*][ \t]+", re.MULTILINE)
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__")
_ITALIC = re.compile(r"(?<![*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![*\w])")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
# Stands in for Slack's bold ``*`` until italics are converted, so ``*x*`` from a bold or a
# heading is not read as Markdown italics.
_STRONG = "\x01"
_KEPT = re.compile("\x00(\\d+)\x00")


def to_mrkdwn(markdown: str) -> str:
    """``markdown`` as Slack mrkdwn, escaped like any text Cognee posts (G48).

    A heading becomes a bold line, and a leading one that only says "Answer" goes (the
    answer is already the reply to the question); bold, italics, strikethrough, bullets and
    https links are converted; code is kept as written.
    """
    return _render(markdown, plain=False)


def to_plain(markdown: str) -> str:
    """``markdown`` as plain text for a message's ``text`` (notifications, screen readers):
    the same rules as ``to_mrkdwn``, without any markers. Still escaped."""
    return _render(markdown, plain=True)


def _render(markdown: str, *, plain: bool) -> str:
    kept: list[str] = []

    def keep(text: str) -> str:
        kept.append(text)
        return f"\x00{len(kept) - 1}\x00"

    def code(match: re.Match) -> str:
        text = match.group(0)
        return keep(text.strip("`").strip("\n") if plain else text)

    def link(match: re.Match) -> str:
        label, url = _inline(match.group(1), plain), match.group(2)
        return keep(f"<{url}|{label}>" if _linkable(url) and not plain else label)

    text = _escaped(_ANSWER_HEADING.sub("", markdown, count=1))
    text = _CODE.sub(code, text)
    text = _LINK.sub(link, text)
    text = _HEADING.sub(lambda match: _strong(match.group(1), plain), text)
    text = _BULLET.sub(lambda match: f"{match.group(1)}• ", text)
    text = _inline(text, plain)
    # A kept piece can hold another (a link label with code in it): restore until none.
    while _KEPT.search(text):
        text = _KEPT.sub(lambda match: kept[int(match.group(1))], text)
    return text.strip()


def _strong(text: str, plain: bool) -> str:
    return text if plain else f"{_STRONG}{text}{_STRONG}"


def _inline(text: str, plain: bool) -> str:
    text = _BOLD.sub(lambda match: _strong(match.group(1) or match.group(2), plain), text)
    text = _ITALIC.sub(lambda match: match.group(1) if plain else f"_{match.group(1)}_", text)
    text = _STRIKE.sub(lambda match: match.group(1) if plain else f"~{match.group(1)}~", text)
    return text.replace(_STRONG, "*")


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
_SOURCES_LEAD = "Sources · {count}"
_SOURCES_GAP = "   "
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


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
    """The documents one answer read, numbered, in element texts within Slack's limit.

    Read from the payloads of the chunks it read: a connector chunk carries its document's
    title (``document_title``), link (``document_url``, a Slack permalink for a synced
    message, SDK-1500) and where it sits (``document_parent``, a channel's ``#name``). Each
    document is named once, in the order the search ranked them (G10), as
    ``1 <link|#channel · author · Nov 12>`` (``_label``). A document without a title is
    left out: its stored file name tells a member nothing. Every source is shown: the
    numbers fill one element after another (G86); only a label that alone would pass
    Slack's limit is shortened.
    """
    entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for chunk_payload in _cited_chunks(result):
        title, url = chunk_payload.get("document_title"), chunk_payload.get("document_url")
        if not isinstance(title, str) or not title.strip():
            continue
        key = str(chunk_payload.get("document_id") or title)
        if key in seen:
            continue
        seen.add(key)
        valid_url = url if isinstance(url, str) and _linkable(url) else None
        entries.append(
            (_label(title.strip(), valid_url, chunk_payload.get("document_parent")), valid_url)
        )
    if not entries:
        return []

    lead = _SOURCES_LEAD.format(count=len(entries))
    links = []
    for number, (label, url) in enumerate(entries, 1):
        frame = len(f"{number} <{url}|>") if url else len(f"{number} ")
        fitted = _fitted(label, _SLACK_TEXT_LIMIT - len(lead) - len(_SOURCES_GAP) - frame)
        links.append(f"{number} <{url}|{fitted}>" if url else f"{number} {fitted}")

    elements: list[str] = []
    current = lead
    for link in links:
        joined = f"{current}{_SOURCES_GAP}{link}"
        if len(joined) <= _SLACK_TEXT_LIMIT:
            current = joined
        else:
            elements.append(current)
            current = link
    elements.append(current)
    return elements


def _label(title: str, url: str | None, parent: Any) -> str:
    """Where, who and when for a synced Slack message (``#general · ana · Nov 5``), read
    from its title (``split_title``) and its channel; any other document by its title."""
    parts = split_title(title) if url and "slack.com/archives/" in url else None
    if parts is None:
        return title
    _, author, day = parts
    where = [parent] if isinstance(parent, str) and parent else []
    return " · ".join([*where, author, _day(day)])


def _day(day: date) -> str:
    """``Nov 5``, with the year when it is not this one. Month names are fixed, not the
    process locale's."""
    shown = f"{_MONTHS[day.month - 1]} {day.day}"
    return shown if day.year == datetime.now(timezone.utc).year else f"{shown}, {day.year}"


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
            # Never inside a link (``<url|a label>``): Slack would show both halves raw.
            opened = text.rfind("<", 0, cut)
            if opened > text.rfind(">", 0, cut) and opened > 0:
                cut = opened
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

    ``blocks`` is each fact rendered as mrkdwn (``to_mrkdwn``) in as many sections as it
    needs (``_sections``), each followed by the documents it read (``_sources``), with a
    divider between facts. No header: the answer is the reply to the question.
    ``fallback_text`` is the same facts as plain text (``to_plain``). Only the results of a
    search that retrieved something are shown (``_retrieved_something``).
    """
    answers = [
        (fact, _sources(result))
        for result in results
        if isinstance(result, dict)
        and _retrieved_something(result)
        and (fact := _extract_fact(result))
    ][:3]
    if not answers:
        return "No relevant information found.", []

    blocks: list[dict[str, Any]] = []
    for fact, sources in answers:
        # Slack refuses the whole message over its text limit, so a long answer takes as
        # many sections as it needs (at most 50 blocks a message: about 140,000 characters
        # over three answers before Slack would refuse it).
        blocks += [
            {"type": "section", "text": {"type": "mrkdwn", "text": piece}}
            for piece in _sections(to_mrkdwn(fact))
        ]
        for first in range(0, len(sources), _CONTEXT_ELEMENTS):
            texts = sources[first : first + _CONTEXT_ELEMENTS]
            blocks.append(
                {"type": "context", "elements": [{"type": "mrkdwn", "text": t} for t in texts]}
            )
        blocks.append({"type": "divider"})
    blocks.pop()  # no trailing divider after the last fact
    return "\n\n".join(to_plain(fact) for fact, _ in answers), blocks


def finish_answer(
    blocks: list[dict[str, Any]], *, footer: str, rating_ref: str | None
) -> list[dict[str, Any]]:
    """``blocks`` with the line saying what the answer read, and 👍 / 👎 when the answer
    could be kept for rating (``rating_ref``, the answer's id from ``record_answer``)."""
    finished = [*blocks, {"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]}]
    if rating_ref is None:
        return finished
    buttons = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": emoji, "emoji": True},
            "action_id": action_id,
            "value": rating_ref,
        }
        for emoji, action_id in (("👍", RATE_UP_ACTION_ID), ("👎", RATE_DOWN_ACTION_ID))
    ]
    return [*finished, {"type": "actions", "elements": buttons}]


# -- Rating ------------------------------------------------------------------------------


def answer_session_id(team_id: str, member: str) -> str:
    """The session the answers given to one Slack member are recorded in, for rating."""
    return f"slack:{team_id}:{member}"


async def record_answer(
    credential: Any,
    member: str,
    *,
    question: str,
    answer: str,
    used_graph_element_ids: dict[str, list[str]] | None,
) -> str | None:
    """Record a posted answer as session Q&A and return its id, the rating buttons' value;
    ``None`` with the session cache off (``CACHING=false``).

    Kept under the workspace's owner, who owns the workspace dataset the feedback weights
    are written to (``apply_answer_feedback``), in the asker's own session, so only the
    asker can rate it (``rate_answer``). The cache's own expiry applies; nothing here adds
    one.
    """
    return await get_session_manager().add_qa(
        user_id=str(credential.user_id),
        question=question,
        context="",
        answer=answer,
        session_id=answer_session_id(str(credential.provider_account_id), member),
        used_graph_element_ids=used_graph_element_ids,
    )


async def rate_answer(credential: Any, member: str, qa_id: str, score: int) -> bool:
    """Record ``member``'s rating (1..5) of the answer ``qa_id`` and apply it to the graph.

    ``member`` is the Slack user of the signed click, never anything the button carries: an
    answer is found only in its asker's own session, so anyone else's click, or a forged
    id, finds nothing and returns ``False``. A second click stores its score in place of
    the first. The feedback stage applies a changed score as a new step from the current
    weights, without undoing the first; the same score again changes nothing.
    """
    rated = await get_session_manager().add_feedback(
        user_id=str(credential.user_id),
        qa_id=qa_id,
        feedback_score=score,
        session_id=answer_session_id(str(credential.provider_account_id), member),
    )
    if rated:
        await apply_answer_feedback(credential, member)
    return rated


async def apply_answer_feedback(credential: Any, member: str) -> None:
    """Move the feedback weights of the graph elements ``member``'s rated answers used.

    The improve stage's own pipeline (``feedback_weights``), run as the workspace's owner
    over the workspace dataset; elements of another dataset (a direct message's answer
    reads the member's own too) are not in it and are left alone. The rating is stored
    whatever happens here, so a failure is logged; nothing applies it until the member
    rates that session again.
    """
    from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline
    from cognee.modules.users.methods import get_user

    from cognee_connector_slack.cognee_host.sync import dataset_name_for_credential

    team_id = str(credential.provider_account_id)
    try:
        await apply_feedback_weights_pipeline(
            user=await get_user(credential.user_id),
            session_ids=[answer_session_id(team_id, member)],
            dataset=dataset_name_for_credential(credential),
        )
    except Exception:  # the rating is stored; applying it waits for the next rating
        logger.exception("Applying a Slack answer's rating failed for team %s", team_id)


async def forget_answer_records(team_id: str, owner_id: UUID, *, every_owner: bool = False) -> None:
    """Delete the answers recorded for rating from ``team_id``'s workspace that are kept
    under ``owner_id``, or under any owner the workspace had with ``every_owner``.

    A channel answer's record quotes the workspace's memory and lives in the session cache
    and its vector index, outside every dataset, so deleting the datasets leaves it. The
    owner's records are listed from the cache itself, since their ``session_records`` rows
    are written best-effort (as ``forget(everything=True)`` does); those rows still find the
    records kept under an earlier owner. Nothing is recorded while the session cache is off.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.delete_session import delete_sessions
    from cognee.modules.session_lifecycle.models import SessionRecord
    from sqlalchemy import select

    sessions_cache = get_session_manager()
    if not sessions_cache.is_available:
        return
    prefix = answer_session_id(team_id, "")
    sessions = {
        (owner_id, session)
        for session in await sessions_cache.list_session_ids(user_id=str(owner_id))
        if session.startswith(prefix)
    }
    query = select(SessionRecord.user_id, SessionRecord.session_id).where(
        SessionRecord.session_id.startswith(prefix, autoescape=True)
    )
    if not every_owner:
        query = query.where(SessionRecord.user_id == owner_id)
    async with get_relational_engine().get_async_session() as db:
        sessions.update((owner, session) for owner, session in (await db.execute(query)).all())
    if sessions:
        await delete_sessions(sorted(sessions, key=str))
