"""Unit tests for modules.integrations.slack.handle_cognee_ask: the answer review of
``@cognee <question>`` (Share and Discard) and how an answer's text is built from the
search results. Asking itself is ``handle_cognee_mention.ask`` (test_handle_cognee_mention).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.credentials import STATUS_ACTIVE
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
    _format_answer,
    _pending_answers,
    handle_cognee_ask_discard,
    handle_cognee_ask_share,
)
from cognee.modules.search.types import SearchType

MODULE = "cognee.modules.integrations.slack.handle_cognee_ask"


def _body(
    team_id: str = "T123",
    text: str = "what happened in Q3?",
    response_url: str = "https://hooks.slack.com/x",
    user_id: str = "U100",
) -> bytes:
    return urlencode(
        {"team_id": team_id, "text": text, "response_url": response_url, "user_id": user_id}
    ).encode()


def _credential(user_id=None):
    return SimpleNamespace(
        tenant_id=uuid4(), user_id=user_id or uuid4(), provider_metadata={}, status=STATUS_ACTIVE
    )


# ── handle_cognee_ask: fast path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_share_posts_the_question_and_answer_in_channel():
    _pending_answers["abc123"] = (
        "what happened in Q3?",
        "*Answer:*\nfact one",
        [{"type": "section"}],
    )

    response = await handle_cognee_ask_share("https://hooks.slack.com/x", "abc123")

    assert response["response_type"] == "in_channel"
    assert response["replace_original"] is True
    assert "what happened in Q3?" in response["text"]
    assert "fact one" in response["text"]
    # The question is its own block, ahead of the answer's own blocks —
    # nothing about either was visible to the channel before this.
    assert response["blocks"][0]["text"]["text"] == '*Asked:* "what happened in Q3?"'
    assert any(b.get("type") == "section" for b in response["blocks"][1:])
    assert "abc123" not in _pending_answers  # popped, not left behind


@pytest.mark.asyncio
async def test_share_of_unknown_answer_id_is_a_clean_ephemeral_message():
    # Server restart, double-click, or an expired/already-handled id.
    response = await handle_cognee_ask_share("https://hooks.slack.com/x", "does-not-exist")

    assert response["response_type"] == "ephemeral"
    assert "no longer available" in response["text"]


@pytest.mark.asyncio
async def test_discard_drops_the_answer_and_replies_ephemeral():
    _pending_answers["abc456"] = (
        "what happened in Q3?",
        "*Answer:*\nfact one",
        [{"type": "section"}],
    )

    response = await handle_cognee_ask_discard("abc456")

    assert response == {
        "response_type": "ephemeral",
        "text": "Discarded.",
        "replace_original": True,
    }
    assert "abc456" not in _pending_answers


# ── _format_answer ──────────────────────────────────────────────────────────
#
# ``@cognee`` asks ``cognee.search(..., verbose=True, include_references=True)``, which
# returns one dict per dataset searched with the answer in "text_result" and the
# documents it read in "evidence", whether backend access control is on or off.


def _answer(text, evidence=None):
    return {"text_result": text, "evidence": evidence or []}


def _cited(data_id, title=None, url=None):
    return {"kind": "segment", "data_id": data_id, "title": title, "url": url}


def test_format_answer_fallback_text_joins_up_to_three_results():
    results = [_answer(f"fact {i}") for i in range(5)]
    fallback_text, _ = _format_answer(results)
    assert fallback_text.startswith("*Answer:*")
    assert "fact 0" in fallback_text and "fact 2" in fallback_text
    assert "fact 3" not in fallback_text


def test_format_answer_blocks_have_one_section_per_fact_with_dividers_between():
    results = [_answer(f"fact {i}") for i in range(3)]
    _, blocks = _format_answer(results)

    assert blocks[0] == {
        "type": "header",
        "text": {"type": "plain_text", "text": "Answer", "emoji": True},
    }
    sections = [b for b in blocks if b["type"] == "section"]
    assert [s["text"]["text"] for s in sections] == ["fact 0", "fact 1", "fact 2"]
    # Dividers sit between facts, never trailing after the last one.
    assert blocks[-1]["type"] == "section"


def test_an_answer_carries_no_footer_counting_its_results():
    """Goran, 2026-11-10: "Searched across your Cognee memory · 1 result" told the member
    nothing; the answer is the facts and the Share and Discard buttons."""
    _, blocks = _format_answer([_answer("fact 0"), _answer("fact 1")])
    assert not [block for block in blocks if block["type"] == "context"]


def test_format_answer_unwraps_answer_lists():
    # HYBRID_COMPLETION wraps the answer in a single-element list.
    results = [_answer(["fact 0"]), _answer(["fact 1"])]
    fallback_text, blocks = _format_answer(results)
    sections = [b for b in blocks if b["type"] == "section"]
    assert [s["text"]["text"] for s in sections] == ["fact 0", "fact 1"]
    assert "fact 0" in fallback_text


def test_format_answer_empty_results():
    fallback_text, blocks = _format_answer([])
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_format_answer_results_without_an_answer():
    fallback_text, blocks = _format_answer([_answer(None)])
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_format_answer_skips_unrecognized_result_shapes():
    fallback_text, blocks = _format_answer([object(), 42, None])
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_format_answer_filters_out_refusal_facts():
    # A chunk irrelevant to the question produces its own "I can't answer"
    # completion alongside a real answer from a relevant chunk — the
    # refusal must not show up as if it were a second fact.
    results = [
        _answer("LLMs are neural language models built on Transformers."),
        _answer(
            "I can't—the provided context only covers the 2026 FIFA World Cup "
            "and contains no information about LLMs."
        ),
    ]
    fallback_text, blocks = _format_answer(results)
    sections = [b for b in blocks if b["type"] == "section"]
    assert len(sections) == 1
    assert "Transformers" in sections[0]["text"]["text"]
    assert "World Cup" not in fallback_text


def test_format_answer_all_refusals_reads_as_no_relevant_information():
    results = [_answer("I cannot answer this from the given context.")]
    fallback_text, blocks = _format_answer(results)
    assert fallback_text == "No relevant information found."
    assert blocks == []


@pytest.mark.asyncio
async def test_a_shared_answer_cannot_ping_the_channel_or_disguise_a_link():
    """Share posts in the channel: Slack's control sequences in the question or in an
    answer that quotes memory are escaped, so they show as text (G48)."""
    fallback, blocks = _format_answer(
        [_answer(["See <https://evil.example/?q=x|the runbook> & <!here>."])]
    )
    answer_id = uuid4().hex
    _pending_answers[answer_id] = ("<!channel> what is the salary?", fallback, blocks)

    shared = await handle_cognee_ask_share("https://hooks.slack.com/actions/T/1/x", answer_id)

    posted = json.dumps(shared)
    for control in ("<!channel>", "<!here>", "<https://"):
        assert control not in posted
    assert "&lt;!channel&gt;" in posted and "&amp;" in posted


def test_a_dataset_that_does_not_know_does_not_hide_the_one_that_does():
    """Live, 2026-11-10: every readable dataset answers on its own; the synced channel said
    "The capital city is not specified in the provided context." next to the note's
    "Berlin.", and that sentence was shown as a fact."""
    _, blocks = _format_answer(
        [
            _answer(["The capital city is not specified in the provided context."]),
            _answer(["Berlin."]),
            _answer(["The given context does not mention a capital."]),
            _answer(["That is not mentioned in the context."]),
        ]
    )
    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert sections == ["Berlin."]


PERMALINK = "https://acme.slack.com/archives/C1/p1793872900000100"
TITLE = "Who owns the runbook? — Ana Kovac, 2026-11-05"


def test_an_answer_names_the_documents_it_read_as_links():
    """G10: a member sees where an answer came from. The engine's Evidence block, with
    data and chunk ids, is not shown; each document read is one link, once."""
    evidence = [
        _cited("d1", TITLE, PERMALINK),
        _cited("d1", TITLE, PERMALINK),
        _cited("d2", "Deploys — Ben Ross, 2026-11-06", PERMALINK.replace("p17", "p18")),
        _cited("d3"),
    ]
    block_text = f'Evidence:\n- chunk 1 of "{TITLE}" <{PERMALINK}> (data_id: d1, chunk_id: c1)'
    fallback, blocks = _format_answer([_answer([f"Ben owns it.\n\n{block_text}"], evidence)])

    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert sections == ["Ben owns it."]
    (sources,) = [block for block in blocks if block["type"] == "context"]
    # One element: a context block takes at most ten, and no document is dropped.
    second = f"<{PERMALINK.replace('p17', 'p18')}|Deploys — Ben Ross, 2026-11-06>"
    assert [element["text"] for element in sources["elements"]] == [
        f"Sources: <{PERMALINK}|{TITLE}> · {second}"
    ]
    assert "data_id" not in fallback and "Evidence" not in fallback


def test_a_source_cannot_ping_or_disguise_a_link():
    """A title is the first words of a Slack message: its control characters show as text,
    and only an https link becomes one."""
    evidence = [
        _cited("d1", "<!here> & <https://evil.example|x>", PERMALINK),
        _cited("d2", "No link", "javascript:alert(1)"),
    ]
    _, blocks = _format_answer([_answer(["Yes."], evidence)])

    (sources,) = [block for block in blocks if block["type"] == "context"]
    texts = [element["text"] for element in sources["elements"]]
    assert texts == [
        f"Sources: <{PERMALINK}|&lt;!here&gt; &amp; &lt;https://evil.example|x&gt;> · No link"
    ]


def test_a_dataset_that_does_not_know_shows_no_sources():
    evidence = [_cited("d1", TITLE, PERMALINK)]
    _, blocks = _format_answer(
        [_answer(["That is not mentioned in the context."], evidence), _answer(["Berlin."])]
    )
    assert not [block for block in blocks if block["type"] == "context"]


@pytest.mark.parametrize(
    "url",
    ["https://evil.example/x><!channel>", "https://evil.example/|Board minutes", "https://a/<b"],
)
def test_a_source_link_that_could_break_out_of_its_brackets_is_shown_as_its_title(url):
    """G79: a dataset writer's own ``url`` with ``>``, ``<`` or ``|`` closed Slack's link and
    pinged the channel or disguised the link once the answer was shared."""
    _, blocks = _format_answer([_answer(["Yes."], [_cited("d1", "Q3 report", url)])])

    (sources,) = [block for block in blocks if block["type"] == "context"]
    assert sources["elements"][0]["text"] == "Sources: Q3 report"


def test_many_or_long_sources_stay_within_slacks_text_limit_and_none_is_dropped():
    """G86: the sources were one text of any length, and Slack refuses a block text over
    3,000 characters, so the whole answer was lost. Every source is still shown, split over
    elements (at most ten to a context block); only a title longer than a whole element is
    shortened."""
    long_titles = [
        _cited(f"d{i}", f"{'Board minutes & notes ' * 9}{i}", PERMALINK) for i in range(15)
    ]
    huge = _cited("dx", "x" * 3100, PERMALINK)
    _, blocks = _format_answer([_answer(["Yes."], [*long_titles, huge])])

    contexts = [block for block in blocks if block["type"] == "context"]
    elements = [element["text"] for block in contexts for element in block["elements"]]
    assert all(len(text) <= 3000 for text in elements)
    assert all(len(block["elements"]) <= 10 for block in contexts)
    shown = " ".join(elements)
    for i in range(15):
        assert f"&amp; notes {i}>" in shown or f"notes {i}>" in shown
    assert "…" in shown
