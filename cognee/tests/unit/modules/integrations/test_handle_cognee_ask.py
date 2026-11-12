"""Unit tests for modules.integrations.slack.handle_cognee_ask: how an ``@cognee`` answer's
text is built from the search results. Asking itself is ``handle_cognee_mention.ask``
(test_handle_cognee_mention).
"""

import json

import pytest

from cognee.modules.integrations.slack.handle_cognee_ask import _format_answer

# ── _format_answer ──────────────────────────────────────────────────────────
#
# ``@cognee`` asks ``cognee.search(..., verbose=True, include_references=True)``, which
# returns one dict per dataset searched with the answer in "text_result" and the chunks it
# read in "objects_result" (``{"chunks": [...]}`` for hybrid, a list for RAG). A connector
# chunk's payload carries its document's ``document_title`` and ``document_url``.


def _answer(text, cited=None):
    return {"text_result": text, "objects_result": {"chunks": cited or []}}


def _missed(text):
    """A dataset whose retrieval found nothing: its completion still talks."""
    return {"text_result": text, "context_result": "", "objects_result": {"chunks": []}}


def _cited(document_id, title=None, url=None):
    payload = {"document_id": document_id, "text": "a chunk"}
    if title is not None:
        payload["document_title"] = title
    if url is not None:
        payload["document_url"] = url
    return {"id": f"chunk-{document_id}", "payload": payload}


def test_format_answer_fallback_text_joins_up_to_three_results():
    results = [_answer(f"fact {i}") for i in range(5)]
    fallback_text, _ = _format_answer(results)
    assert fallback_text == "fact 0\n\nfact 1\n\nfact 2"


def test_format_answer_blocks_have_one_section_per_fact_with_dividers_between():
    results = [_answer(f"fact {i}") for i in range(3)]
    _, blocks = _format_answer(results)

    assert [block["type"] for block in blocks] == [
        "section",
        "divider",
        "section",
        "divider",
        "section",
    ]
    sections = [b for b in blocks if b["type"] == "section"]
    assert [s["text"]["text"] for s in sections] == ["fact 0", "fact 1", "fact 2"]
    # Dividers sit between facts, never trailing after the last one.
    assert blocks[-1]["type"] == "section"


def test_an_answer_carries_no_footer_counting_its_results():
    """Goran, 2026-11-10: "Searched across your Cognee memory · 1 result" told the member
    nothing; the answer is the facts."""
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


def test_format_answer_leaves_out_a_dataset_that_retrieved_nothing():
    # A dataset whose retrieval found nothing still writes a completion; it must not show
    # up as if it were a second fact, whatever it says.
    results = [
        _answer("LLMs are neural language models built on Transformers."),
        _missed("The 2026 FIFA World Cup is in North America."),
    ]
    fallback_text, blocks = _format_answer(results)
    sections = [b for b in blocks if b["type"] == "section"]
    assert len(sections) == 1
    assert "Transformers" in sections[0]["text"]["text"]
    assert "World Cup" not in fallback_text


def test_format_answer_all_misses_reads_as_no_relevant_information():
    results = [_missed("I cannot answer this from the given context.")]
    fallback_text, blocks = _format_answer(results)
    assert fallback_text == "No relevant information found."
    assert blocks == []


def test_a_posted_answer_cannot_ping_the_channel_or_disguise_a_link():
    """The answer is posted for everyone in the channel: Slack's control sequences in an
    answer that quotes memory are escaped, so they show as text (G48)."""
    fallback_text, blocks = _format_answer(
        [_answer(["See <https://evil.example/?q=x|the runbook> & <!here>."])]
    )

    posted = json.dumps([fallback_text, blocks])
    for control in ("<!here>", "<https://"):
        assert control not in posted
    assert "&lt;!here&gt;" in posted and "&amp;" in posted


def test_a_dataset_that_does_not_know_does_not_hide_the_one_that_does():
    """Live, 2026-11-10: every readable dataset answers on its own; the synced channel said
    "The capital city is not specified in the provided context." next to the note's
    "Berlin.", and that sentence was shown as a fact. A dataset that retrieved nothing is
    left out by its empty context, not by what its completion says (F1)."""
    _, blocks = _format_answer(
        [
            _missed(["The capital city is not specified in the provided context."]),
            {**_answer(["Berlin."]), "context_result": "The capital of Germany is Berlin."},
            _missed(["The given context does not mention a capital."]),
            {"text_result": ["A capital."], "error": "no searchable memory"},
        ]
    )
    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert sections == ["Berlin."]


PERMALINK = "https://acme.slack.com/archives/C1/p1793872900000100"
TITLE = "Who owns the runbook? — Ana Kovac, 2026-11-05"


def _shown_day(shown: str, year: int) -> str:
    """How a source's day reads: ``Nov 5``, with the year only when it is not this one."""
    from datetime import datetime, timezone

    return shown if datetime.now(timezone.utc).year == year else f"{shown}, {year}"


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
    second = f"<{PERMALINK.replace('p17', 'p18')}|Ben Ross · {_shown_day('Nov 6', 2026)}>"
    assert [element["text"] for element in sources["elements"]] == [
        f"Sources · 2   1 <{PERMALINK}|Ana Kovac · {_shown_day('Nov 5', 2026)}>   2 {second}"
    ]
    assert "data_id" not in fallback and "Evidence" not in fallback


def test_a_rag_answer_names_its_documents_too():
    """RAG returns its chunks as a list, not under "chunks"."""
    result = {"text_result": ["Ben owns it."], "objects_result": [_cited("d1", TITLE, PERMALINK)]}
    _, blocks = _format_answer([result])

    (sources,) = [block for block in blocks if block["type"] == "context"]
    assert sources["elements"][0]["text"] == (
        f"Sources · 1   1 <{PERMALINK}|Ana Kovac · {_shown_day('Nov 5', 2026)}>"
    )


def test_structured_evidence_names_no_source():
    """SDK-1500: the title and link are on the chunk payload only; evidence keeps the
    stored file name, which tells a member nothing."""
    evidence = [{"kind": "segment", "data_id": "d1", "title": TITLE, "url": PERMALINK}]
    _, blocks = _format_answer([{"text_result": ["Yes."], "evidence": evidence}])

    assert not [block for block in blocks if block["type"] == "context"]


def test_a_source_cannot_ping_or_disguise_a_link():
    """A title is stored text: its control characters show as text, and only an https link
    becomes one. A title that is not the Slack connector's is shown whole."""
    evidence = [
        _cited("d1", "<!here> & <https://evil.example|x>", PERMALINK),
        _cited("d2", "No link", "javascript:alert(1)"),
    ]
    _, blocks = _format_answer([_answer(["Yes."], evidence)])

    (sources,) = [block for block in blocks if block["type"] == "context"]
    texts = [element["text"] for element in sources["elements"]]
    assert texts == [
        (
            f"Sources · 2   1 <{PERMALINK}|&lt;!here&gt; &amp; &lt;https://evil.example|x&gt;>"
            "   2 No link"
        )
    ]


def test_a_dataset_that_does_not_know_shows_no_sources():
    _, blocks = _format_answer(
        [_missed(["That is not mentioned in the context."]), _answer(["Berlin."])]
    )
    assert not [block for block in blocks if block["type"] == "context"]


@pytest.mark.parametrize(
    "url",
    ["https://evil.example/x><!channel>", "https://evil.example/|Board minutes", "https://a/<b"],
)
def test_a_source_link_that_could_break_out_of_its_brackets_is_shown_as_its_title(url):
    """G79: a dataset writer's own ``url`` with ``>``, ``<`` or ``|`` closed Slack's link and
    pinged the channel or disguised the link in the posted answer."""
    _, blocks = _format_answer([_answer(["Yes."], [_cited("d1", "Q3 report", url)])])

    (sources,) = [block for block in blocks if block["type"] == "context"]
    assert sources["elements"][0]["text"] == "Sources · 1   1 Q3 report"


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


# -- Review 2026-11-11 (F1): red until fixed ---------------------------------------------


def test_a_long_answer_stays_within_slacks_text_limit_and_loses_nothing():
    """F1: only the sources were held to Slack's 3,000-character text limit. A longer answer
    made Slack refuse the whole message, so the member saw "Searching…" and then nothing."""
    words = [f"word{i}" for i in range(900)]
    _, blocks = _format_answer([_answer([" ".join(words)])])

    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert sections and all(len(text) <= 3000 for text in sections)
    shown = " ".join(sections)
    assert all(word in shown for word in words)


def test_an_answer_that_uses_a_refusal_phrase_is_still_shown():
    """F1: refusals were told apart by phrases ("not mentioned", "provided context"), so a
    real answer that uses one was dropped as if the dataset did not know."""
    answer = "The launch date was not mentioned until Friday; it is March 3."
    _, blocks = _format_answer([_answer([answer])])

    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert sections == [answer]


# -- The answer's design (SDK-1502, Goran 2026-11-12): red until fixed --------------------
#
# Modelled on cognee's own Search answer: the answer in plain paragraphs, one compact
# "Sources · N" line, a scope line and the rating buttons. The LLM writes Markdown; Slack
# reads mrkdwn, so every answer goes through one renderer (``to_mrkdwn``).


def _renderer():
    from cognee.modules.integrations.slack import handle_cognee_ask

    return handle_cognee_ask


@pytest.mark.parametrize(
    "markdown, mrkdwn",
    [
        ("## Answer\n\nWe chose Neon.", "We chose Neon."),
        ("# Answer\nWe chose Neon.", "We chose Neon."),
        ("### Why Neon\nBranching.", "*Why Neon*\nBranching."),
        ("It is **fast** and __cheap__.", "It is *fast* and *cheap*."),
        ("It is *really* fast.", "It is _really_ fast."),
        ("~~Postgres~~ Neon", "~Postgres~ Neon"),
        ("- one\n- two\n* three", "• one\n• two\n• three"),
        ("See [the plan](https://acme.example/plan).", "See <https://acme.example/plan|the plan>."),
        ("See [the plan](http://acme.example/plan).", "See the plan."),
        ("See [x](https://a.example/?q=1|<!here>).", "See x."),
        ("Run `**not bold**` now.", "Run `**not bold**` now."),
        ("```\n# not a heading\n**kept**\n```", "```\n# not a heading\n**kept**\n```"),
        ("Tell <!channel> & <@U1>", "Tell &lt;!channel&gt; &amp; &lt;@U1&gt;"),
        ("[**bold** label](https://a.example/x)", "<https://a.example/x|*bold* label>"),
    ],
    ids=[
        "an Answer heading is dropped",
        "a level-one Answer heading too",
        "another heading becomes a bold line",
        "bold both ways",
        "italics",
        "strikethrough",
        "bullets",
        "an https link",
        "a link that is not https keeps its text",
        "a link that would break out of its brackets keeps its text",
        "inline code is kept as written",
        "a code block is kept as written",
        "Slack control sequences stay escaped (G48)",
        "a link label keeps its formatting",
    ],
)
def test_markdown_becomes_slack_mrkdwn(markdown, mrkdwn):
    assert _renderer().to_mrkdwn(markdown) == mrkdwn


@pytest.mark.parametrize(
    "markdown, plain",
    [
        ("## Answer\n\nWe chose **Neon** for *branching*.", "We chose Neon for branching."),
        ("- one\n- two", "• one\n• two"),
        ("See [the plan](https://acme.example/plan).", "See the plan."),
        ("Tell <!channel>", "Tell &lt;!channel&gt;"),
    ],
    ids=["markers go", "bullets stay readable", "a link keeps its text", "still escaped"],
)
def test_the_plain_text_of_an_answer_carries_no_markdown(markdown, plain):
    """The ``text`` of a message is what a notification or a screen reader shows."""
    assert _renderer().to_plain(markdown) == plain


def test_an_answer_has_no_header_and_its_fallback_is_the_plain_answer():
    """The answer is the thread reply under the question, so it needs no "Answer" header;
    live, the posted answer showed our bold "Answer" and then the LLM's "## Answer"."""
    fallback_text, blocks = _format_answer([_answer(["## Answer\n\nWe chose **Neon**."])])

    assert [block["type"] for block in blocks] == ["section"]
    assert blocks[0]["text"]["text"] == "We chose *Neon*."
    assert fallback_text == "We chose Neon."


def _cited_slack(document_id, day, channel="#general"):
    payload = _cited(document_id, f"Who owns the runbook? — ana, {day}", PERMALINK)
    payload["payload"]["document_parent"] = channel
    return payload


def test_sources_are_one_numbered_line_naming_channel_author_and_day():
    """Each source reads as where, who and when; the title's first words are not repeated.
    The year shows only when it is not this one."""
    from datetime import datetime, timezone

    this_year = datetime.now(timezone.utc).year
    _, blocks = _format_answer(
        [
            _answer(
                ["fact"],
                [
                    _cited_slack("d1", f"{this_year}-11-05"),
                    _cited_slack("d2", f"{this_year - 1}-03-09", channel="#leads-public"),
                    _cited("d3", "Quarterly plan.pdf", None),
                ],
            )
        ]
    )

    (sources,) = [block for block in blocks if block["type"] == "context"]
    (element,) = sources["elements"]
    text = element["text"]
    assert text.startswith("Sources · 3")
    assert f"1 <{PERMALINK}|#general · ana · Nov 5>" in text
    assert f"2 <{PERMALINK}|#leads-public · ana · Mar 9, {this_year - 1}>" in text
    assert "3 Quarterly plan.pdf" in text


# -- Rating an answer (SDK-1502): red until fixed -----------------------------------------


def _workspace_credential():
    from types import SimpleNamespace
    from uuid import uuid4

    return SimpleNamespace(provider_account_id="T0RATE", user_id=uuid4(), provider_metadata={})


async def _stored_score(credential, member, qa_id):
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    entries = await get_session_manager().get_session_entries_by_ids(
        user_id=str(credential.user_id),
        qa_ids=[qa_id],
        session_id=_renderer().answer_session_id("T0RATE", member),
    )
    return entries[0].feedback_score if entries else "missing"


@pytest.mark.asyncio
async def test_the_asker_rates_an_answer_and_a_second_click_changes_the_rating():
    """👍 is 5 and 👎 is 1 on the answer's own session Q&A, kept under the workspace's
    owner with the graph elements it used, so improve's feedback_weights stage can apply it.
    A second click re-rates; it does not add a vote."""
    from unittest.mock import AsyncMock, patch

    ask = _renderer()
    credential = _workspace_credential()
    qa_id = await ask.record_answer(
        credential,
        "U0ASKER",
        question="why Neon?",
        answer="We chose Neon.",
        used_graph_element_ids={"node_ids": ["n1"], "edge_ids": ["e1"]},
    )
    assert qa_id is not None, "the session cache is on in the unit tests"

    with patch.object(ask, "apply_answer_feedback", AsyncMock()) as apply:
        assert await ask.rate_answer(credential, "U0ASKER", qa_id, 5) is True
        assert await _stored_score(credential, "U0ASKER", qa_id) == 5
        assert await ask.rate_answer(credential, "U0ASKER", qa_id, 1) is True
        assert await _stored_score(credential, "U0ASKER", qa_id) == 1

    assert apply.await_count == 2


@pytest.mark.asyncio
async def test_someone_else_cannot_rate_an_answer_they_did_not_ask_for():
    """The button value is the answer's random id alone; the rater is the signed payload's
    user. Another member's click finds no answer in their own session and changes nothing."""
    from unittest.mock import AsyncMock, patch

    ask = _renderer()
    credential = _workspace_credential()
    qa_id = await ask.record_answer(
        credential,
        "U0ASKER",
        question="why Neon?",
        answer="We chose Neon.",
        used_graph_element_ids={"node_ids": ["n1"], "edge_ids": []},
    )

    with patch.object(ask, "apply_answer_feedback", AsyncMock()) as apply:
        assert await ask.rate_answer(credential, "U0OTHER", qa_id, 1) is False

    assert await _stored_score(credential, "U0ASKER", qa_id) is None
    apply.assert_not_awaited()
