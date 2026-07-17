"""Unit tests for the Block Kit citation renderer (issue #3609, commit 5).

Pure rendering — no cognee, no Slack, no keys.
"""

from src.citations import (
    DEFAULT_MAX_SOURCES,
    _NO_SOURCES_NOTE,
    notification_text,
    render_answer,
)
from src.memory_adapter import Answer, Citation


def _cite(ts="1700000000.000100", *, permalink="https://slack.example/x", ok=True, **kw):
    base = dict(channel_id="C1", ts=ts, permalink=permalink, author="alice", snippet="snip", ok=ok)
    base.update(kw)
    return Citation(**base)


def _blocks_text(blocks):
    """Flatten all mrkdwn text in a Block Kit block list for assertions."""
    out = []
    for block in blocks:
        if "text" in block and isinstance(block["text"], dict):
            out.append(block["text"]["text"])
        for element in block.get("elements", []):
            out.append(element.get("text", ""))
    return "\n".join(out)


def test_section_block_carries_answer_text():
    blocks = render_answer(Answer(text="We shipped Friday.", citations=[]))
    assert blocks[0]["type"] == "section"
    assert blocks[0]["text"]["type"] == "mrkdwn"
    assert blocks[0]["text"]["text"] == "We shipped Friday."


def test_two_citations_render_two_ordered_permalink_sources():
    answer = Answer(
        text="Answer.",
        citations=[
            _cite(ts="1700000000.000100", permalink="https://slack.example/a", author="alice"),
            _cite(ts="1700000100.000100", permalink="https://slack.example/b", author="bob"),
        ],
    )
    blocks = render_answer(answer)
    text = _blocks_text(blocks)

    assert "https://slack.example/a" in text
    assert "https://slack.example/b" in text
    # Order preserved (relevance order from the adapter).
    assert text.index("https://slack.example/a") < text.index("https://slack.example/b")
    # Rendered as mrkdwn links.
    assert "<https://slack.example/a|" in text
    assert "#C1" in text and "alice" in text


def test_same_ts_citations_dedupe_to_one_source():
    answer = Answer(
        text="Answer.",
        citations=[
            _cite(ts="1700000000.000100", permalink="https://slack.example/a"),
            _cite(ts="1700000000.000100", permalink="https://slack.example/a"),
        ],
    )
    blocks = render_answer(answer)
    text = _blocks_text(blocks)
    assert text.count("https://slack.example/a") == 1


def test_missing_permalink_renders_plain_text_no_broken_link():
    answer = Answer(
        text="Answer.",
        citations=[_cite(permalink="", ok=False, author="alice")],
    )
    blocks = render_answer(answer)
    text = _blocks_text(blocks)

    assert "<|" not in text  # never a broken link
    assert "alice" in text  # descriptive label still shown
    assert "#C1" in text


def test_missing_permalink_and_empty_label_falls_back_to_snippet():
    # The adapter's missing-index-row fallback: blank channel/ts/author.
    answer = Answer(
        text="Answer.",
        citations=[
            Citation(channel_id="", ts="", permalink="", author="", snippet="orphan text", ok=False)
        ],
    )
    text = _blocks_text(render_answer(answer))
    assert "orphan text" in text
    assert "<|" not in text


def test_zero_citations_render_no_sources_note():
    blocks = render_answer(Answer(text="Answer with no sources.", citations=[]))
    text = _blocks_text(blocks)
    assert "Answer with no sources." in text
    assert _NO_SOURCES_NOTE in text
    # No empty Sources block — the second block is the note, not a blank list.
    assert "*Sources:*" not in text


def test_blank_answer_uses_calm_fallback():
    blocks = render_answer(Answer(text="   ", citations=[]))
    section = blocks[0]["text"]["text"]
    assert section  # non-empty
    assert "couldn't find" in section.lower()


def test_source_cap_notes_plus_n_more():
    citations = [
        _cite(ts=f"170000000{i}.000100", permalink=f"https://slack.example/{i}")
        for i in range(DEFAULT_MAX_SOURCES + 3)
    ]
    blocks = render_answer(Answer(text="A.", citations=citations))
    text = _blocks_text(blocks)
    # Only DEFAULT_MAX_SOURCES links shown, with a "+3 more" note.
    assert text.count("https://slack.example/") == DEFAULT_MAX_SOURCES
    assert "+3 more" in text


def test_notification_text_falls_back_when_blank():
    assert notification_text(Answer(text="hi", citations=[])) == "hi"
    assert "couldn't find" in notification_text(Answer(text="", citations=[])).lower()


def test_block_kit_structure_is_section_then_context():
    blocks = render_answer(Answer(text="A.", citations=[_cite()]))
    assert [b["type"] for b in blocks] == ["section", "context"]
    assert blocks[1]["elements"][0]["type"] == "mrkdwn"
