from uuid import uuid4

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.utils.temporal_hints import (
    _looks_like_date_reference,
    hint_lines,
    normalize_absolute_date,
    temporal_hints_for_chunk,
)


@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("23 March 1947", "1947-03-23"),
        ("March 1947", "1947-03"),
        ("April 27, 1791", "1791-04-27"),
        ("05:32 on 27 April 1986", "1986-04-27 05:32:00"),
        ("the 1950s", None),
        ("spring of 1943", None),
        # day-only spans must not become a two-digit year (July 24 -> 2024-07)
        ("July 24", None),
        ("24 July", None),
        ("July 13", None),
        ("May 5", None),
        ("four weeks later", None),
        ("that spring", None),
    ],
)
def test_normalize_absolute_date(name, normalized):
    assert normalize_absolute_date(name) == normalized


def test_hint_lines_rolls_base_and_marks_inferred():
    lines, base = hint_lines(
        "On 26 April 1986 the reactor exploded. The following night of 27 April, engineers worked.",
        None,
    )
    assert (base.year, base.month, base.day) == (1986, 4, 26)
    assert len(lines) == 1
    assert "1986-04-27" in lines[0]
    assert "inferred from context" in lines[0]


def test_hint_lines_without_stated_year_yields_nothing():
    lines, base = hint_lines("The following night of 27 April, engineers worked.", None)
    assert lines == []
    assert base is None


@pytest.mark.parametrize(
    ("span", "hinted"),
    [
        ("at 4:00", True),
        ("05:32 UTC) on April 27", True),
        ("of 27 April", True),
        ("four weeks later", True),
        ("that spring", True),
        ("1:2 at the", False),
        ("9-1", False),
        ("6th", False),
        ("of The year", False),
        ("3-2 and Second", False),
        ("1920s and", False),
        # the month pattern must not fire on "decade" or on the verb "may"
        ("before this decade is out", False),
        ("they may well ask", False),
        ("in this decade and", False),
        ("in May", True),
        ("in December", True),
        ("on Sept 12", True),
        ("a four-year", False),
        ("ten years", False),
        ("two minutes", False),
        ("90 seconds and", False),
    ],
)
def test_date_reference_gate(span, hinted):
    assert _looks_like_date_reference(span) is hinted


def test_hint_lines_ignores_scores_and_durations():
    lines, base = hint_lines(
        "On 26 April 1986 they won 9-1. The broadcast lasted two minutes.", None
    )
    assert lines == []
    assert (base.year, base.month, base.day) == (1986, 4, 26)


def _chunk(document, text):
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
    )


def _document():
    return TextDocument(
        id=uuid4(),
        name="doc.txt",
        raw_data_location="doc.txt",
        external_metadata=None,
        mime_type="text/plain",
    )


def test_base_rolls_across_a_documents_chunks_but_not_across_documents():
    document = _document()
    first = _chunk(document, "On 26 April 1986 the reactor exploded.")
    second = _chunk(document, "The following night of 27 April, engineers worked.")
    other = _chunk(_document(), "The following night of 27 April, engineers worked.")

    assert temporal_hints_for_chunk(first) == []
    assert document._temporal_hint_base.year == 1986
    second_hints = temporal_hints_for_chunk(second)
    assert len(second_hints) == 1 and "1986-04-27" in second_hints[0]
    # Another document knows nothing about 1986.
    assert temporal_hints_for_chunk(other) == []
