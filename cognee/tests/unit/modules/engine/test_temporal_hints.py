from uuid import uuid4

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.utils import temporal_hints
from cognee.modules.engine.utils.temporal_hints import (
    DateBase,
    attach_temporal_hints,
    chunk_temporal_hints,
    document_temporal_hints,
    hint_lines,
    normalize_absolute_date,
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
    # the hinted day advances the base too, so a later time can land on it
    assert (base.year, base.month, base.day) == (1986, 4, 27)
    assert lines == ['- "27 April" -> 1986-04-27 (inferred from context)']


def test_hint_lines_without_stated_year_yields_nothing():
    lines, base = hint_lines("The following night of 27 April, engineers worked.", None)
    assert lines == []
    assert base is None


# --- what a stated date is -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("On 26 April 1986 the reactor exploded.", DateBase(1986, 4, 26)),
        ("It was in July, 1805, and the speaker was Anna.", DateBase(1805, 7, None)),
        ("In 1805 Prince Andrew left.", DateBase(1805, None, None)),
        ("Released 2021-03-04, the patch fixed it.", DateBase(2021, 3, 4)),
        ("On October 11, 1806, one of the regiments halted.", DateBase(1806, 10, 11)),
        ("On the twenty-eighth of October, 1805, Kutuzov crossed.", DateBase(1805, 10, 28)),
        ("Born March of 1947 in Lyon.", DateBase(1947, 3, None)),
        (
            "The winter of 1812 was hard; between 1805 and 1807 he served.",
            DateBase(1805, None, None),
        ),
        # a stated date with an impossible day keeps only its month
        ("The 31 April 1986 report was wrong.", DateBase(1986, 4, None)),
    ],
)
def test_stated_dates_advance_the_base_and_are_not_hinted(text, expected):
    lines, base = hint_lines(text, None)
    assert lines == []
    assert base == expected


@pytest.mark.parametrize(
    "text",
    [
        "The budget was $1986 and the score 9-1.",
        "They counted 1,986 crates and 1986% growth.",
        "The 1920s and the 1950s were loud.",
        # a four-digit quantity is not a year without a dating word before it
        "The reactor was restored to 1000 MW and the ridge rose 2300 feet.",
        "The 1812 campaign is described in the second volume.",
    ],
)
def test_numbers_that_are_not_years_do_not_become_the_base(text):
    assert hint_lines(text, None) == ([], None)


def test_a_year_written_right_after_the_date_belongs_to_it():
    """ "On October 11, 1806" is one stated date — not a year-less "October 11"
    resolved against the previous year plus a bare "1806". The old split let a
    hint contradict the year written a few characters later."""
    lines, base = hint_lines(
        "In October, 1805, the army occupied Braunau. On October 11, 1806, a regiment halted.",
        None,
    )
    assert lines == []
    assert base == DateBase(1806, 10, 11)


# --- what gets hinted ------------------------------------------------------------

BASE_1805 = DateBase(1805, 7, None)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("On October 11, the day when all was astir, he came.", "1805-10-11"),
        ("He returned on 27 April with the crates.", "1805-04-27"),
        ("The Guards left on the tenth of August, and her son stayed.", "1805-08-10"),
        ("On the twenty-eighth of October Kutuzov crossed.", "1805-10-28"),
        ("Our dispatch of November 18 was not received.", "1805-11-18"),
        ("By early May the roads were dry.", "1805-05"),
        ("She wrote in December about the frost.", "1805-12"),
        ("Deliveries from Sept. 3rd onward were late.", "1805-09-03"),
    ],
)
def test_year_less_calendar_dates_inherit_the_base_year_at_their_own_precision(text, value):
    lines, _ = hint_lines(text, BASE_1805)
    assert len(lines) == 1, lines
    assert lines[0].endswith(f"-> {value} (inferred from context)")


@pytest.mark.parametrize(
    "text",
    [
        # words that look like months are not dates without a dating word
        "I will come to supper with you. May I?",
        "May God support you, Prince.",
        "March on, destroy the Russian army.",
        "His Most August Majesty the Emperor was not pleased.",
        # relative expressions are never resolved: the base has no hour, and a
        # document does not promise narrative order
        "An hour later Tikhon came. Six weeks later he was married.",
        "Five minutes later he returned. He had seen her a year ago.",
        "Later that spring the ice broke. The following night they left.",
        # scores, durations, decades, verses, impossible days
        "They won 9-1 and the broadcast lasted two minutes.",
        "John 3:16 was read aloud at the service.",
        "It was 31 April, which cannot be.",
    ],
)
def test_nothing_else_is_hinted(text):
    lines, base = hint_lines(text, BASE_1805)
    assert lines == []
    assert base == BASE_1805


def test_hint_lines_ignores_scores_and_durations():
    lines, base = hint_lines(
        "On 26 April 1986 they won 9-1. The broadcast lasted two minutes.", None
    )
    assert lines == []
    assert (base.year, base.month, base.day) == (1986, 4, 26)


# --- clock times -------------------------------------------------------------------


def test_a_time_is_placed_on_the_base_day_and_only_then():
    lines, base = hint_lines(
        "On 26 April 1986 the reactor exploded. At 01:23 the test began; "
        "by 5:32 a.m. the roof was gone. The 20:17 UTC call came later.",
        None,
    )
    assert lines == [
        '- "01:23" -> 1986-04-26 01:23:00 (inferred from context)',
        '- "5:32 a.m." -> 1986-04-26 05:32:00 (inferred from context)',
        '- "20:17 UTC" -> 1986-04-26 20:17:00 (inferred from context)',
    ]
    assert base == DateBase(1986, 4, 26)


def test_a_time_follows_a_hinted_day_inside_the_chunk():
    lines, _ = hint_lines("On 27 April the fire was out at 9:15 pm.", DateBase(1986, 4, 26))
    assert lines == [
        '- "27 April" -> 1986-04-27 (inferred from context)',
        '- "9:15 pm" -> 1986-04-27 21:15:00 (inferred from context)',
    ]


def test_a_time_belongs_to_the_date_in_its_own_sentence_first():
    """ "At 12:30 on May 20" is 12:30 on May 20, although a different day was
    the base when the time was read."""
    lines, _ = hint_lines(
        "At 12:30 on May 20, the assembly departed. At 06:30 they headed out.",
        DateBase(1969, 2, 27),
    )
    assert lines == [
        '- "12:30" -> 1969-05-20 12:30:00 (inferred from context)',
        '- "May 20" -> 1969-05-20 (inferred from context)',
        '- "06:30" -> 1969-05-20 06:30:00 (inferred from context)',
    ]


def test_a_time_without_a_day_in_the_base_is_not_hinted():
    """The old implementation filled the missing day from the wall clock, so
    the same text hinted a different date every day of the month."""
    lines, base = hint_lines("In November, 1805, he left at 4:00.", None)
    assert lines == []
    assert base == DateBase(1805, 11, None)


# --- purity --------------------------------------------------------------------------


def test_hints_never_consult_a_date_parser_or_the_clock(monkeypatch):
    """Everything a hint says comes from the text and the base: the dateparser
    used by ``normalize_absolute_date`` is not on this path at all."""

    def refuse(*args, **kwargs):
        raise AssertionError("hint_lines must not parse dates with dateparser")

    monkeypatch.setattr(temporal_hints, "DateDataParser", refuse)
    lines, base = hint_lines(
        "In November, 1805, Prince Vasili left. An hour later Tikhon came. "
        "On November 18 the dispatch arrived at 6:00 pm.",
        None,
    )
    assert lines == [
        '- "November 18" -> 1805-11-18 (inferred from context)',
        '- "6:00 pm" -> 1805-11-18 18:00:00 (inferred from context)',
    ]
    assert base == DateBase(1805, 11, 18)


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


FIRST = "On 26 April 1986 the reactor exploded."
SECOND = "The following night of 27 April, engineers worked."


def test_document_hints_roll_the_base_in_order_and_are_a_pure_function():
    hints = document_temporal_hints([FIRST, SECOND])
    assert hints[0] == []
    assert len(hints[1]) == 1 and "1986-04-27" in hints[1][0]
    # same texts, same hints — nothing is remembered between calls
    assert document_temporal_hints([FIRST, SECOND]) == hints
    # the base never flows backwards or across documents
    assert document_temporal_hints([SECOND, FIRST]) == [[], []]
    assert document_temporal_hints([SECOND]) == [[]]


def test_attach_temporal_hints_stores_each_chunks_lines_on_it():
    document = _document()
    first, second = _chunk(document, FIRST), _chunk(document, SECOND)
    assert chunk_temporal_hints(first) is None  # no document pass has run yet

    attach_temporal_hints([first, second])

    assert chunk_temporal_hints(first) == []
    assert "1986-04-27" in chunk_temporal_hints(second)[0]
    # a prompt input, not a node property
    assert "_temporal_hints" not in second.model_dump()
