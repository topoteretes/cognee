"""Deterministic date hints for graph extraction, and absolute-date normalization.

Date expressions whose year is not stated in the text ("of 27 April", "later
that spring") are resolved against a rolling per-document base — the last
date with a stated year seen earlier in the document — and handed to the
graph prompt as normalization hints, so the LLM names Timestamp nodes with
absolute dates instead of guessing. The chunk text itself is never modified.

``normalize_absolute_date`` is the other half: it turns a date the LLM left
in prose ("23 March 1947") into the normalized shape ``timestamp_from_text``
accepts, at the precision the expression states.
"""

import re
from typing import Any

from dateparser.date import DateDataParser
from dateparser.search import search_dates

_STATED_YEAR = re.compile(r"\b\d{4}\b")
# Hint only spans anchored to a real date reference: a month name, a time of
# day (two-digit minutes, so the score "1:2" fails while "05:32" passes), or a
# relative expression with an anchor word. search_dates also matches scores,
# ordinals, durations, and decades; a dropped hint costs nothing (the raw text
# is still in the prompt) while a bogus hint is misinformation, so the gate
# trades recall for precision.
_DATE_ANCHOR = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b\d{1,2}:\d{2}\b"
    r"|\b(?:ago|later|earlier|next|last|following|previous)\b"
    r"|\bthat (?:spring|summer|autumn|fall|winter|year|month|week|day|night|morning|evening)\b",
    re.IGNORECASE,
)
_SETTINGS = {
    "RETURN_AS_TIMEZONE_AWARE": True,
    "RETURN_TIME_AS_PERIOD": True,
    "TIMEZONE": "UTC",
    "TO_TIMEZONE": "UTC",
}
# Absolute dates only: "four weeks later" must not resolve against today, and
# "March" alone must not inherit the current year.
_ABSOLUTE_SETTINGS = {
    "PARSERS": ["absolute-time"],
    "REQUIRE_PARTS": ["year"],
    "PREFER_DAY_OF_MONTH": "first",
    "RETURN_TIME_AS_PERIOD": True,
}
_FORMATS = {"time": "%Y-%m-%d %H:%M:%S", "day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}


def _looks_like_date_reference(text: str) -> bool:
    return _DATE_ANCHOR.search(text) is not None


def _format_date(value, period: str) -> str:
    return value.strftime(_FORMATS.get(period, "%Y-%m-%d"))


def normalize_absolute_date(text: str) -> str | None:
    """Normalize a date expression to the shapes ``timestamp_from_text`` accepts.

    "23 March 1947" becomes "1947-03-23" and "March 1947" becomes "1947-03":
    the output precision follows dateparser's own period detection, so a month
    name never fabricates a day. Relative or year-less expressions return None.
    """
    date_data = DateDataParser(languages=["en"], settings=_ABSOLUTE_SETTINGS).get_date_data(text)
    if date_data.date_obj is None:
        return None
    return _format_date(date_data.date_obj, date_data.period)


def hint_lines(text: str, base: Any) -> tuple[list[str], Any]:
    """Return (hint lines, updated base) for one chunk of text.

    Expressions with a stated year advance the base and need no hint; the rest
    pass the ``_DATE_ANCHOR`` gate and are re-parsed with ``RELATIVE_BASE`` set
    to the current base, so the hinted date is explicit about which parts came
    from context rather than the text.
    """
    matches = search_dates(text, languages=["en"], settings=_SETTINGS, add_detected_language=False)

    lines: list[str] = []
    for source_text, resolved in matches or []:
        span = source_text.strip()
        # "$1986" is money, and a trailing " a"/" p" is a truncated am/pm that
        # lost its meridian — drop both before they can touch the base.
        if span.startswith("$") or span.endswith((" a", " p")):
            continue
        if _STATED_YEAR.search(source_text):
            base = resolved
            continue
        if base is None or not _looks_like_date_reference(source_text):
            continue
        parser = DateDataParser(languages=["en"], settings={**_SETTINGS, "RELATIVE_BASE": base})
        date_data = parser.get_date_data(source_text)
        if date_data.date_obj is None:
            continue
        line = (
            f'- "{source_text}" -> {_format_date(date_data.date_obj, date_data.period)}'
            " (inferred from context)"
        )
        if line not in lines:
            lines.append(line)
    return lines, base


def temporal_hints_for_chunk(chunk) -> list[str]:
    """Hint lines for one chunk, advancing its document's rolling base.

    Call in document order: the base lives on ``chunk.is_part_of`` so it
    carries across the batches a document's chunks are extracted in.
    """
    document = chunk.is_part_of
    lines, base = hint_lines(chunk.text, document._temporal_hint_base)
    document._temporal_hint_base = base
    return lines
