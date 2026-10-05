"""Deterministic date hints for graph extraction, and absolute-date normalization.

Calendar expressions whose year is not stated in the text ("of 27 April",
"in early May", "at 05:32") are resolved against a rolling per-document base
— the last date with a stated year seen earlier in the document — and handed
to the graph prompt as normalization hints, so the LLM names Timestamp nodes
with absolute dates instead of guessing. The chunk text itself is never
modified.

The hints are a pure function of the text. Expressions are found by pattern,
not by a fuzzy date search, and a hint states exactly the parts the text
states plus what the base supplies: a day-and-month expression becomes a day
in the base's year, a month alone becomes a month, a clock time becomes a
time on the base's day — and only when the base has one. Relative
expressions ("an hour later", "that spring") are never hinted: resolving
them needs a precision the base does not have and a narrative order a
document does not promise. Nothing is ever filled in from the wall clock.

``normalize_absolute_date`` is the other half: it turns a date the LLM left
in prose ("23 March 1947") into the normalized shape ``timestamp_from_text``
accepts, at the precision the expression states.
"""

import bisect
import calendar
import re
from dataclasses import dataclass

from dateparser.date import DateDataParser

_STATED_YEAR = re.compile(r"\b\d{4}\b")

# Month names are matched case-sensitively and as whole words: lower-case
# "may" is usually the verb, and a prefix match on "dec" would fire on
# "decade". An abbreviation may carry its dot ("Sept.").
_MONTH = (
    r"(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_ORDINAL_WORDS = {
    word: number
    for number, word in enumerate(
        [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
            "twentieth",
            "twenty-first",
            "twenty-second",
            "twenty-third",
            "twenty-fourth",
            "twenty-fifth",
            "twenty-sixth",
            "twenty-seventh",
            "twenty-eighth",
            "twenty-ninth",
            "thirtieth",
            "thirty-first",
        ],
        start=1,
    )
}
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_DAY_WORD = rf"(?P<day_word>{'|'.join(_ORDINAL_WORDS)})"
_YEAR = r"(?P<year>\d{4})"
# A four-digit number on its own is a year only when a dating word introduces
# it ("in 1805", "the winter of 1812", "circa 1600"): "1000 MW", "2300 feet",
# "$1986" and "the 1812 campaign" are not, and a number that is merely
# mentioned ("a population of 1500") must not become the document's base.
_BARE_YEAR = re.compile(
    r"\b(?P<lead>[Ii]n|[Oo]f|[Bb]y|[Uu]ntil|[Tt]ill|[Ss]ince|[Dd]uring|[Ff]rom|[Tt]hrough"
    r"|[Cc]irca|[Aa]round|[Aa]bout|[Bb]efore|[Aa]fter|[Bb]etween|[Ee]arly|[Ll]ate|[Mm]id|year) "
    r"(?P<year>1\d{3}|20\d{2})\b(?![\d%])"
)
# A year that is a heading dates everything under it: alone on its line
# ("1936"), as "2005: Elections", "1936–1937", or behind heading markup
# ("## 1805"). A line that merely starts with a number ("1500 men marched")
# does not qualify — the year must be the whole heading or end at a colon
# or dash.
_HEADING_YEAR = re.compile(
    r"^[ \t#=*\-–—]*(?P<year>1\d{3}|20\d{2})(?=[ \t]*(?:[:\-–—]|$))", re.MULTILINE
)
_SENTENCE_END = re.compile(r"[.!?]+(?=\s)")

# One pattern per expression shape, in priority order: a span carrying a
# year wins over one carrying a day, which wins over a bare month, so
# "October 11, 1806" is one stated date and never a year-less "October 11"
# plus a bare "1806" — the split that let a hint contradict the year written
# a few characters later.
_CALENDAR_PATTERNS = (
    # 23 March 1947 · 27th of April · the twenty-eighth of October, 1805
    re.compile(rf"\b(?:the )?{_DAY}(?: of)? {_MONTH}(?:,? (?:of )?{_YEAR})?\b"),
    re.compile(rf"\b(?:the )?{_DAY_WORD} of {_MONTH}(?:,? (?:of )?{_YEAR})?\b"),
    # October 11 · April 27th, 1791
    re.compile(rf"\b{_MONTH} {_DAY}(?:,? {_YEAR})?\b"),
    # July, 1805 · March 1947 · March of 1947
    re.compile(rf"\b{_MONTH},? (?:of )?{_YEAR}\b"),
    # ISO: 1947-03-23 · 1947-03 (stated, so it only advances the base)
    re.compile(
        r"\b(?P<year>\d{4})-(?P<month_number>0[1-9]|1[0-2])(?:-(?P<day>0[1-9]|[12]\d|3[01]))?\b"
    ),
    # A month alone counts only after a word that dates something ("in May",
    # "by late August"); "May I?", "March on" and "Most August Majesty" have
    # none and are not dates.
    re.compile(
        rf"\b(?P<lead>(?i:in|of|by|until|till|since|during|from|through|early|late|mid|that)) "
        rf"{_MONTH}\b(?!\.? ?\d)"
    ),
)
# A clock time, marked as one by "at" before it or a meridian/zone after it:
# "at 4:00", "20:17 UTC", "05:32 a.m."; a verse ("John 3:16") or a score has
# neither and is skipped.
_TIME_PATTERN = re.compile(
    r"(?P<at>\b[Aa]t )?\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?"
    r"(?P<suffix> ?(?:[ap]m|[ap]\.m\.|UTC|GMT)(?!\w))?"
)
_ABSOLUTE_SETTINGS = {
    "PARSERS": ["absolute-time"],
    "REQUIRE_PARTS": ["year"],
    "PREFER_DAY_OF_MONTH": "first",
    "RETURN_TIME_AS_PERIOD": True,
}
_FORMATS = {"time": "%Y-%m-%d %H:%M:%S", "day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}


@dataclass(frozen=True)
class DateBase:
    """The last stated date in a document so far: a year, and the month and
    day when the text stated them. A hint never supplies a part the base
    does not have."""

    year: int
    month: int | None = None
    day: int | None = None


def _format_date(value, period: str) -> str:
    return value.strftime(_FORMATS.get(period, "%Y-%m-%d"))


def normalize_absolute_date(text: str) -> str | None:
    """Normalize a date expression to the shapes ``timestamp_from_text`` accepts.

    "23 March 1947" becomes "1947-03-23" and "March 1947" becomes "1947-03":
    the output precision follows dateparser's own period detection, so a month
    name never fabricates a day. Relative or year-less expressions return None.
    The year must be written as four digits: dateparser would otherwise read
    the day in "July 24" as the year 2024, and ``REQUIRE_PARTS`` does not
    catch that.
    """
    if not _STATED_YEAR.search(text):
        return None
    date_data = DateDataParser(languages=["en"], settings=_ABSOLUTE_SETTINGS).get_date_data(text)
    if date_data.date_obj is None:
        return None
    return _format_date(date_data.date_obj, date_data.period)


def _month_number(groups: dict) -> int | None:
    if groups.get("month_number"):
        return int(groups["month_number"])
    if groups.get("month"):
        return _MONTH_NUMBERS[groups["month"][:3].lower()]
    return None


def _day_number(groups: dict) -> int | None:
    if groups.get("day"):
        return int(groups["day"])
    if groups.get("day_word"):
        return _ORDINAL_WORDS[groups["day_word"]]
    return None


def _valid_day(year: int, month: int, day: int) -> bool:
    return 1 <= day <= calendar.monthrange(year, month)[1]


def _calendar_matches(text: str) -> list[tuple[int, int, dict]]:
    """Non-overlapping calendar spans in text order: ``(start, end, groups)``.

    Patterns are tried in priority order and a later pattern never claims
    text an earlier one already covers.
    """
    taken: list[tuple[int, int]] = []
    found = []
    for pattern in _CALENDAR_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            found.append((start, end, match.groupdict()))
    for pattern in (_BARE_YEAR, _HEADING_YEAR):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            found.append((start, end, {"year": match["year"]}))
    found.sort()
    return found


def _hint_line(span: str, value: str) -> str:
    return f'- "{span}" -> {value} (inferred from context)'


def _time_value(match: re.Match, base: DateBase) -> str | None:
    hour, minute = int(match["hour"]), int(match["minute"])
    second = int(match["second"] or 0)
    meridian = (match["suffix"] or "").strip().lower().replace(".", "")
    if meridian.startswith("p") and hour < 12:
        hour += 12
    elif meridian.startswith("a") and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return f"{base.year:04d}-{base.month:02d}-{base.day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _sentence_starts(text: str) -> list[int]:
    return [0] + [match.end() for match in _SENTENCE_END.finditer(text)]


def _sentence_of(starts: list[int], position: int) -> int:
    return bisect.bisect_right(starts, position) - 1


def hint_lines(text: str, base: DateBase | None) -> tuple[list[str], DateBase | None]:
    """Return (hint lines, updated base) for one chunk of text.

    A span that states its year advances the base and needs no hint. A
    day-and-month or month-only span inherits the base's year and is hinted
    at the precision it states; it also advances the base's month and day. A
    clock time is placed on the day-level date nearest to it in its own
    sentence ("At 12:30 on May 20"), else on the base's day as of its
    position — and only when that day exists. Without a base nothing is
    hinted. No part of a hint ever comes from the current date.
    """
    # Pass 1: calendar spans in text order, each recorded with the base as it
    # stands after it, so a time can look up the date of its own sentence
    # whether that date comes before or after it.
    _incoming_base = base
    resolved: list[tuple[int, DateBase | None]] = []
    hints: list[tuple[int, str]] = []
    for start, end, groups in _calendar_matches(text):
        month, day = _month_number(groups), _day_number(groups)
        if groups.get("year"):
            year = int(groups["year"])
            if month is not None and day is not None and not _valid_day(year, month, day):
                day = None
            base = DateBase(year, month, day)
        elif base is not None and month is not None:
            if day is None:
                value = f"{base.year:04d}-{month:02d}"
            elif _valid_day(base.year, month, day):
                value = f"{base.year:04d}-{month:02d}-{day:02d}"
            else:
                continue
            base = DateBase(base.year, month, day)
            hints.append((start, _hint_line(text[start:end].strip(), value)))
        resolved.append((start, base))

    # Pass 2: clock times.
    starts = _sentence_starts(text)
    for match in _TIME_PATTERN.finditer(text):
        if not (match["at"] or match["suffix"]):
            continue
        position = match.start("hour")
        sentence = _sentence_of(starts, position)
        in_sentence = [
            (abs(start - position), day_base)
            for start, day_base in resolved
            if day_base is not None and day_base.day is not None
            if _sentence_of(starts, start) == sentence
        ]
        if in_sentence:
            anchor = min(in_sentence)[1]
        else:
            before = [day_base for start, day_base in resolved if start < position]
            anchor = before[-1] if before else _incoming_base
        if anchor is None or anchor.month is None or anchor.day is None:
            continue
        value = _time_value(match, anchor)
        if value is not None:
            hints.append((position, _hint_line(text[position : match.end()].strip(), value)))

    lines: list[str] = []
    for _, line in sorted(hints):
        if line not in lines:
            lines.append(line)
    return lines, base


def document_temporal_hints(chunk_texts: list[str]) -> list[list[str]]:
    """Hint lines for every chunk of one document, in document order.

    A pure function of the texts: the rolling base (the last date with a
    stated year) is a local that advances from one chunk to the next, so the
    same texts always yield the same hints, however the chunks are later
    batched for extraction. Dates never carry across documents — call once
    per document.
    """
    base = None
    hints: list[list[str]] = []
    for text in chunk_texts:
        lines, base = hint_lines(text, base)
        hints.append(lines)
    return hints


def attach_temporal_hints(chunks) -> None:
    """Compute ``document_temporal_hints`` for ``chunks`` (one document, in order)
    and store each chunk's lines on it, where extraction reads them."""
    for chunk, lines in zip(chunks, document_temporal_hints([chunk.text for chunk in chunks])):
        chunk._temporal_hints = lines


def chunk_temporal_hints(chunk) -> list[str] | None:
    """The hints attached to ``chunk``, or None when no document-level pass ran."""
    return getattr(chunk, "_temporal_hints", None)
