"""Tests for ``truncate_text``.

Guards the length invariant: ``truncate_text(value, limit)`` must never return
a string longer than ``limit`` characters. Historically a limit below 3 hit a
negative slice (``value[: limit - 3]`` slices from the end) and returned a
string *longer* than the requested limit.
"""

import pytest

from cognee.modules.agent_memory.sanitization import truncate_text


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 10, 100])
def test_truncate_text_never_exceeds_limit(limit):
    value = "a" * 500
    result = truncate_text(value, limit)
    assert len(result) <= limit


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (0, ""),
        (1, "a"),
        (2, "ab"),
        (3, "..."),
        (4, "a..."),
        (5, "ab..."),
        (6, "abc..."),
    ],
)
def test_truncate_text_small_limits(limit, expected):
    # Value is 1 char longer than the largest limit under test, so every case
    # exercises the truncation branch, not the early return.
    assert truncate_text("abcdefg", limit) == expected


def test_truncate_text_returns_value_unchanged_when_within_limit():
    value = "hello"
    assert truncate_text(value, 10) == value
    assert truncate_text(value, len(value)) == value


def test_truncate_text_appends_ellipsis_for_large_limits():
    value = "hello world"
    result = truncate_text(value, 8)
    assert result == "hello..."
    assert len(result) == 8


def test_truncate_text_empty_value():
    assert truncate_text("", 2) == ""
    assert truncate_text("", 0) == ""
