"""Tests for truncate_text length-bounding contract.

Regression: for ``limit < 3`` the implementation used ``value[: limit - 3]``,
a negative slice that returned a string *longer* than ``limit`` (e.g.
``truncate_text("hello world", 2)`` returned 13 characters). truncate_text must
never return more than ``limit`` characters.
"""

import pytest

from cognee.modules.agent_memory.sanitization import truncate_text


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 5, 10, 50, 1000])
def test_never_exceeds_limit(limit):
    value = "hello world, this is a reasonably long string to truncate"
    assert len(truncate_text(value, limit)) <= limit


def test_short_value_returned_unchanged():
    assert truncate_text("hi", 100) == "hi"


def test_long_value_uses_ellipsis_and_hits_limit_exactly():
    # For limit >= 3, truncation appends "..." and fills the limit exactly.
    result = truncate_text("x" * 100, 10)
    assert result == "x" * 7 + "..."
    assert len(result) == 10


@pytest.mark.parametrize(
    "value,limit,expected",
    [
        ("hello world", 0, ""),
        ("hello world", 1, "h"),
        ("hello world", 2, "he"),
        ("hello world", 3, "..."),
        ("hello world", 4, "h..."),
    ],
)
def test_small_limits(value, limit, expected):
    assert truncate_text(value, limit) == expected
