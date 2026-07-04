import pytest

from cognee.modules.retrieval.utils.validate_queries import validate_queries


# ── single-query whitespace rejection ─────────────────────────────────────────


def test_empty_string_is_rejected():
    is_valid, msg = validate_queries("", None)
    assert is_valid is False
    assert "non-empty" in msg


def test_whitespace_only_spaces_is_rejected():
    is_valid, msg = validate_queries("   ", None)
    assert is_valid is False
    assert "non-empty" in msg


def test_tab_newline_whitespace_is_rejected():
    is_valid, msg = validate_queries("\t\n", None)
    assert is_valid is False
    assert "non-empty" in msg


# ── single-query valid cases ───────────────────────────────────────────────────


def test_plain_query_passes():
    is_valid, msg = validate_queries("cat", None)
    assert is_valid is True
    assert msg == ""


def test_query_with_surrounding_whitespace_passes():
    """Surrounding whitespace is fine — only stripped value must be non-empty."""
    is_valid, msg = validate_queries("  cat  ", None)
    assert is_valid is True
    assert msg == ""


# ── batch whitespace rejection ─────────────────────────────────────────────────


def test_batch_item_whitespace_only_is_rejected():
    is_valid, msg = validate_queries(None, ["  ", "x"])
    assert is_valid is False
    assert "non-empty" in msg


def test_batch_item_empty_string_is_rejected():
    is_valid, msg = validate_queries(None, ["a", ""])
    assert is_valid is False
    assert "non-empty" in msg


# ── batch valid cases ──────────────────────────────────────────────────────────


def test_valid_batch_passes():
    is_valid, msg = validate_queries(None, ["a", "b"])
    assert is_valid is True
    assert msg == ""


def test_batch_item_with_surrounding_whitespace_passes():
    """Items with surrounding whitespace are still valid."""
    is_valid, msg = validate_queries(None, ["  hello  ", "world"])
    assert is_valid is True
    assert msg == ""
