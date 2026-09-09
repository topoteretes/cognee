"""Unit tests for sanitize_value dict key collision handling (issue #3880)."""

from __future__ import annotations

import logging

from cognee.modules.agent_memory.sanitization import sanitize_value


class TestSanitizeValueDictKeyCollision:
    """Tests that sanitize_value does not silently drop dict entries when keys
    collide after str() conversion."""

    def test_no_collision_normal_dict(self):
        """A normal dict with string keys passes through unchanged (after truncation)."""
        result = sanitize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_int_and_string_key_collision(self):
        """Dict with {1: 'first', '1': 'second'} should preserve both entries,
        not silently overwrite the first."""
        result = sanitize_value({1: "first", "1": "second"})
        # Both values must be present — no silent data loss.
        assert "first" in result.values()
        assert "second" in result.values()
        # The int key 1 is processed first (insertion order), so it gets
        # the "1" slot. The string key "1" collides and is disambiguated.
        assert result["1"] == "first"
        assert result["1_2"] == "second"

    def test_multiple_int_string_collisions(self):
        """Multiple colliding keys should each get unique disambiguated names."""
        # Note: in Python, 1 and 1.0 are equal as dict keys, so we use
        # a tuple and its string form to create a real collision.
        result = sanitize_value({(1, 2): "a", "(1, 2)": "b", 3: "c"})
        # All three values must be present — no silent data loss.
        assert "a" in result.values()
        assert "b" in result.values()
        assert "c" in result.values()

    def test_collision_logs_warning(self, caplog):
        """A key collision should emit a warning, not fail silently."""
        with caplog.at_level(logging.WARNING, logger="cognee.modules.agent_memory.sanitization"):
            sanitize_value({1: "first", "1": "second"})
        assert any("collision" in record.message for record in caplog.records)

    def test_no_warning_when_no_collision(self, caplog):
        """No warning should be emitted when there are no collisions."""
        with caplog.at_level(logging.WARNING, logger="cognee.modules.agent_memory.sanitization"):
            sanitize_value({"a": 1, "b": 2, 3: "three"})
        assert not caplog.records

    def test_nested_dict_with_collision(self):
        """Nested dicts with key collisions should also be handled."""
        result = sanitize_value({"outer": {1: "x", "1": "y"}})
        inner = result["outer"]
        assert "x" in inner.values()
        assert "y" in inner.values()

    def test_tuple_key_collision(self):
        """Tuple keys that collide after str() should be disambiguated."""
        result = sanitize_value({(1, 2): "first", "(1, 2)": "second"})
        assert "first" in result.values()
        assert "second" in result.values()
