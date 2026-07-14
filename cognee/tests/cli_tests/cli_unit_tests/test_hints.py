"""Unit tests for CLI next-step hint helpers.

The hints module renders the "Next: ..." line after each primary command.
These tests pin the exact copy so a reword is a deliberate change, and that
the empty-recall hint points the user back at ``remember``.
"""

import pytest

from cognee.cli import hints


@pytest.fixture(autouse=True)
def _capture_echo(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(hints.fmt, "echo", lambda msg: lines.append(msg))
    return lines


def test_hint_recall_uses_dataset(_capture_echo):
    hints.hint_recall("docs")
    assert _capture_echo == ['Next: cognee-cli recall "your question" -d docs']


def test_hint_recall_empty_points_at_remember(_capture_echo):
    hints.hint_recall_empty("docs")
    assert len(_capture_echo) == 1
    assert 'no matches in "docs"' in _capture_echo[0]
    assert "cognee-cli remember" in _capture_echo[0]


def test_hint_remember_after_forget(_capture_echo):
    hints.hint_remember("docs")
    assert len(_capture_echo) == 1
    assert "cognee-cli remember" in _capture_echo[0]
