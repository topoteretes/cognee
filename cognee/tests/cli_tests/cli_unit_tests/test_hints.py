"""Unit tests for CLI next-step hint helpers.

The hints module renders the "Next: ..." line after each primary
command. These tests pin behaviour that scripting users depend on:
``--quiet`` fully silences the hint, and the recall hint is only fired
when the search returned zero results.
"""

from argparse import Namespace

import pytest

from cognee.cli import hints


@pytest.fixture(autouse=True)
def _capture_echo(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(hints.fmt, "echo", lambda msg: lines.append(msg))
    return lines


def _args(**kwargs) -> Namespace:
    kwargs.setdefault("quiet", False)
    return Namespace(**kwargs)


def test_remember_hint_prints_dataset(_capture_echo):
    hints.remember_hint(_args(), "docs")
    assert _capture_echo == ['Next: cognee-cli recall "your question" -d docs']


def test_cognify_hint_mirrors_remember(_capture_echo):
    hints.cognify_hint(_args(), "docs")
    assert _capture_echo == ['Next: cognee-cli recall "your question" -d docs']


def test_recall_hint_skipped_on_hit(_capture_echo):
    hints.recall_hint(_args(), "docs", had_results=True)
    assert _capture_echo == []


def test_recall_hint_fires_on_miss(_capture_echo):
    hints.recall_hint(_args(), "docs", had_results=False)
    assert len(_capture_echo) == 1
    assert 'no matches in "docs"' in _capture_echo[0]


def test_forget_hint_prints_restart_line(_capture_echo):
    hints.forget_hint(_args(), "docs")
    assert len(_capture_echo) == 1
    assert "cognee-cli remember" in _capture_echo[0]


@pytest.mark.parametrize(
    "fn,extra",
    [
        (hints.remember_hint, {}),
        (hints.cognify_hint, {}),
        (hints.recall_hint, {"had_results": False}),
        (hints.forget_hint, {}),
    ],
)
def test_quiet_flag_suppresses_all_hints(fn, extra, _capture_echo):
    fn(_args(quiet=True), "docs", **extra)
    assert _capture_echo == []
