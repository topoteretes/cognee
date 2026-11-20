"""Recall items with no readable text are dropped, not stringified (#457).

``_result_text`` used to fall back to ``str(value)``, so an item carrying no
text-bearing key with a string value (an empty completion, ``text: None``, an
error payload) was injected into the agent's prompt as the whole response
envelope and counted as a hit. Only strings are memory now, the key order is
kept, and unconvertible objects no longer turn into their repr.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes.provider import (  # noqa: E402
    CogneeMemoryProvider,
    _coerce_result_dict,
    _result_text,
)

DEGENERATE = {
    "kind": "graph_completion",
    "search_type": "HYBRID_COMPLETION",
    "text": "",
    "score": None,
    "dataset_name": "agent_sessions",
    "metadata": {},
    "raw": {"value": ""},
    "structured": None,
    "source": "graph",
}


@pytest.mark.parametrize("text", ["", None, 0, False, [], {}, "   ", "\n\t"])
def test_a_falsy_or_blank_text_slot_is_not_memory(text):
    assert _result_text({"text": text, "source": "graph"}) == ""


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"score": 0.5, "metadata": {}},
        {"error": "Unauthorized", "status": 401},
        {"raw": {"value": "..."}, "text": ""},
        DEGENERATE,
    ],
)
def test_items_without_a_text_key_are_not_memory(item):
    assert _result_text(item) == ""


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"text": "", "answer": "real"}, "real"),
        ({"text": None, "content": "real"}, "real"),
        ({"content": "", "summary": "real"}, "real"),
        ({"answer": "first", "text": "second"}, "first"),
        ("a plain string result", "a plain string result"),
        ({"text": ["mem one", "mem two"]}, "mem one\nmem two"),
    ],
)
def test_the_key_order_is_kept_and_empty_slots_are_skipped(item, expected):
    assert _result_text(item) == expected


def test_an_unconvertible_object_does_not_become_its_repr():
    class Opaque:
        def model_dump(self):
            raise RuntimeError("no")

    assert _coerce_result_dict(Opaque()) == {}
    assert _result_text(Opaque()) == ""


def test_the_memory_lane_drops_degenerate_items():
    provider = CogneeMemoryProvider()
    real = {"text": "User's server runs on port 8011", "source": "graph"}
    assert provider._memory_lane_texts([DEGENERATE]) == []
    assert provider._memory_lane_texts([real]) == ["User's server runs on port 8011"]
    assert provider._memory_lane_texts([DEGENERATE, real]) == ["User's server runs on port 8011"]


def test_prefetch_lines_skip_items_without_text():
    provider = CogneeMemoryProvider()
    real = {"text": "a real fact", "source": "graph"}
    assert provider._format_recall_lines([DEGENERATE], limit=5) == []
    assert provider._format_recall_lines([DEGENERATE, real], limit=5) == ["- [graph] a real fact"]


def test_the_recall_tool_reports_a_miss_for_textless_results(monkeypatch):
    provider = CogneeMemoryProvider()

    class Backend:
        def overflow_hint(self):
            return ""

        def empty_recall_hint(self):
            return ""

    provider._backend = Backend()
    monkeypatch.setattr(provider, "_recall", lambda *a, **k: [DEGENERATE])
    body = json.loads(provider._handle_recall({"query": "anything"}))
    assert body == {"result": "No relevant Cognee memory found.", "count": 0}
