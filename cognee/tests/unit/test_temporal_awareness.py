"""Unit tests for temporal awareness (issue #3700).

Stubs the mistralai/instructor/litellm chain that is broken in this local
environment so the tests run on any Python 3.10+ install with pydantic.
"""

import importlib.util
import logging
import sys
import time
import types
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Stub the broken transitive dep before any cognee import.
# instructor uses importlib.util.find_spec("mistralai") which requires a
# proper __spec__ — a plain ModuleType stub raises ValueError.
# ---------------------------------------------------------------------------
if "mistralai" not in sys.modules:
    _spec = importlib.util.spec_from_loader("mistralai", loader=None)
    _m = types.ModuleType("mistralai")
    _m.__spec__ = _spec
    _m.Mistral = object
    sys.modules["mistralai"] = _m

# ---------------------------------------------------------------------------
# Now safe to import cognee modules
# ---------------------------------------------------------------------------

from cognee.infrastructure.engine.models.DataPoint import DataPoint  # noqa: E402
from cognee.modules.graph.utils.deduplicate_nodes_and_edges import (  # noqa: E402
    _content_differs,
    deduplicate_nodes_and_edges,
)
from cognee.tasks.storage.close_node import is_valid  # noqa: E402


# ---------------------------------------------------------------------------
# DataPoint.valid_to field
# ---------------------------------------------------------------------------

def test_valid_to_defaults_to_none():
    dp = DataPoint(id=uuid4())
    assert dp.valid_to is None


def test_valid_to_can_be_set():
    ts = int(time.time() * 1000)
    dp = DataPoint(id=uuid4(), valid_to=ts)
    assert dp.valid_to == ts


def test_valid_to_survives_json_round_trip():
    ts = int(time.time() * 1000)
    dp = DataPoint(id=uuid4(), valid_to=ts)
    restored = DataPoint.from_json(dp.to_json())
    assert restored.valid_to == ts


# ---------------------------------------------------------------------------
# is_valid()
# ---------------------------------------------------------------------------

def test_is_valid_when_valid_to_is_none():
    dp = DataPoint(id=uuid4())
    assert is_valid(dp)


def test_is_not_valid_when_valid_to_in_past():
    past_ms = int(time.time() * 1000) - 60_000
    dp = DataPoint(id=uuid4(), valid_to=past_ms)
    assert not is_valid(dp)


def test_is_valid_when_valid_to_in_future():
    future_ms = int(time.time() * 1000) + 3_600_000
    dp = DataPoint(id=uuid4(), valid_to=future_ms)
    assert is_valid(dp)


def test_is_valid_with_dict_node():
    assert is_valid({"valid_to": None})
    assert not is_valid({"valid_to": int(time.time() * 1000) - 1000})


def test_is_valid_at_explicit_timestamp():
    ts = 1_000_000_000_000
    dp = DataPoint(id=uuid4(), valid_to=ts + 1)
    assert is_valid(dp, at_ms=ts)
    assert not is_valid(dp, at_ms=ts + 2)


# ---------------------------------------------------------------------------
# Contradiction detection in deduplicate_nodes_and_edges
# ---------------------------------------------------------------------------

class _Fact(DataPoint):
    name: str
    value: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


def test_dedup_no_warning_for_identical_nodes(caplog):
    node = _Fact(name="fact1", value="v1")
    with caplog.at_level(logging.WARNING):
        nodes, _ = deduplicate_nodes_and_edges([node, node], [])
    assert len(nodes) == 1
    assert "Contradiction" not in caplog.text


def test_dedup_warns_on_contradiction(caplog):
    node_a = _Fact(name="fact1", value="v1")
    node_b = _Fact(name="fact1", value="v2")
    with caplog.at_level(logging.WARNING):
        nodes, _ = deduplicate_nodes_and_edges([node_a, node_b], [])
    assert "Contradiction" in caplog.text


def test_dedup_keeps_first_on_contradiction():
    node_a = _Fact(name="fact1", value="first")
    node_b = _Fact(name="fact1", value="second")
    nodes, _ = deduplicate_nodes_and_edges([node_a, node_b], [])
    assert nodes[0].value == "first"


def test_content_differs_ignores_timestamps():
    dp1 = DataPoint(id=uuid4())
    time.sleep(0.01)
    dp2 = DataPoint(id=dp1.id)
    assert not _content_differs(dp1, dp2)


def test_content_differs_detects_field_change():
    node_a = _Fact(name="x", value="old")
    node_b = _Fact(name="x", value="new")
    assert _content_differs(node_a, node_b)


def test_dedup_normal_dedup_still_works():
    a = DataPoint(id=uuid4())
    b = DataPoint(id=uuid4())
    nodes, _ = deduplicate_nodes_and_edges([a, b, a], [])
    assert len(nodes) == 2
