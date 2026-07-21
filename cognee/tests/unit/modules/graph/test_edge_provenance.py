"""Tests for provenance-by-default (issue #3632).

Two things are covered:

1. The ``provenance_enabled`` config flag — on by default, disable-able via the
   ``PROVENANCE_ENABLED`` environment variable.
2. Edge provenance stamping in ``ensure_default_edge_properties`` — edges inherit
   their source endpoint's source lineage so every edge is traceable to the same
   source as its nodes, including the disabled / unstamped case where no
   provenance should be copied.
"""

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import ensure_default_edge_properties

_PROVENANCE_KEYS = (
    "source_pipeline",
    "source_task",
    "source_node_set",
    "source_content_hash",
    "source_user",
    "source_dataset_id",
    "source_document_id",
    "source_chunk_id",
)


class NamedPoint(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


# ---------------------------------------------------------------------------
# provenance_enabled config flag
# ---------------------------------------------------------------------------


def test_provenance_enabled_defaults_to_true():
    from cognee.base_config import get_base_config

    get_base_config.cache_clear()
    assert get_base_config().provenance_enabled is True


def test_provenance_enabled_can_be_disabled_via_env(monkeypatch):
    from cognee.base_config import get_base_config

    monkeypatch.setenv("PROVENANCE_ENABLED", "false")
    get_base_config.cache_clear()
    try:
        assert get_base_config().provenance_enabled is False
    finally:
        # Restore the cached default for the rest of the suite.
        get_base_config.cache_clear()


# ---------------------------------------------------------------------------
# edge provenance stamping
# ---------------------------------------------------------------------------


def test_edge_inherits_source_node_provenance():
    source = NamedPoint(name="Alice")
    source.source_pipeline = "cognify_pipeline"
    source.source_task = "extract_graph"
    source.source_node_set = "my_dataset"
    source.source_content_hash = "abc123"
    source.source_user = "alice@example.com"
    source.source_dataset_id = "ds-1"
    source.source_document_id = "doc-1"
    source.source_chunk_id = "chunk-1"
    target = NamedPoint(name="Acme")

    edge = (str(source.id), str(target.id), "works_at", {})
    result = ensure_default_edge_properties([edge], nodes=[source, target])
    props = result[0][3]

    assert props["source_pipeline"] == "cognify_pipeline"
    assert props["source_task"] == "extract_graph"
    assert props["source_node_set"] == "my_dataset"
    assert props["source_content_hash"] == "abc123"
    assert props["source_user"] == "alice@example.com"
    assert props["source_dataset_id"] == "ds-1"
    assert props["source_document_id"] == "doc-1"
    assert props["source_chunk_id"] == "chunk-1"


def test_edge_provenance_absent_when_source_node_unstamped():
    # Mirrors provenance disabled: nodes carry no lineage, so edges get none.
    source = NamedPoint(name="Alice")
    target = NamedPoint(name="Acme")

    edge = (str(source.id), str(target.id), "works_at", {})
    result = ensure_default_edge_properties([edge], nodes=[source, target])
    props = result[0][3]

    for field in _PROVENANCE_KEYS:
        assert field not in props


def test_explicit_edge_provenance_is_not_overwritten():
    source = NamedPoint(name="Alice")
    source.source_pipeline = "cognify_pipeline"
    target = NamedPoint(name="Acme")

    edge = (str(source.id), str(target.id), "works_at", {"source_pipeline": "explicit"})
    result = ensure_default_edge_properties([edge], nodes=[source, target])

    assert result[0][3]["source_pipeline"] == "explicit"


def test_edge_provenance_noop_without_nodes():
    # No node lookup available -> nothing to inherit, and no crash.
    edge = ("s", "t", "rel", {})
    result = ensure_default_edge_properties([edge])
    props = result[0][3]

    for field in _PROVENANCE_KEYS:
        assert field not in props
