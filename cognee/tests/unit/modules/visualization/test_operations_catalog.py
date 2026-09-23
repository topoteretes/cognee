"""The catalog's improve rows: descriptors keyed by stage name, pinned to the registry.

The registry describes WHAT runs; the catalog describes how each stage is
shown. The key-set equality below is what keeps the two from drifting now
that the stage classes carry no presentation fields.
"""

from cognee.modules.improve import DEFAULT_STAGES, stage_names
from cognee.modules.visualization.operations_catalog import (
    _IMPROVE_STAGE_DESCRIPTORS,
    get_operations_catalog,
    iter_improve_operations,
)


def test_every_registry_stage_has_exactly_one_descriptor():
    assert set(_IMPROVE_STAGE_DESCRIPTORS) == set(stage_names(DEFAULT_STAGES))


def test_improve_rows_follow_registry_order_and_shape():
    rows = list(iter_improve_operations())

    assert [row["name"] for row in rows] == stage_names(DEFAULT_STAGES)
    for row in rows:
        assert row["kind"] == "self_improve"
        assert row["scope"] in ("whole", "subset")
        assert row["label"] and row["summary"]
        assert isinstance(row["effects"], list)


def test_scope_comes_from_needs_sessions():
    scope_by_name = {row["name"]: row["scope"] for row in iter_improve_operations()}
    for stage in DEFAULT_STAGES:
        assert scope_by_name[stage.name] == ("subset" if stage.needs_sessions else "whole")


def test_node_sets_are_derived_from_produces_effects():
    rows = {row["name"]: row for row in iter_improve_operations()}
    assert rows["persist_session_qa"]["node_sets"] == ["user_sessions_from_cache"]
    assert rows["triplet_enrichment"]["node_sets"] == []


def test_catalog_appends_improve_rows_after_the_curated_ones():
    catalog = get_operations_catalog()
    names = [op["name"] for op in catalog]
    for stage_name in stage_names(DEFAULT_STAGES):
        assert stage_name in names
