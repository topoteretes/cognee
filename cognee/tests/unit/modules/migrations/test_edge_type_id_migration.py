"""Tests for the EdgeType point-id migration (namespace_edge_type_point_ids).

Same philosophy as the entity-migration tests: frozen derivations pinned by
literal UUIDs, a drift alarm against the live model, and rekey behavior driven
through the REAL-contract fakes from test_graph_id_migration.
"""

import asyncio
from uuid import UUID

from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions.namespace_edge_type_point_ids import (
    _frozen_bare_id,
    _frozen_model_id,
    build_id_remap,
    build_id_remap_reverse,
    migrate,
    downgrade,
)
from cognee.tests.unit.modules.migrations.test_graph_id_migration import (
    _FakeLanceTable,
    _make_lance_adapter,
)


def test_frozen_id_derivations_are_pinned():
    """These literals ARE the migration. If this fails, the migration's
    meaning changed — write a NEW migration instead."""
    assert str(_frozen_bare_id("works_at")) == "c727a980-d048-5704-9f05-f5382146eaf7"
    assert str(_frozen_model_id("works_at")) == "732f3b53-b098-53cb-8871-283477dc2b43"
    # normalization: case, spaces, apostrophes
    assert _frozen_bare_id("Works At") == _frozen_bare_id("works_at")
    assert _frozen_model_id("O'Brien knows") == _frozen_model_id("obrien_knows")


def test_live_model_matches_frozen_new_scheme():
    """DRIFT ALARM: the live EdgeType identity derivation must equal this
    migration's frozen NEW scheme. If it diverges, the live scheme changed —
    that is a NEW migration, never an edit here."""
    from cognee.modules.graph.models.EdgeType import EdgeType

    for text in ("works_at", "Alice works at Acme.", "O'Brien Knows Bob"):
        assert EdgeType.id_for(text) == _frozen_model_id(text)
        # constructed instances derive the same id (identity_fields path)
        assert EdgeType(relationship_name=text, number_of_edges=1).id == _frozen_model_id(text)


def test_build_id_remap_prefers_edge_text_and_skips_self_edges():
    edges = [
        ("a", "b", "works_at", {"edge_text": "Alice works at Acme."}),
        ("b", "c", "knows", {}),  # no edge_text -> relationship name
        ("d", "d", "SELF", {}),  # Ladybug synthetic placeholder -> skipped
        ("a", "c", "works_at", {"edge_text": "Alice works at Acme."}),  # dup text -> one entry
    ]
    remap = build_id_remap(edges)
    assert remap == {
        str(_frozen_bare_id("Alice works at Acme.")): str(_frozen_model_id("Alice works at Acme.")),
        str(_frozen_bare_id("knows")): str(_frozen_model_id("knows")),
    }
    reverse = build_id_remap_reverse(edges)
    assert reverse == {new: old for old, new in remap.items()}


class _FakeGraph:
    def __init__(self, edges):
        self._edges = edges

    async def get_graph_data(self):
        return [], self._edges


def test_migrate_moves_points_natively_and_downgrade_restores():
    """Upgrade re-keys the EdgeType collection in place (vectors preserved);
    downgrade restores the exact original ids. Uses the LanceDB-contract fake."""
    text = "Alice works at Acme."
    old_id = str(_frozen_bare_id(text))
    new_id = str(_frozen_model_id(text))
    table = _FakeLanceTable(
        [{"id": old_id, "vector": [0.5] * 4, "payload": {"id": old_id, "text": text}}]
    )
    adapter = _make_lance_adapter(table)
    edges = [("a", "b", "works_at", {"edge_text": text})]
    context = MigrationContext(graph_engine=_FakeGraph(edges), vector_engine=adapter)

    asyncio.run(migrate(context))
    assert list(table.rows) == [new_id]
    assert table.rows[new_id]["payload"]["text"] == text
    assert table.rows[new_id]["vector"] == [0.5] * 4  # moved, not re-embedded

    asyncio.run(downgrade(context))
    assert list(table.rows) == [old_id]

    # idempotency: re-running either direction changes nothing further
    asyncio.run(downgrade(context))
    assert list(table.rows) == [old_id]


def test_migrate_tolerates_missing_collection_and_empty_graph():
    adapter = _make_lance_adapter(None)  # collection never created
    context = MigrationContext(
        graph_engine=_FakeGraph([("a", "b", "knows", {})]), vector_engine=adapter
    )
    asyncio.run(migrate(context))  # must not raise

    context_empty = MigrationContext(graph_engine=_FakeGraph([]), vector_engine=adapter)
    asyncio.run(migrate(context_empty))  # no edges -> no-op
