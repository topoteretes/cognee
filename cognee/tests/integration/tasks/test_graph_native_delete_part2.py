"""Integration tests for Part 2 graph-native delete/rollback on the default stack.

These are realistic end-to-end tests against the default backends
(Ladybug + LanceDB + SQLite): add → cognify → delete/rollback, asserting that
graph nodes/edges and their vector points disappear for unowned artifacts while
shared / cross-dataset artifacts survive.

They are GATED OFF until Part 1 lands. The graph-native path only activates when
``set_graph_metadata`` / ``attach_*`` are implemented on the real adapters
(Part 1). Until then ``ensure_graph_native_for_new_graph`` returns ``False`` on
the default stack and every graph stays on the relational ledger, so these tests
would exercise the ledger path instead of the graph-native path. Enable once the
Part 1 real adapters land.
"""

import pytest

# COG-5522 Part 1 gate: keep the whole module collected-but-skipped until the
# real adapters implement Lazar's provenance contract.
pytestmark = pytest.mark.skip(
    reason="Part 2 integration gate: enable once Part 1 real adapters land — COG-5522 Part 1"
)


@pytest.mark.asyncio
async def test_delete_by_source_ref_end_to_end():
    """add → cognify two data items in one dataset, delete one; its unowned graph
    nodes/edges and vector points vanish while shared/other-item artifacts remain.
    """
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("Alice founded Acme in Berlin.", dataset_name="ds")
    await cognee.add("Bob joined Globex in Paris.", dataset_name="ds")
    await cognee.cognify(datasets=["ds"])

    # TODO(Part 1): resolve the data id for the first item, call delete via the
    # graph-native route, then assert the first item's unowned nodes/edges and
    # their LanceDB points are gone while the second item's survive.
    raise AssertionError("enable after Part 1")


@pytest.mark.asyncio
async def test_delete_by_dataset_id_preserves_other_dataset():
    """Deleting one dataset leaves a second dataset's artifacts and shared
    entities intact (cross-dataset preservation)."""
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("Alice founded Acme in Berlin.", dataset_name="ds_a")
    await cognee.add("Alice also advises Initech.", dataset_name="ds_b")
    await cognee.cognify(datasets=["ds_a", "ds_b"])

    # TODO(Part 1): delete ds_a; assert ds_a-only nodes/points gone, ds_b intact,
    # shared "Alice" entity survives with ds_b ownership only.
    raise AssertionError("enable after Part 1")


@pytest.mark.asyncio
async def test_rollback_by_pipeline_run_id_end_to_end():
    """A cognify run's solely-introduced artifacts are removed on rollback;
    artifacts owned by an earlier run survive."""
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("Alice founded Acme.", dataset_name="ds")
    await cognee.cognify(datasets=["ds"])
    await cognee.add("Bob founded Globex.", dataset_name="ds")
    # second cognify run introduces new artifacts whose run we will roll back
    await cognee.cognify(datasets=["ds"])

    # TODO(Part 1): capture the second run's pipeline_run_id, roll it back, then
    # assert only that run's artifacts (and their vector points) are gone.
    raise AssertionError("enable after Part 1")
