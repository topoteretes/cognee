"""Pure-logic tests for the versioning inverse: ownership partitioning, Model-A
run grouping, and as-of visibility — no databases involved."""

from uuid import uuid4

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.source_refs import make_source_run_ref
from cognee.modules.versioning.methods.as_of_read import _is_visible
from cognee.modules.versioning.methods.inverse import (
    _group_keys_by_owning_run,
    _is_unowned,
)


def test_is_unowned_mirrors_planner_rule():
    key_a, key_b = "source_ref:v1:d:1", "source_ref:v1:d:2"
    assert _is_unowned([key_a], [key_a])
    assert not _is_unowned([key_a, key_b], [key_a])
    assert _is_unowned([key_a, key_b], [key_a, key_b])
    assert _is_unowned([], [])


def test_group_keys_by_owning_run_model_a():
    """Each key maps to at most one run ref (the first attacher); keys without
    a run ref group under None (non-rollbackable-by-run attachments)."""
    dataset_id, run_a, run_b = uuid4(), uuid4(), uuid4()
    key_1 = make_source_ref_key(dataset_id, uuid4())
    key_2 = make_source_ref_key(dataset_id, uuid4())
    key_3 = make_source_ref_key(dataset_id, uuid4())

    run_refs = [
        make_source_run_ref(run_a, key_1),
        make_source_run_ref(run_b, key_2),
    ]

    groups = _group_keys_by_owning_run([key_1, key_2, key_3], run_refs)

    assert groups == {
        str(run_a): [key_1],
        str(run_b): [key_2],
        None: [key_3],
    }


def test_is_visible_requires_allowed_run_and_matching_dataset():
    dataset_id, other_dataset = uuid4(), uuid4()
    run_old, run_new = uuid4(), uuid4()

    own_ref = make_source_run_ref(run_old, make_source_ref_key(dataset_id, uuid4()))
    new_ref = make_source_run_ref(run_new, make_source_ref_key(dataset_id, uuid4()))
    foreign_ref = make_source_run_ref(run_old, make_source_ref_key(other_dataset, uuid4()))

    allowed = {str(run_old)}

    assert _is_visible([own_ref], allowed, str(dataset_id))
    # Attached only by a run after T -> not visible.
    assert not _is_visible([new_ref], allowed, str(dataset_id))
    # Attached at T but for another dataset -> not visible for this dataset.
    assert not _is_visible([foreign_ref], allowed, str(dataset_id))
    # Mixed: any qualifying ref makes it visible.
    assert _is_visible([new_ref, own_ref], allowed, str(dataset_id))
    # No dataset scoping: the foreign ref counts.
    assert _is_visible([foreign_ref], allowed, None)
