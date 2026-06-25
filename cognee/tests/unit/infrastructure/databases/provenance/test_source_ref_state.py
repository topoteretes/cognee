"""Fast, DB-free unit tests for the pure source-ref state transitions (COG-5522).

These cover the Part 0 contract invariants directly, without spinning up a graph
backend — so a regression in the derive/merge logic is caught instantly.
"""

from uuid import uuid4

from cognee.infrastructure.databases.provenance.source_refs import (
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.infrastructure.databases.provenance.source_ref_state import (
    provenance_after_attach,
    provenance_after_remove,
)


def test_attach_materializes_and_dedupes():
    d1, r1 = uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d1, uuid4())  # same dataset

    cols = provenance_after_attach([], [], [key_a], str(r1))
    assert cols.source_ref_keys == [key_a]
    assert cols.source_dataset_ids == [str(d1)]
    assert cols.source_run_ids == [str(r1)]
    assert cols.source_run_refs == [make_source_run_ref(r1, key_a)]

    cols = provenance_after_attach(cols.source_ref_keys, cols.source_run_refs, [key_b], str(r1))
    assert set(cols.source_ref_keys) == {key_a, key_b}
    assert cols.source_dataset_ids == [str(d1)]  # both refs share the dataset
    assert cols.source_run_ids == [str(r1)]


def test_reattach_same_run_and_key_adds_nothing():
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    cols = provenance_after_attach([], [], [key], str(r1))
    cols = provenance_after_attach(cols.source_ref_keys, cols.source_run_refs, [key], str(r1))
    assert cols.source_ref_keys == [key]
    assert cols.source_run_refs == [make_source_run_ref(r1, key)]


def test_new_run_on_existing_key_records_a_run_ref():
    d1, r1, r2 = uuid4(), uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    cols = provenance_after_attach([], [], [key], str(r1))
    cols = provenance_after_attach(cols.source_ref_keys, cols.source_run_refs, [key], str(r2))
    assert cols.source_ref_keys == [key]
    assert sorted(cols.source_run_refs) == sorted(
        [make_source_run_ref(r1, key), make_source_run_ref(r2, key)]
    )
    assert cols.source_run_ids == sorted([str(r1), str(r2)])


def test_attach_without_run_is_not_rollbackable():
    d1 = uuid4()
    key = make_source_ref_key(d1, uuid4())
    cols = provenance_after_attach([], [], [key], None)
    assert cols.source_ref_keys == [key]
    assert cols.source_dataset_ids == [str(d1)]
    assert cols.source_run_ids == []
    assert cols.source_run_refs == []


def test_remove_keeps_dataset_id_when_sibling_ref_shares_it():
    d1, r1, r2 = uuid4(), uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d1, uuid4())  # same dataset
    cols = provenance_after_attach([], [], [key_a], str(r1))
    cols = provenance_after_attach(cols.source_ref_keys, cols.source_run_refs, [key_b], str(r2))
    cols = provenance_after_remove(cols.source_ref_keys, cols.source_run_refs, [key_a])
    assert cols.source_ref_keys == [key_b]
    assert cols.source_dataset_ids == [str(d1)]  # survives via key_b
    assert cols.source_run_ids == [str(r2)]
    assert cols.source_run_refs == [make_source_run_ref(r2, key_b)]


def test_remove_keeps_run_id_when_sibling_ref_shares_it():
    d1, d2, r1 = uuid4(), uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d2, uuid4())
    cols = provenance_after_attach([], [], [key_a, key_b], str(r1))
    cols = provenance_after_remove(cols.source_ref_keys, cols.source_run_refs, [key_a])
    assert cols.source_ref_keys == [key_b]
    assert cols.source_run_ids == [str(r1)]  # survives via key_b's run ref
    assert cols.source_run_refs == [make_source_run_ref(r1, key_b)]


def test_remove_last_ref_drops_dataset_and_run():
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    cols = provenance_after_attach([], [], [key], str(r1))
    cols = provenance_after_remove(cols.source_ref_keys, cols.source_run_refs, [key])
    assert cols.source_ref_keys == []
    assert cols.source_dataset_ids == []
    assert cols.source_run_ids == []
    assert cols.source_run_refs == []


def test_remove_is_idempotent():
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    cols = provenance_after_attach([], [], [key], str(r1))
    once = provenance_after_remove(cols.source_ref_keys, cols.source_run_refs, [key])
    twice = provenance_after_remove(once.source_ref_keys, once.source_run_refs, [key])
    assert twice == once
