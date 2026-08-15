"""provenance_after_move must equal attach(new)+remove(old) in one transition,
preserving per-artifact run ids and staying a no-op on already-moved artifacts
(sweep convergence after interruption)."""

from cognee.infrastructure.databases.provenance.source_ref_state import (
    provenance_after_attach,
    provenance_after_move,
    provenance_after_remove,
)

DATASET = "11111111-1111-4111-8111-111111111111"
RUN = "22222222-2222-4222-8222-222222222222"
OLD_KEY = f"source_ref:v1:{DATASET}:33333333-3333-4333-8333-333333333333"
NEW_KEY = f"source_ref:v1:{DATASET}:44444444-4444-4444-8444-444444444444"
OTHER_KEY = f"source_ref:v1:{DATASET}:55555555-5555-4555-8555-555555555555"


def test_move_replaces_key_and_rewrites_run_refs():
    old_ref = f"source_run_ref:v1:{RUN}:{OLD_KEY}"

    cols = provenance_after_move([OLD_KEY], [old_ref], OLD_KEY, NEW_KEY)

    assert cols.source_ref_keys == [NEW_KEY]
    assert cols.source_run_refs == [f"source_run_ref:v1:{RUN}:{NEW_KEY}"]
    assert cols.source_run_ids == [RUN]
    assert cols.source_dataset_ids == [DATASET]


def test_move_matches_attach_then_remove_end_state():
    old_ref = f"source_run_ref:v1:{RUN}:{OLD_KEY}"

    moved = provenance_after_move([OLD_KEY], [old_ref], OLD_KEY, NEW_KEY)
    attached = provenance_after_attach([OLD_KEY], [old_ref], [NEW_KEY], RUN)
    legacy = provenance_after_remove(attached.source_ref_keys, attached.source_run_refs, [OLD_KEY])

    assert set(moved.source_ref_keys) == set(legacy.source_ref_keys)
    assert set(moved.source_run_refs) == set(legacy.source_run_refs)
    assert moved.source_dataset_ids == legacy.source_dataset_ids
    assert moved.source_run_ids == legacy.source_run_ids


def test_move_preserves_each_artifacts_own_run_ids():
    other_run = "66666666-6666-4666-8666-666666666666"
    refs = [
        f"source_run_ref:v1:{RUN}:{OLD_KEY}",
        f"source_run_ref:v1:{other_run}:{OLD_KEY}",
    ]

    cols = provenance_after_move([OLD_KEY], refs, OLD_KEY, NEW_KEY)

    assert cols.source_run_refs == [
        f"source_run_ref:v1:{RUN}:{NEW_KEY}",
        f"source_run_ref:v1:{other_run}:{NEW_KEY}",
    ]
    assert sorted(cols.source_run_ids) == sorted([RUN, other_run])


def test_move_leaves_other_keys_untouched():
    other_ref = f"source_run_ref:v1:{RUN}:{OTHER_KEY}"
    old_ref = f"source_run_ref:v1:{RUN}:{OLD_KEY}"

    cols = provenance_after_move([OTHER_KEY, OLD_KEY], [other_ref, old_ref], OLD_KEY, NEW_KEY)

    assert cols.source_ref_keys == [OTHER_KEY, NEW_KEY]
    assert other_ref in cols.source_run_refs


def test_move_is_noop_when_old_key_absent():
    other_ref = f"source_run_ref:v1:{RUN}:{OTHER_KEY}"

    cols = provenance_after_move([OTHER_KEY], [other_ref], OLD_KEY, NEW_KEY)

    assert cols.source_ref_keys == [OTHER_KEY]
    assert cols.source_run_refs == [other_ref]
    assert NEW_KEY not in cols.source_ref_keys


def test_move_dedupes_when_new_key_already_present():
    refs = [
        f"source_run_ref:v1:{RUN}:{OLD_KEY}",
        f"source_run_ref:v1:{RUN}:{NEW_KEY}",
    ]

    cols = provenance_after_move([OLD_KEY, NEW_KEY], refs, OLD_KEY, NEW_KEY)

    assert cols.source_ref_keys == [NEW_KEY]
    assert cols.source_run_refs == [f"source_run_ref:v1:{RUN}:{NEW_KEY}"]
