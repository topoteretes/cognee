import math
from uuid import UUID

import pytest

from cognee.modules.truth_subspace.centroids import (
    build_centroids_from_learning_vectors,
    centroid_id,
    centroids_changed,
    extend_centroids_with_learning_vectors,
    learning_id,
    normalize,
    pad_coords,
    unique_learning_vectors,
    weighted_centroid,
)
from cognee.modules.truth_subspace.models import TruthCentroidPayload


def test_centroid_id_is_deterministic_uuid():
    first = centroid_id("dataset-1", 0)
    second = centroid_id("dataset-1", 0)
    assert first == second
    assert isinstance(first, UUID)
    assert first != centroid_id("dataset-1", 1)


def test_learning_id_normalizes_whitespace_and_case():
    assert learning_id(" Coffee  Matters ") == learning_id("coffee matters")


def test_normalize_zero_vector_preserves_shape():
    assert normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_pad_coords_keeps_fixed_slot_count():
    assert pad_coords([0.1, 0.2], k=4) == [0.1, 0.2, 0.0, 0.0]
    assert pad_coords([0.1, 0.2, 0.3], k=2) == [0.1, 0.2]


def test_weighted_centroid_updates_toward_new_vector():
    updated = weighted_centroid([1.0, 0.0], 1, [0.0, 1.0])
    assert math.isclose(updated[0], updated[1], rel_tol=1e-9)


def test_unique_learning_vectors_deduplicates_by_statement_text():
    pairs = unique_learning_vectors(
        ["Coffee matters", " coffee   matters ", "Tea matters"],
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
    )
    assert len(pairs) == 2


def test_build_centroids_creates_slots_until_limit():
    learning_vectors = [(str(index), [1.0, float(index), 0.0]) for index in range(10)]
    centroids = build_centroids_from_learning_vectors(
        "dataset-1",
        learning_vectors,
        truth_epoch=3,
        updated_at=123,
        k=8,
    )

    assert [centroid.slot for centroid in centroids] == list(range(8))
    assert all(centroid.truth_epoch == 3 for centroid in centroids)
    assert sum(centroid.count for centroid in centroids) == 10


def test_build_centroids_is_deterministic_for_same_inputs():
    learning_vectors = [(str(index), [1.0, float(index), 0.0]) for index in range(10)]
    first = build_centroids_from_learning_vectors("dataset-1", learning_vectors, 1, 123)
    second = build_centroids_from_learning_vectors("dataset-1", learning_vectors, 9, 123)

    assert [centroid.centroid for centroid in first] == [centroid.centroid for centroid in second]
    assert [centroid.count for centroid in first] == [centroid.count for centroid in second]


def test_extend_centroids_adds_new_learning_ids_without_double_counting():
    existing = build_centroids_from_learning_vectors(
        "dataset-1",
        [("a", [1.0, 0.0])],
        truth_epoch=1,
        updated_at=123,
        k=2,
    )

    first = extend_centroids_with_learning_vectors(
        "dataset-1",
        existing,
        [("a", [1.0, 0.0]), ("b", [0.0, 1.0])],
        truth_epoch=2,
        updated_at=456,
        k=2,
    )
    second = extend_centroids_with_learning_vectors(
        "dataset-1",
        first,
        [("b", [0.0, 1.0])],
        truth_epoch=3,
        updated_at=789,
        k=2,
    )

    assert sum(centroid.count for centroid in first) == 2
    assert [centroid.learning_ids for centroid in first] == [["a"], ["b"]]
    assert [centroid.count for centroid in second] == [centroid.count for centroid in first]


def test_centroids_changed_ignores_epoch_only_changes():
    learning_vectors = [("a", [1.0, 0.0]), ("b", [0.0, 1.0])]
    old = build_centroids_from_learning_vectors("dataset-1", learning_vectors, 1, 123)
    new = build_centroids_from_learning_vectors("dataset-1", learning_vectors, 2, 123)

    assert centroids_changed(old, new) is False


def test_centroids_changed_detects_count_or_vector_changes():
    old = build_centroids_from_learning_vectors("dataset-1", [("a", [1.0, 0.0])], 1, 123)
    new = build_centroids_from_learning_vectors(
        "dataset-1",
        [("a", [1.0, 0.0]), ("b", [0.0, 1.0])],
        1,
        123,
    )

    assert centroids_changed(old, new) is True


def test_extend_centroids_preserves_slot_identity_across_gaps():
    built = build_centroids_from_learning_vectors(
        "dataset-1",
        [("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])],
        truth_epoch=1,
        updated_at=123,
        k=4,
    )
    assert [centroid.slot for centroid in built] == [0, 1, 2]

    # Simulate slot 1 having gone missing from the vector store (e.g. a dropped
    # or unparseable row), leaving a gap between the survivors at slots 0 and 2.
    gapped = [centroid for centroid in built if centroid.slot != 1]
    assert [centroid.slot for centroid in gapped] == [0, 2]

    # Re-running with no new learnings must NOT positionally renumber the
    # survivors: slot 2 has to stay slot 2 or its persisted point id moves.
    rebuilt = extend_centroids_with_learning_vectors(
        "dataset-1",
        gapped,
        [],
        truth_epoch=2,
        updated_at=456,
        k=4,
    )
    assert [centroid.slot for centroid in rebuilt] == [0, 2]

    original_slot2 = next(centroid for centroid in built if centroid.slot == 2)
    rebuilt_slot2 = next(centroid for centroid in rebuilt if centroid.slot == 2)
    assert rebuilt_slot2.learning_ids == original_slot2.learning_ids
    assert rebuilt_slot2.centroid == original_slot2.centroid
    assert centroid_id("dataset-1", rebuilt_slot2.slot) == centroid_id("dataset-1", 2)


def test_extend_centroids_fills_gap_before_appending():
    built = build_centroids_from_learning_vectors(
        "dataset-1",
        [("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])],
        truth_epoch=1,
        updated_at=123,
        k=4,
    )
    gapped = [centroid for centroid in built if centroid.slot != 1]

    rebuilt = extend_centroids_with_learning_vectors(
        "dataset-1",
        gapped,
        [("d", [0.2, 0.9])],
        truth_epoch=2,
        updated_at=456,
        k=4,
    )

    # The new learning reuses the freed slot 1 instead of colliding with slot 2.
    assert sorted(centroid.slot for centroid in rebuilt) == [0, 1, 2]
    filled = next(centroid for centroid in rebuilt if centroid.slot == 1)
    assert filled.learning_ids == ["d"]


def test_extend_centroids_drops_slots_outside_addressable_range():
    # A slot >= k can never be reloaded by load_centroids (it reads range(k)),
    # so it must be dropped, never renumbered into a live slot.
    out_of_range = TruthCentroidPayload(
        dataset_id="dataset-1",
        slot=9,
        count=1,
        truth_epoch=1,
        updated_at=123,
        centroid=[1.0, 0.0],
        learning_ids=["x"],
    )
    rebuilt = extend_centroids_with_learning_vectors(
        "dataset-1",
        [out_of_range],
        [("y", [0.0, 1.0])],
        truth_epoch=2,
        updated_at=456,
        k=4,
    )

    assert [centroid.slot for centroid in rebuilt] == [0]
    assert rebuilt[0].learning_ids == ["y"]
