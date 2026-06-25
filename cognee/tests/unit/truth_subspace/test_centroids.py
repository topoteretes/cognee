import math
from uuid import UUID

import pytest

from cognee.modules.truth_subspace.centroids import (
    build_centroids_from_learning_vectors,
    centroid_id,
    centroids_changed,
    learning_id,
    normalize,
    pad_coords,
    unique_learning_vectors,
    weighted_centroid,
)


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
