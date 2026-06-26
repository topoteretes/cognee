"""Centroid-slot helpers for truth-subspace reranking.

The core invariant is simple: slot ``i`` always means the centroid stored in
slot ``i`` for the current truth epoch. Everything here is deterministic so a
rebuild from the same learning statements produces the same slots.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence
from uuid import NAMESPACE_OID, UUID, uuid5

from .align import cosine
from .constants import DEFAULT_K, TRUTH_CENTROID_COLLECTION
from .models import TruthCentroidPayload


def centroid_id(dataset_id: str, slot: int) -> UUID:
    return uuid5(NAMESPACE_OID, f"TruthCentroid:{dataset_id}:{slot}")


def learning_id(statement: str) -> str:
    normalized = _normalize_statement(statement)
    return str(uuid5(NAMESPACE_OID, f"TruthLearning:{normalized}"))


def normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def pad_coords(coords: Sequence[float], k: int = DEFAULT_K) -> list[float]:
    values = [float(value) for value in list(coords)[:k]]
    return values + [0.0 for _ in range(max(0, k - len(values)))]


def weighted_centroid(old: Sequence[float], count: int, new: Sequence[float]) -> list[float]:
    old_values = list(old)
    new_values = list(new)
    if not old_values:
        return normalize(new_values)
    safe_count = max(0, int(count))
    merged = [
        (safe_count * old_value + new_value) / (safe_count + 1)
        for old_value, new_value in zip(old_values, new_values)
    ]
    return normalize(merged)


def unique_learning_vectors(
    statements: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> list[tuple[str, list[float]]]:
    unique: dict[str, list[float]] = {}
    for statement, vector in zip(statements, vectors):
        if not str(statement).strip():
            continue
        unique.setdefault(learning_id(statement), list(vector))
    return sorted(unique.items(), key=lambda item: item[0])


def build_centroids_from_learning_vectors(
    dataset_id: str,
    learning_vectors: Sequence[tuple[str, Sequence[float]]],
    truth_epoch: int,
    updated_at: int | None = None,
    k: int = DEFAULT_K,
) -> list[TruthCentroidPayload]:
    return extend_centroids_with_learning_vectors(
        dataset_id,
        [],
        learning_vectors,
        truth_epoch=truth_epoch,
        updated_at=updated_at,
        k=k,
    )


def extend_centroids_with_learning_vectors(
    dataset_id: str,
    existing_centroids: Sequence[TruthCentroidPayload],
    learning_vectors: Sequence[tuple[str, Sequence[float]]],
    truth_epoch: int,
    updated_at: int | None = None,
    k: int = DEFAULT_K,
) -> list[TruthCentroidPayload]:
    if updated_at is None:
        updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)

    slots = [
        {
            "centroid": list(centroid.centroid),
            "count": int(centroid.count),
            "learning_ids": list(centroid.learning_ids),
        }
        for centroid in sorted(existing_centroids, key=lambda centroid: centroid.slot)[:k]
    ]
    seen_learning_ids = {learning_id for slot in slots for learning_id in slot["learning_ids"]}

    for new_learning_id, vector in learning_vectors:
        if new_learning_id in seen_learning_ids:
            continue
        normalized_vector = normalize(vector)
        if len(slots) < k:
            slots.append(
                {"centroid": normalized_vector, "count": 1, "learning_ids": [new_learning_id]}
            )
            seen_learning_ids.add(new_learning_id)
            continue

        nearest_slot = max(
            range(len(slots)),
            key=lambda index: cosine(normalized_vector, slots[index]["centroid"]),
        )
        slot = slots[nearest_slot]
        slot["centroid"] = weighted_centroid(slot["centroid"], slot["count"], normalized_vector)
        slot["count"] += 1
        slot["learning_ids"].append(new_learning_id)
        seen_learning_ids.add(new_learning_id)

    return [
        TruthCentroidPayload(
            dataset_id=str(dataset_id),
            slot=slot_index,
            count=int(slot["count"]),
            truth_epoch=int(truth_epoch),
            updated_at=updated_at,
            centroid=list(slot["centroid"]),
            learning_ids=list(slot["learning_ids"]),
        )
        for slot_index, slot in enumerate(slots)
    ]


def centroids_changed(
    old: Sequence[TruthCentroidPayload],
    new: Sequence[TruthCentroidPayload],
    tolerance: float = 1e-6,
) -> bool:
    if len(old) != len(new):
        return True

    old_by_slot = {centroid.slot: centroid for centroid in old}
    for new_centroid in new:
        old_centroid = old_by_slot.get(new_centroid.slot)
        if old_centroid is None:
            return True
        if old_centroid.count != new_centroid.count:
            return True
        if len(old_centroid.centroid) != len(new_centroid.centroid):
            return True
        if old_centroid.learning_ids != new_centroid.learning_ids:
            return True
        for old_value, new_value in zip(old_centroid.centroid, new_centroid.centroid):
            if abs(float(old_value) - float(new_value)) > tolerance:
                return True
    return False


async def load_centroids(vector_engine, dataset_id: str, k: int = DEFAULT_K):
    centroid_ids = [str(centroid_id(str(dataset_id), slot)) for slot in range(k)]
    rows = await vector_engine.retrieve(TRUTH_CENTROID_COLLECTION, centroid_ids)
    centroids = []
    for row in rows:
        payload = getattr(row, "payload", None)
        if not isinstance(payload, dict):
            continue
        centroid = TruthCentroidPayload.model_validate(payload)
        if centroid.dataset_id == str(dataset_id):
            centroids.append(centroid)
    return sorted(centroids, key=lambda centroid: centroid.slot)


async def upsert_centroids(vector_engine, centroids: Sequence[TruthCentroidPayload]) -> None:
    if not centroids:
        return
    points = [
        {
            "id": centroid_id(centroid.dataset_id, centroid.slot),
            "vector": centroid.centroid,
            "payload": centroid.model_dump(),
        }
        for centroid in centroids
    ]
    await vector_engine.upsert_raw_vectors(
        TRUTH_CENTROID_COLLECTION,
        points,
        payload_schema=TruthCentroidPayload,
    )


def _normalize_statement(statement: str) -> str:
    return " ".join(str(statement).casefold().split())
