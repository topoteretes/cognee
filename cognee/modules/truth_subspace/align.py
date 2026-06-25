"""Pure truth-subspace alignment functions.

No I/O, no database access, no LLM calls — just deterministic math over plain
python lists. Everything here is NEUTRAL when inputs are missing/empty/zero:
``truth_score`` returns ``0.5`` and ``truth_factor`` returns ``1.0`` so callers
that pass nothing leave baseline scoring untouched.
"""

import hashlib
import math
from typing import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 for a zero/empty vector."""
    if not a or not b:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def node_coords(node_vec: Sequence[float], basis_vecs: Sequence[Sequence[float]]) -> list[float]:
    """Project ``node_vec`` onto each basis vector using cosine similarity.

    The result is zero-padded to ``len(basis_vecs)`` so the coordinate vector
    always has one entry per basis vector.
    """
    coords = [cosine(node_vec, basis_vec) for basis_vec in basis_vecs]
    # cosine already yields 0.0 per vector, so length == len(basis_vecs) holds;
    # pad defensively to keep the contract explicit.
    while len(coords) < len(basis_vecs):
        coords.append(0.0)
    return coords


def query_coords(q_vec: Sequence[float], basis_vecs: Sequence[Sequence[float]]) -> list[float]:
    """Project a query vector onto each basis vector, zero-padded."""
    return node_coords(q_vec, basis_vecs)


def truth_score(node_coords: Sequence[float], q_coords: Sequence[float]) -> float:
    """Truth score in [0, 1]: the node's alignment with directions the query cares about.

    A query-relevance-weighted average of the node's per-direction alignments, using
    the (clamped) query coordinates as weights. This is magnitude-sensitive on
    purpose: a node strongly aligned with those directions scores higher. Cosine of the
    two coord vectors does NOT work here — every basis cosine is positive, so all
    coord vectors share one octant and their cosine collapses to ~1 regardless of
    magnitude, erasing the very signal we rank on.

    Returns ``0.5`` (NEUTRAL) when either coord vector is empty, or when the query
    aligns with no direction (no weight to spread).
    """
    if not node_coords or not q_coords:
        return 0.5

    weights = [max(float(q), 0.0) for q in q_coords]
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.5

    weighted = sum(float(n) * w for n, w in zip(node_coords, weights))
    return max(0.0, min(1.0, weighted / total_weight))


def truth_factor(node_coords: Sequence[float], q_coords: Sequence[float]) -> float:
    """Multiplicative score factor in [0.75, 1.25].

    ``0.75 + 0.5 * truth_score``. Returns ``1.0`` (NEUTRAL) when coords are
    missing/zero, since ``truth_score`` is ``0.5`` there.
    """
    return 0.75 + 0.5 * truth_score(node_coords, q_coords)


def stable_signature(ordered_ids: Sequence[object]) -> str:
    """Stable sha1 signature of an ordered id sequence."""
    joined = "|".join(str(item_id) for item_id in ordered_ids)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
