from __future__ import annotations

from collections.abc import Callable, Sequence

# Below this fraction of a group's size, a side is treated as "too imbalanced"
# and the deterministic alphabetical fallback is used instead. Guards against
# recursion depth degrading toward O(n) when a genuine pole-based split keeps
# producing lopsided sides (e.g. many items tie against the same pole).
MIN_SIDE_FRACTION = 0.1


def divisive_split(
    item_ids: Sequence[str],
    similarity_fn: Callable[[str, str], float],
    pole_a_fn: Callable[[list[str]], str],
    max_bucket_size: int,
    group_similarity_fn: Callable[[str, list[str]], float] | None = None,
) -> list[list[str]]:
    """
    Recursively partition ``item_ids`` top-down: pick two "poles" (dissimilar
    reference items), assign every other item to whichever pole it scores
    strictly higher against, and recurse on each side until every group fits
    ``max_bucket_size``.

    ``similarity_fn``/``pole_a_fn`` are pluggable so the same algorithm serves
    both the graph-based (entity) and vector-based (embedding cosine
    similarity) flavors -- see ``graph_distance.py``/``vector_distance.py``.

    Items that score exactly equal against both poles are held aside and
    resolved against the two sides actually formed by the strict winners
    (via ``group_similarity_fn``, e.g. distance from each side's centroid or
    entity union), instead of defaulting to a fixed pole. If
    ``group_similarity_fn`` is omitted, ties default to pole A's side.
    """
    if max_bucket_size < 1:
        raise ValueError("max_bucket_size must be at least 1.")
    if not item_ids:
        return []

    buckets: list[list[str]] = []
    _split(
        sorted(item_ids), similarity_fn, pole_a_fn, max_bucket_size, buckets, group_similarity_fn
    )
    return buckets


def _split(
    ids: list[str],
    similarity_fn: Callable[[str, str], float],
    pole_a_fn: Callable[[list[str]], str],
    max_bucket_size: int,
    buckets: list[list[str]],
    group_similarity_fn: Callable[[str, list[str]], float] | None,
) -> None:
    if len(ids) <= max_bucket_size:
        buckets.append(ids)
        return

    pole_a, pole_b = _choose_poles(ids, similarity_fn, pole_a_fn)

    side_a: list[str] = []
    side_b: list[str] = []
    tied: list[str] = []
    for item_id in ids:
        score_a = similarity_fn(item_id, pole_a)
        score_b = similarity_fn(item_id, pole_b)
        if score_a > score_b:
            side_a.append(item_id)
        elif score_b > score_a:
            side_b.append(item_id)
        else:
            tied.append(item_id)

    _resolve_ties(tied, side_a, side_b, group_similarity_fn)

    if _is_stalled_or_imbalanced(side_a, side_b, len(ids)):
        midpoint = len(ids) // 2
        side_a, side_b = ids[:midpoint], ids[midpoint:]

    _split(side_a, similarity_fn, pole_a_fn, max_bucket_size, buckets, group_similarity_fn)
    _split(side_b, similarity_fn, pole_a_fn, max_bucket_size, buckets, group_similarity_fn)


def _resolve_ties(
    tied: list[str],
    side_a: list[str],
    side_b: list[str],
    group_similarity_fn: Callable[[str, list[str]], float] | None,
) -> None:
    if not tied:
        return

    if group_similarity_fn is None:
        side_a.extend(tied)
        return

    # Score every tied item against a fixed snapshot of the strict winners --
    # not the lists as they grow -- so resolving one tie never shifts the
    # reference group used for the next one.
    strict_side_a = list(side_a)
    strict_side_b = list(side_b)

    for item_id in tied:
        group_score_a = (
            group_similarity_fn(item_id, strict_side_a) if strict_side_a else float("-inf")
        )
        group_score_b = (
            group_similarity_fn(item_id, strict_side_b) if strict_side_b else float("-inf")
        )
        (side_a if group_score_a >= group_score_b else side_b).append(item_id)


def _choose_poles(
    ids: list[str],
    similarity_fn: Callable[[str, str], float],
    pole_a_fn: Callable[[list[str]], str],
) -> tuple[str, str]:
    pole_a = pole_a_fn(ids)

    best_pole_b: str | None = None
    best_score = float("inf")
    for candidate_id in ids:
        if candidate_id == pole_a:
            continue
        score = similarity_fn(pole_a, candidate_id)
        if score < best_score:
            best_score = score
            best_pole_b = candidate_id

    return pole_a, best_pole_b


def _is_stalled_or_imbalanced(side_a: list[str], side_b: list[str], group_size: int) -> bool:
    if not side_a or not side_b:
        return True

    min_side_size = max(1, round(MIN_SIDE_FRACTION * group_size))
    return min(len(side_a), len(side_b)) < min_side_size
