"""Unique string keys for dicts that are converted to JSON objects."""

from collections.abc import Sequence


def _suffixed(name: str, suffix: int, max_length: int | None) -> str:
    """``name_suffix``, cut to ``max_length`` by shortening the name, never the suffix.

    A name that was itself cut (ends in ``...``) keeps that marker, so a bounded
    key still reads as truncated. The uniqueness check runs on the result, so a
    cut that happens to match another key just moves on to the next suffix.
    """
    tail = f"_{suffix}"
    if max_length is None or len(name) + len(tail) <= max_length:
        return name + tail
    keep = max(max_length - len(tail), 0)
    if name.endswith("...") and keep > 3:
        return name[: keep - 3] + "..." + tail
    return name[:keep] + tail


def unique_string_keys(
    names: Sequence[str], preferred: Sequence[bool], *, max_length: int | None = None
) -> list[str]:
    """Make the converted string names of a dict's keys unique, keeping their order.

    ``names[i]`` is the string form of the i-th key, and ``preferred[i]`` says
    whether that key should keep its name when another key converts to the same
    string. Callers prefer keys that already were strings, so ``{1: a, "1": b}``
    always yields ``{"1_2": a, "1": b}`` whatever the insertion order. A key that
    loses gets the first free ``_2``, ``_3``, ... suffix that is not any other
    key's own name, so a suffixed key never takes a literal key's slot.

    ``max_length`` keeps a suffixed key within the cap the caller already
    applied to ``names``: the name is shortened to make room for the suffix.
    Collisions are still resolved on the capped names, because two keys that
    differ only past the cap are the same key once stored.
    """
    if len(set(names)) == len(names):
        return list(names)

    reserved = set(names)
    taken: set[str] = set()
    result = [""] * len(names)
    order = sorted(range(len(names)), key=lambda index: not preferred[index])
    # Where the last key with this name stopped searching. A suffix it skipped
    # stays unavailable (taken, or another key's literal name), so the next
    # loser resumes from there and k colliding keys cost O(k), not O(k^2).
    next_suffix: dict[str, int] = {}
    for index in order:
        name = candidate = names[index]
        suffix = next_suffix.get(name, 2)
        while candidate in taken or (candidate != name and candidate in reserved):
            candidate = _suffixed(name, suffix, max_length)
            suffix += 1
        next_suffix[name] = suffix
        taken.add(candidate)
        result[index] = candidate
    return result
