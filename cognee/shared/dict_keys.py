"""Unique string keys for dicts that are converted to JSON objects."""

from collections.abc import Sequence


def unique_string_keys(names: Sequence[str], preferred: Sequence[bool]) -> list[str]:
    """Make the converted string names of a dict's keys unique, keeping their order.

    ``names[i]`` is the string form of the i-th key, and ``preferred[i]`` says
    whether that key should keep its name when another key converts to the same
    string. Callers prefer keys that already were strings, so ``{1: a, "1": b}``
    always yields ``{"1_2": a, "1": b}`` whatever the insertion order. A key that
    loses gets the first free ``_2``, ``_3``, ... suffix that is not any other
    key's own name, so a suffixed key never takes a literal key's slot.
    """
    if len(set(names)) == len(names):
        return list(names)

    reserved = set(names)
    taken: set[str] = set()
    result = [""] * len(names)
    order = sorted(range(len(names)), key=lambda index: not preferred[index])
    for index in order:
        name = candidate = names[index]
        suffix = 2
        while candidate in taken or (candidate != name and candidate in reserved):
            candidate = f"{name}_{suffix}"
            suffix += 1
        taken.add(candidate)
        result[index] = candidate
    return result
