from itertools import permutations
from uuid import UUID

import pytest

from cognee.modules.agent_memory.sanitization import (
    MAX_SERIALIZED_VALUE_LENGTH,
    MAX_TRACE_CONTAINER_ITEMS,
    sanitize_value,
)


@pytest.mark.parametrize("entries", permutations([(1, "integer"), ("1", "string")]))
def test_string_key_keeps_its_name_in_either_order(entries):
    assert sanitize_value(dict(entries)) == {"1": "string", "1_2": "integer"}


@pytest.mark.parametrize(
    "entries", permutations([(1, "integer"), ("1", "string"), ("1_2", "reserved")])
)
def test_collision_suffix_does_not_take_an_existing_key(entries):
    assert sanitize_value(dict(entries)) == {
        "1": "string",
        "1_2": "reserved",
        "1_3": "integer",
    }


@pytest.mark.parametrize("order", ["uuid-first", "string-first"])
def test_uuid_and_string_collisions_are_preserved_inside_nested_containers(order):
    identifier = UUID("00000000-0000-0000-0000-000000000001")
    entries = [(identifier, "uuid"), (str(identifier), "string")]
    if order == "string-first":
        entries.reverse()

    result = sanitize_value({"nested": [dict(entries)]})

    assert result == {"nested": [{str(identifier): "string", f"{identifier}_2": "uuid"}]}


def test_collision_handling_preserves_the_container_limit():
    # The 20-item cap applies before collisions are resolved, so the two
    # colliding keys count towards it and the last two "key_N" entries are cut.
    entries = [(1, "integer"), ("1", "string")]
    entries.extend((f"key_{index}", index) for index in range(MAX_TRACE_CONTAINER_ITEMS))

    result = sanitize_value(dict(entries))

    assert len(result) == MAX_TRACE_CONTAINER_ITEMS
    assert result["1"] == "string"
    assert result["1_2"] == "integer"
    assert f"key_{MAX_TRACE_CONTAINER_ITEMS - 2}" not in result


def test_noncolliding_keys_keep_the_existing_representation():
    assert sanitize_value({"name": "value", 2: (True, None)}) == {
        "name": "value",
        "2": [True, None],
    }


def test_long_keys_are_truncated_like_values():
    key = "x" * (MAX_SERIALIZED_VALUE_LENGTH * 5)

    (result_key,) = sanitize_value({key: 1})

    assert len(result_key) == MAX_SERIALIZED_VALUE_LENGTH
    assert result_key.endswith("...")


def test_object_keys_are_deterministic():
    class Opaque:
        pass

    class WithId:
        id = "abc"

    assert sanitize_value({Opaque(): 1}) == {"<Opaque>": 1}
    assert sanitize_value({WithId(): 1}) == {"WithId:abc": 1}
