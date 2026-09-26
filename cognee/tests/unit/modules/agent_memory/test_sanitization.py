from itertools import permutations
from uuid import UUID

import pytest

from cognee.modules.agent_memory.sanitization import (
    MAX_TRACE_CONTAINER_ITEMS,
    sanitize_value,
)


@pytest.mark.parametrize("entries", permutations([(1, "integer"), ("1", "string")]))
def test_colliding_keys_preserve_both_values(entries):
    result = sanitize_value(dict(entries))
    assert len(result) == 2
    assert set(result.values()) == {"integer", "string"}
    assert all(isinstance(key, str) for key in result)


@pytest.mark.parametrize(
    "entries", permutations([(1, "integer"), ("1", "string"), ("1_2", "reserved")])
)
def test_collision_suffix_does_not_take_an_existing_key(entries):
    result = sanitize_value(dict(entries))
    assert len(result) == 3
    assert set(result.values()) == {"integer", "string", "reserved"}
    assert result["1_2"] == "reserved"
    assert sanitize_value(dict(entries)) == result


def test_uuid_and_string_collisions_are_preserved_inside_nested_containers():
    identifier = UUID("00000000-0000-0000-0000-000000000001")
    result = sanitize_value({"nested": [{identifier: "uuid", str(identifier): "string"}]})
    assert len(result["nested"][0]) == 2
    assert set(result["nested"][0].values()) == {"uuid", "string"}


def test_collision_handling_preserves_the_container_limit():
    entries = [(1, "integer"), ("1", "string")]
    entries.extend((f"key_{index}", index) for index in range(MAX_TRACE_CONTAINER_ITEMS))
    result = sanitize_value(dict(entries))
    assert len(result) == MAX_TRACE_CONTAINER_ITEMS
    assert set(result.values()) == {"integer", "string", *range(MAX_TRACE_CONTAINER_ITEMS - 2)}


def test_noncolliding_keys_keep_the_existing_representation():
    assert sanitize_value({"name": "value", 2: (True, None)}) == {
        "name": "value",
        "2": [True, None],
    }
