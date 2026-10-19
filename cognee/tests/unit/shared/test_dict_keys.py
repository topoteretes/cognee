import pytest

from cognee.shared.dict_keys import unique_string_keys


def test_distinct_names_are_returned_unchanged():
    assert unique_string_keys(["a", "b", "c"], [True, False, True]) == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("names", "preferred", "expected"),
    [
        (["1", "1"], [False, True], ["1_2", "1"]),
        (["1", "1"], [True, False], ["1", "1_2"]),
        (["1", "1", "1_2"], [False, True, True], ["1_3", "1", "1_2"]),
        (["1", "1", "1"], [False, True, False], ["1_2", "1", "1_3"]),
    ],
    ids=["preferred-second", "preferred-first", "suffix-skips-literal", "two-losers"],
)
def test_preferred_key_keeps_its_name(names, preferred, expected):
    assert unique_string_keys(names, preferred) == expected


def test_many_colliding_keys_stay_unique_and_ordered():
    count = 5000
    names = ["k"] * count + ["k_2", "k_7"]
    preferred = [False] * count + [True, True]
    result = unique_string_keys(names, preferred)
    assert len(set(result)) == len(names)
    assert result[-2:] == ["k_2", "k_7"]
    # Losers take the free suffixes in order, skipping the literal names.
    assert result[:6] == ["k", "k_3", "k_4", "k_5", "k_6", "k_8"]
