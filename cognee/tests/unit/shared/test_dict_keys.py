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
