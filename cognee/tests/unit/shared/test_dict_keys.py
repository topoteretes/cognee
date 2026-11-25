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


def test_max_length_shortens_the_name_not_the_suffix():
    names = ["abcdefghij", "abcdefghij", "abcdefghij", "abcdefg_2"]
    preferred = [True, False, False, True]

    result = unique_string_keys(names, preferred, max_length=10)

    # "abcdefghij_2" would be 12 chars, so the name is cut to make room for the
    # suffix; the literal "abcdefg_2" is a different key and keeps its slot.
    assert result == ["abcdefghij", "abcdefgh_2", "abcdefgh_3", "abcdefg_2"]
    assert all(len(key) <= 10 for key in result)
    assert len(set(result)) == len(names)


def test_max_length_keeps_a_truncation_marker_and_skips_a_cut_collision():
    names = ["abcdef...", "abcdef...", "abcd..._2"]
    preferred = [True, False, True]

    result = unique_string_keys(names, preferred, max_length=9)

    # The cut candidate "abcd..._2" is a literal key, so the loser moves on to _3.
    assert result == ["abcdef...", "abcd..._3", "abcd..._2"]


def test_many_colliding_keys_stay_unique_and_ordered():
    count = 5000
    names = ["k"] * count + ["k_2", "k_7"]
    preferred = [False] * count + [True, True]
    result = unique_string_keys(names, preferred)
    assert len(set(result)) == len(names)
    assert result[-2:] == ["k_2", "k_7"]
    # Losers take the free suffixes in order, skipping the literal names.
    assert result[:6] == ["k", "k_3", "k_4", "k_5", "k_6", "k_8"]
