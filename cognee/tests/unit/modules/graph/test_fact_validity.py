"""fact_validity: one reading of "still current?" shared by ranking and rendering (SDK-90)."""

from cognee.modules.graph.utils.fact_validity import (
    STALE_DISTANCE_FACTOR,
    is_current,
    now_ms,
    to_epoch_ms,
    validity_marker,
)


def test_to_epoch_ms_accepts_ints_numeric_strings_and_edge_timestamps():
    assert to_epoch_ms(1_700_000_000_000) == 1_700_000_000_000
    assert to_epoch_ms("1700000000000") == 1_700_000_000_000
    assert to_epoch_ms("2020-01-01 00:00:00") == 1_577_836_800_000
    assert to_epoch_ms(None) is None
    assert to_epoch_ms("") is None
    assert to_epoch_ms("not a date") is None
    assert to_epoch_ms(True) is None


def test_is_current_defaults_to_now():
    assert is_current(None)
    assert is_current({})
    assert is_current({"valid_to": None})
    assert is_current({"valid_to": now_ms() + 60_000})
    assert not is_current({"valid_to": now_ms() - 60_000})
    assert not is_current({"superseded": True})


def test_is_current_respects_reference_time():
    closed_in_2020 = {"valid_to": "2020-06-01 00:00:00"}
    assert is_current(closed_in_2020, as_of_ms=to_epoch_ms("2019-01-01 00:00:00"))
    assert not is_current(closed_in_2020, as_of_ms=to_epoch_ms("2021-01-01 00:00:00"))
    # Closed exactly at the reference moment is no longer current.
    assert not is_current(closed_in_2020, as_of_ms=to_epoch_ms("2020-06-01 00:00:00"))
    # Supersession is a flag, not a time: stale for any reference.
    assert not is_current({"superseded": True}, as_of_ms=0)


def test_validity_marker():
    assert validity_marker(None) is None
    assert validity_marker({"relationship_type": "ceo_of"}) is None
    assert validity_marker({"valid_to": "2020-06-01 12:00:00"}) == "valid until 2020-06-01"
    assert validity_marker({"superseded": True}) == "superseded by a newer assertion"
    assert validity_marker({"superseded": True, "valid_to": 1_577_836_800_000}) == (
        "valid until 2020-01-01; superseded by a newer assertion"
    )


def test_stale_factor_demotes_but_keeps():
    assert STALE_DISTANCE_FACTOR > 1.0
