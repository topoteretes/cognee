"""get_live_events: the delta cursor for the Memory tab's live timeline.

Authorization and session-event collection are mocked (get_authorized_
existing_datasets, collect_session_events); what this pins is the cursor
arithmetic itself — strict '>' filtering so nothing repeats, the response
cursor coming from the newest delivered event rather than being re-derived,
and datetime.now(timezone.utc)-vs-naive-UTC-string comparisons not raising.
"""

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.visualize.visualize import get_live_events as _get_live_events
from cognee.modules.users.exceptions import PermissionDeniedError

visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]

DATASET_ID = "11111111-1111-1111-1111-111111111111"


def _event(time: str, kind: str = "search"):
    return {"kind": kind, "time": time, "qa_id": time}


def _patches(events, authorized=True):
    return (
        patch.object(
            visualize_module,
            "get_authorized_existing_datasets",
            AsyncMock(return_value=[SimpleNamespace(id=DATASET_ID)] if authorized else []),
        ),
        patch.object(visualize_module, "collect_session_events", AsyncMock(return_value=events)),
    )


@pytest.mark.asyncio
async def test_no_since_returns_every_event_and_cursor_is_the_newest():
    events = [_event("2026-08-03T09:00:00.000000"), _event("2026-08-03T09:00:05.000000")]

    ctx_a, ctx_b = _patches(events)
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(DATASET_ID)

    assert result["events"] == events
    assert result["cursor"] == "2026-08-03T09:00:05.000000"


@pytest.mark.asyncio
async def test_since_filters_strictly_greater_so_the_cursor_event_never_repeats():
    events = [
        _event("2026-08-03T09:00:00.000000"),
        _event("2026-08-03T09:00:05.000000"),
        _event("2026-08-03T09:00:10.000000"),
    ]

    ctx_a, ctx_b = _patches(events)
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(
            DATASET_ID,
            since=datetime(2026, 8, 3, 9, 0, 5),  # noqa: DTZ001 - naive cursor, as older event timestamps are
        )

    # The event AT the cursor is excluded — only strictly newer ones return.
    assert result["events"] == [_event("2026-08-03T09:00:10.000000")]
    assert result["cursor"] == "2026-08-03T09:00:10.000000"


@pytest.mark.asyncio
async def test_nothing_new_echoes_the_given_since_back_as_cursor():
    events = [_event("2026-08-03T09:00:00.000000")]
    since = datetime(2026, 8, 3, 9, 30, 0)  # noqa: DTZ001 - naive cursor, as older event timestamps are

    ctx_a, ctx_b = _patches(events)
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(DATASET_ID, since=since)

    assert result["events"] == []
    assert result["cursor"] == since.isoformat()


@pytest.mark.asyncio
async def test_first_call_with_nothing_available_has_a_null_cursor():
    ctx_a, ctx_b = _patches([])
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(DATASET_ID)

    assert result == {"events": [], "cursor": None}


@pytest.mark.asyncio
async def test_timezone_aware_since_is_normalized_before_comparing():
    """Older entries hold naive UTC strings; an aware `since` (a client that
    includes an offset) must not raise trying to compare against them."""
    events = [_event("2026-08-03T09:00:10.000000")]
    aware_since = datetime(2026, 8, 3, 9, 0, 5, tzinfo=timezone.utc)

    ctx_a, ctx_b = _patches(events)
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(DATASET_ID, since=aware_since)

    assert result["events"] == events


@pytest.mark.asyncio
async def test_timezone_aware_event_times_are_normalized_before_comparing():
    """The cache adapters stamp entries with an offset; compared raw against the
    naive cutoff they raised TypeError and failed every live-events poll."""
    events = [_event("2026-08-03T09:00:10.000000+00:00")]

    for since in (
        datetime(2026, 8, 3, 9, 0, 5),  # noqa: DTZ001 - naive cursor from an older client
        datetime(2026, 8, 3, 9, 0, 5, tzinfo=timezone.utc),
    ):
        ctx_a, ctx_b = _patches(events)
        with ctx_a, ctx_b:
            result = await visualize_module.get_live_events(DATASET_ID, since=since)

        assert result["events"] == events
        assert result["cursor"] == "2026-08-03T09:00:10.000000+00:00"


@pytest.mark.asyncio
async def test_event_offsets_are_converted_to_utc_and_mixed_entries_share_one_scale():
    """A non-UTC offset is converted, not ignored, and naive entries from older
    writers sit on the same scale as aware ones from current writers."""
    before_cutoff = _event("2026-08-03T10:59:00+02:00")  # 08:59 UTC
    naive_after = _event("2026-08-03T09:00:06.000000")
    offset_after = _event("2026-08-03T11:00:10+02:00")  # 09:00:10 UTC

    ctx_a, ctx_b = _patches([before_cutoff, naive_after, offset_after])
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(
            DATASET_ID,
            since=datetime(2026, 8, 3, 9, 0, 5),  # noqa: DTZ001 - naive cursor, as older event timestamps are
        )

    assert result["events"] == [naive_after, offset_after]


@pytest.mark.asyncio
async def test_event_time_with_a_trailing_z_is_parsed_on_every_python_version():
    """fromisoformat rejects a trailing "Z" before Python 3.11, which would
    silently drop the event from the delta."""
    events = [_event("2026-08-03T09:00:10Z")]

    ctx_a, ctx_b = _patches(events)
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(
            DATASET_ID,
            since=datetime(2026, 8, 3, 9, 0, 5),  # noqa: DTZ001 - naive cursor, as older event timestamps are
        )

    assert result["events"] == events


@pytest.mark.asyncio
async def test_unauthorized_dataset_raises_permission_denied():
    ctx_a, ctx_b = _patches([], authorized=False)
    with ctx_a, ctx_b, pytest.raises(PermissionDeniedError):
        await visualize_module.get_live_events(DATASET_ID)


@pytest.mark.asyncio
async def test_an_event_with_no_time_is_dropped_by_a_since_filter_rather_than_crashing():
    events = [{"kind": "search", "qa_id": "malformed"}, _event("2026-08-03T09:00:10.000000")]

    ctx_a, ctx_b = _patches(events)
    with ctx_a, ctx_b:
        result = await visualize_module.get_live_events(
            DATASET_ID,
            since=datetime(2026, 8, 3, 9, 0, 0),  # noqa: DTZ001 - naive cursor, as older event timestamps are
        )

    assert result["events"] == [_event("2026-08-03T09:00:10.000000")]


@pytest.mark.asyncio
async def test_the_requested_dataset_is_passed_down_as_the_collection_scope():
    """The scope kwarg is the whole fix, and every test above survives without
    it — they assert on whatever the mocked collector returns, so a dropped
    ``dataset_id=`` would leave them green while re-opening COG-6121."""
    user = SimpleNamespace(id="44444444-4444-4444-4444-444444444444")
    collect = AsyncMock(return_value=[])

    with (
        patch.object(
            visualize_module,
            "get_authorized_existing_datasets",
            AsyncMock(return_value=[SimpleNamespace(id=DATASET_ID)]),
        ),
        patch.object(visualize_module, "collect_session_events", collect),
    ):
        await visualize_module.get_live_events(DATASET_ID, user=user)

    collect.assert_awaited_once_with(user=user, dataset_id=DATASET_ID)
