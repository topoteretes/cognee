"""``list_calendars`` reads every page and asks for hidden calendars too."""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.google_calendar import client


@pytest.mark.asyncio
async def test_calendars_are_collected_across_pages():
    pages = [
        {"items": [{"id": "a"}], "nextPageToken": "page-2"},
        {"items": [{"id": "b"}, {"id": "c"}]},
    ]
    with patch.object(client, "_authorized_get", AsyncMock(side_effect=pages)) as get:
        calendars = await client.list_calendars("token")

    assert [calendar["id"] for calendar in calendars] == ["a", "b", "c"]
    first, second = (call.args for call in get.await_args_list)
    assert first[1] == client.CALENDAR_LIST_URL
    assert first[2] == "token"
    assert "pageToken" not in first[3]
    assert second[3]["pageToken"] == "page-2"


@pytest.mark.asyncio
async def test_hidden_calendars_are_listed_and_deleted_ones_are_not_asked_for():
    with patch.object(client, "_authorized_get", AsyncMock(return_value={})) as get:
        assert await client.list_calendars("token") == []

    params = get.await_args.args[3]
    # The source lists hidden calendars, so the picker must offer them. showDeleted
    # stays at Google's default, which is the same view the source has.
    assert params["showHidden"] == "true"
    assert "showDeleted" not in params


@pytest.mark.asyncio
async def test_a_failed_page_is_not_swallowed():
    error = RuntimeError("Google Calendar listing failed: HTTP 403")
    with (
        patch.object(client, "_authorized_get", AsyncMock(side_effect=error)),
        pytest.raises(RuntimeError, match="HTTP 403"),
    ):
        await client.list_calendars("token")
