"""Calendar sync hands the user's exact selection to one SDK source per account.

An empty selection retires the synced tables and never builds a source, a bad one
stops before any call to Google, and stale tables are only retired after a clean run.
"""

from contextlib import ExitStack
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.google_calendar import sync as sync_module
from cognee.modules.integrations.google_calendar.adapter import GoogleCalendarIntegration


@pytest.fixture
def sync_mocks():
    with ExitStack() as stack:
        mocks = SimpleNamespace(
            source=Mock(return_value="dlt-source"),
            remember=AsyncMock(return_value=SimpleNamespace(status="completed")),
            record=AsyncMock(),
            token=AsyncMock(return_value="token"),
            retire=AsyncMock(),
        )
        for patcher in (
            patch.object(
                sync_module.ingestion,
                "require_active_credential",
                AsyncMock(side_effect=lambda credential: credential),
            ),
            patch.object(sync_module.ingestion, "retire_resources", mocks.retire),
            patch.object(sync_module.ingestion, "source_factory", return_value=mocks.source),
            patch.object(sync_module.ingestion, "build_service", return_value="service"),
            patch(
                "cognee.modules.integrations.google_calendar.adapter.access_token_for", mocks.token
            ),
            patch.object(
                import_module("cognee.api.v1.remember.remember"), "remember", mocks.remember
            ),
            patch("cognee.modules.integrations.credentials.record_sync_result", mocks.record),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        ):
            stack.enter_context(patcher)
        yield mocks


def credential(**metadata):
    return SimpleNamespace(
        provider_account_id="subject",
        user_id="user-1",
        provider_metadata={"email": "person@example.com", **metadata},
    )


def _dataset_name(email, account_id):
    return sync_module.ingestion.dataset_name_for_account(
        sync_module.GOOGLE_CALENDAR_DATASET_PREFIX, email, account_id
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("calendars", [["primary"], ["primary", "team@group.calendar.google.com"]])
async def test_selected_calendars_reach_the_source_factory(calendars, sync_mocks):
    with patch.object(sync_module.ingestion, "extraction_checkpoint", return_value="checkpoint"):
        await sync_module.sync_google_calendar(credential(selected_calendar_ids=calendars))
    sync_mocks.source.assert_called_once_with(
        calendar_ids=calendars,
        service="service",
        check_active="checkpoint",
        resource_name=sync_module.ingestion.resource_name("google_calendar", credential()),
    )
    sync_mocks.remember.assert_awaited_once_with(
        "dlt-source",
        dataset_name=sync_module._dataset_name(credential()),
        user="owner",
        write_disposition="merge",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_sync_reads_the_selection_the_router_stores(sync_mocks):
    # The router writes the picker's choice under this key. Reading another one would
    # see an empty selection forever and report a clean run.
    key = GoogleCalendarIntegration.resource_selection_key
    await sync_module.sync_google_calendar(credential(**{key: ["primary"]}))
    assert sync_mocks.source.call_args.kwargs["calendar_ids"] == ["primary"]


@pytest.mark.asyncio
async def test_disconnect_while_fetching_the_token_stops_before_ingestion(sync_mocks):
    async def disconnect(_credential):
        sync_module.ingestion.require_active_credential.side_effect = CredentialInactiveError()
        return "token"

    sync_mocks.token.side_effect = disconnect
    with pytest.raises(CredentialInactiveError):
        await sync_module.sync_google_calendar(credential(selected_calendar_ids=["primary"]))
    sync_mocks.remember.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"selected_calendar_ids": []}])
async def test_calendar_is_opt_in_even_without_connector_dependencies(metadata, sync_mocks):
    with patch.object(sync_module.ingestion, "source_factory", side_effect=ImportError):
        await sync_module.sync_google_calendar(credential(**metadata))
    # Every previously synced table goes: an empty selection means "sync nothing".
    sync_mocks.retire.assert_awaited_once_with(
        "google_calendar",
        ANY,
        sync_module._dataset_name(credential()),
        set(),
    )
    sync_mocks.token.assert_not_awaited()
    sync_mocks.remember.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"
    assert sync_mocks.record.await_args.kwargs["counts"]["scanned"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [None, "primary", {}, [None], [""], [" "], ["a", " "]])
async def test_malformed_selection_does_not_sync_the_calendar(selection, sync_mocks):
    # null is rejected too: the connector would read it as its env-var default.
    with pytest.raises(ValueError, match="Calendar selection must be a list"):
        await sync_module.sync_google_calendar(credential(selected_calendar_ids=selection))
    sync_mocks.source.assert_not_called()
    sync_mocks.remember.assert_not_awaited()
    sync_mocks.retire.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"


@pytest.mark.asyncio
async def test_errored_result_counts_a_failure_and_keeps_stale_tables(sync_mocks):
    sync_mocks.remember.return_value = SimpleNamespace(status="errored")
    await sync_module.sync_google_calendar(credential(selected_calendar_ids=["primary"]))
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"
    assert sync_mocks.record.await_args.kwargs["counts"]["failed"] == 1
    assert sync_mocks.record.await_args.kwargs["counts"]["failed_ingestion"] == 1
    # A failed run must not delete anything it could not replace.
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_tables_are_retired_after_a_clean_run_keeping_the_account_resource(sync_mocks):
    await sync_module.sync_google_calendar(credential(selected_calendar_ids=["primary"]))
    sync_mocks.retire.assert_awaited_once_with(
        "google_calendar",
        ANY,
        sync_module._dataset_name(credential()),
        {sync_module.ingestion.resource_name("google_calendar", credential())},
    )


@pytest.mark.asyncio
async def test_failed_calendars_reported_by_the_source_degrade_the_sync(sync_mocks):
    stats = {"scanned": 3, "skipped": 0, "failed": 1, "failed_calendar_sync": 1}
    sync_mocks.source.return_value = SimpleNamespace(cognee_sync_stats=stats)
    await sync_module.sync_google_calendar(credential(selected_calendar_ids=["a", "b"]))
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"
    assert sync_mocks.record.await_args.kwargs["counts"] == stats


@pytest.mark.asyncio
async def test_connector_exception_is_visible_and_counted(sync_mocks):
    sync_mocks.remember.side_effect = RuntimeError("ingestion failed")
    with pytest.raises(RuntimeError, match="ingestion failed"):
        await sync_module.sync_google_calendar(credential(selected_calendar_ids=["primary"]))
    assert sync_mocks.record.await_args.kwargs == {
        "status": "degraded",
        "counts": {"scanned": 0, "skipped": 0, "failed": 1},
    }
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_sync_does_not_race_the_account_cursor(sync_mocks):
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()

    async def remember(*args, **kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace(status="completed")

    sync_mocks.remember.side_effect = remember
    account = credential(selected_calendar_ids=["primary"])
    task = asyncio.create_task(sync_module.sync_google_calendar(account))
    await entered.wait()
    try:
        assert sync_module.ingestion.sync_is_running("google_calendar", "subject")
        await sync_module.sync_google_calendar(account)
        assert sync_mocks.remember.await_count == 1
    finally:
        release.set()
        await task
    assert not sync_module.ingestion.sync_is_running("google_calendar", "subject")


@pytest.mark.parametrize(
    ("email", "account_id", "expected"),
    [
        ("Person@Example.com", "", "google_calendar_person_example_com"),
        ("", "", "google_calendar_account"),
    ],
)
def test_dataset_name_is_a_slug_of_the_email(email, account_id, expected):
    assert _dataset_name(email, account_id) == expected


# The dataset name is a stored identifier: changing the slug rule or the length of the
# hash suffix would orphan every dataset already synced.
@pytest.mark.parametrize(
    ("email", "account_id", "expected"),
    [
        ("a_.b@example.com", "", "google_calendar_a__b_example_com"),
        ("person@example.com", "subject", "google_calendar_person_example_com_a9491f4c1b"),
    ],
)
def test_dataset_name_stays_stable(email, account_id, expected):
    assert _dataset_name(email, account_id) == expected


def test_dataset_name_tells_apart_accounts_whose_slugs_collide():
    first = _dataset_name("a.b@example.com", "one")
    second = _dataset_name("a_b@example.com", "two")
    assert first != second
    assert first.startswith("google_calendar_a_b_example_com_")
