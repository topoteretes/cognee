"""Gmail ingestion uses SDK DLT state and preserves the user's scope."""

from contextlib import ExitStack
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cognee.modules.integrations.gmail import sync as sync_module


@pytest.fixture
def sync_mocks():
    with ExitStack() as stack:
        mocks = SimpleNamespace(
            source=Mock(return_value="dlt-source"),
            remember=AsyncMock(return_value=SimpleNamespace(status="completed")),
            record=AsyncMock(),
            token=AsyncMock(return_value="token"),
        )
        for patcher in (
            patch.object(sync_module.ingestion, "source_factory", return_value=mocks.source),
            patch.object(sync_module.ingestion, "build_service", return_value="service"),
            patch("cognee.modules.integrations.gmail.adapter.access_token_for", mocks.token),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("labels", [None, ["INBOX"], ["INBOX", "Label_1"]])
async def test_sdk_source_receives_the_exact_scope(labels, sync_mocks):
    await sync_module.sync_gmail(credential(selected_label_ids=labels))
    sync_mocks.source.assert_called_once_with(label_ids=labels, service="service")
    sync_mocks.remember.assert_awaited_once_with(
        "dlt-source",
        dataset_name=sync_module.dataset_name_for_account("person@example.com", "subject"),
        user="owner",
        write_disposition="merge",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"selected_label_ids": []}])
async def test_mailbox_is_opt_in_even_without_connector_dependencies(metadata, sync_mocks):
    with patch.object(sync_module.ingestion, "source_factory", side_effect=ImportError):
        await sync_module.sync_gmail(credential(**metadata))
    sync_mocks.token.assert_not_awaited()
    sync_mocks.remember.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["counts"]["scanned"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["INBOX", {}, [None], [""]])
async def test_malformed_selection_does_not_sync_the_mailbox(selection, sync_mocks):
    with pytest.raises(ValueError):
        await sync_module.sync_gmail(credential(selected_label_ids=selection))
    sync_mocks.remember.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"


@pytest.mark.asyncio
async def test_connector_exception_is_visible_and_does_not_advance_a_core_cursor(sync_mocks):
    sync_mocks.remember.side_effect = RuntimeError("ingestion failed")
    with pytest.raises(RuntimeError, match="ingestion failed"):
        await sync_module.sync_gmail(credential(selected_label_ids=["INBOX"]))
    assert sync_mocks.record.await_args.kwargs == {
        "status": "degraded",
        "counts": {"scanned": 0, "skipped": 0, "failed": 1},
    }


@pytest.mark.asyncio
async def test_errored_result_does_not_stamp_success(sync_mocks):
    sync_mocks.remember.return_value = SimpleNamespace(status="errored")
    await sync_module.sync_gmail(credential(selected_label_ids=["INBOX"]))
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"
    assert sync_mocks.record.await_args.kwargs["counts"]["failed"] == 1


@pytest.mark.asyncio
async def test_partial_counts_survive_a_failed_message_fetch(sync_mocks):
    counts = {"scanned": 7, "skipped": 0, "failed": 1, "failed_message_fetch": 1}
    sync_mocks.source.return_value = SimpleNamespace(cognee_sync_stats=counts)
    sync_mocks.remember.side_effect = RuntimeError("fetch failed")
    with pytest.raises(RuntimeError, match="fetch failed"):
        await sync_module.sync_gmail(credential(selected_label_ids=["INBOX"]))
    assert sync_mocks.record.await_args.kwargs["counts"] == counts
