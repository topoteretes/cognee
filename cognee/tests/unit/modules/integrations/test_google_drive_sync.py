"""Drive sync delegates extraction and incremental state to the SDK source."""

from contextlib import ExitStack
from hashlib import sha256
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cognee.modules.integrations.google_drive import sync as sync_module
from cognee.modules.integrations.google_drive.sync import dataset_name_for_account, sync_drive


def make_credential(**metadata):
    return SimpleNamespace(
        provider_account_id="subject",
        user_id="user-1",
        provider_metadata={"email": "person@example.com", **metadata},
    )


@pytest.mark.parametrize("email", ["person@example.com", "Name+tag@Example.CO.UK", ""])
def test_dataset_name_is_stable_and_scoped_to_the_account(email):
    name = dataset_name_for_account(email, "subject")
    assert name.startswith("google_drive_")
    assert name == dataset_name_for_account(email, "subject")
    assert name != dataset_name_for_account(email, "another-subject")


def test_accounts_with_colliding_email_slugs_have_distinct_datasets():
    assert dataset_name_for_account("john.doe@gmail.com", "111") != dataset_name_for_account(
        "john_doe@gmail.com", "222"
    )


@pytest.fixture
def sync_mocks():
    with ExitStack() as stack:
        mocks = SimpleNamespace(
            source=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            remember=AsyncMock(return_value=SimpleNamespace(status="completed")),
            record=AsyncMock(),
            token=AsyncMock(return_value="token"),
            drives=AsyncMock(return_value={"drives": []}),
        )
        for patcher in (
            patch.object(sync_module.ingestion, "source_factory", return_value=mocks.source),
            patch.object(sync_module.ingestion, "build_service", return_value="service"),
            patch.object(sync_module.client, "list_drives", mocks.drives),
            patch("cognee.modules.integrations.google_drive.adapter.access_token_for", mocks.token),
            patch.object(
                import_module("cognee.api.v1.remember.remember"), "remember", mocks.remember
            ),
            patch("cognee.modules.integrations.credentials.record_sync_result", mocks.record),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        ):
            stack.enter_context(patcher)
        yield mocks


@pytest.mark.asyncio
async def test_selected_folders_use_stable_independent_sources(sync_mocks):
    await sync_drive(make_credential(selected_folder_ids=["folder-a", "folder-b", "folder-a"]))
    assert sync_mocks.source.call_count == 2
    for call, folder in zip(sync_mocks.source.call_args_list, ["folder-a", "folder-b"]):
        assert call.kwargs == {
            "folder_id": folder,
            "resource_name": f"google_drive_files_{sha256(folder.encode()).hexdigest()[:16]}",
            "service": "service",
        }
    for call in sync_mocks.remember.await_args_list:
        assert not isinstance(call.args[0], list)
        assert call.kwargs == {
            "dataset_name": dataset_name_for_account("person@example.com", "subject"),
            "user": "owner",
            "write_disposition": "merge",
            "primary_key": "id",
            "max_rows_per_table": 0,
            "self_improvement": False,
        }


@pytest.mark.asyncio
async def test_unscoped_sync_includes_shared_drives_across_pages(sync_mocks):
    sync_mocks.drives.side_effect = [
        {"drives": [{"id": "shared-1"}], "nextPageToken": "next"},
        {"drives": [{"id": "shared-2"}]},
    ]
    await sync_drive(make_credential())
    assert [call.kwargs["folder_id"] for call in sync_mocks.source.call_args_list] == [
        "root",
        "shared-1",
        "shared-2",
    ]
    assert "shared_drive_id" not in sync_mocks.source.call_args_list[0].kwargs
    assert sync_mocks.source.call_args_list[1].kwargs["shared_drive_id"] == "shared-1"
    sync_mocks.drives.assert_awaited_with("token", "next")


@pytest.mark.asyncio
async def test_empty_selection_does_not_require_a_connector_or_token(sync_mocks):
    with patch.object(sync_module.ingestion, "source_factory", side_effect=ImportError):
        await sync_drive(make_credential(selected_folder_ids=[]))
    sync_mocks.token.assert_not_awaited()
    sync_mocks.remember.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs == {
        "status": "ok",
        "counts": {"scanned": 0, "skipped": 0, "failed": 0},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["folder", {}, [None], [""]])
async def test_malformed_selection_does_not_widen_to_all_drive(selection, sync_mocks):
    with pytest.raises(ValueError):
        await sync_drive(make_credential(selected_folder_ids=selection))
    sync_mocks.remember.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"


@pytest.mark.asyncio
async def test_missing_connector_is_degraded_and_never_uses_rest_ingestion(sync_mocks):
    with (
        patch.object(sync_module.ingestion, "source_factory", side_effect=RuntimeError("install")),
        pytest.raises(RuntimeError, match="install"),
    ):
        await sync_drive(make_credential())
    sync_mocks.token.assert_not_awaited()
    sync_mocks.remember.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["counts"]["failed"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("failed"), SimpleNamespace(status="errored")])
async def test_one_failed_source_does_not_prevent_remaining_folders(failure, sync_mocks):
    sync_mocks.remember.side_effect = [failure, SimpleNamespace(status="completed")]
    await sync_drive(make_credential(selected_folder_ids=["folder-a", "folder-b"]))
    assert sync_mocks.remember.await_count == 2
    assert sync_mocks.record.await_args.kwargs == {
        "status": "degraded",
        "counts": {"scanned": 0, "skipped": 0, "failed": 1, "failed_ingestion": 1},
    }


@pytest.mark.asyncio
async def test_file_counts_and_skip_reasons_are_aggregated_across_folders(sync_mocks):
    sync_mocks.source.side_effect = lambda **kwargs: SimpleNamespace(
        cognee_sync_stats={"scanned": 5, "skipped": 2, "failed": 0, "skipped_too_large": 2}
    )
    await sync_drive(make_credential(selected_folder_ids=["a", "b"]))
    assert sync_mocks.record.await_args.kwargs["counts"] == {
        "scanned": 10,
        "skipped": 4,
        "failed": 0,
        "skipped_too_large": 4,
    }
