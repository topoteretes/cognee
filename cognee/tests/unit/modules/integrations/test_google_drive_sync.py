"""Unit tests for cognee.modules.integrations.google_drive.sync.

Drive and the ingestion path are both mocked. What is under test is the
policy this module owns: which files it can render, what it does with the
ones it cannot, and that one sync costs exactly one pipeline run.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.google_drive import sync as sync_module
from cognee.modules.integrations.google_drive.sync import (
    dataset_name_for_account,
    format_file,
    sync_drive,
)

_DOC = {
    "id": "doc-1",
    "name": "Q3 plan",
    "mimeType": "application/vnd.google-apps.document",
    "webViewLink": "https://docs.google.com/document/d/doc-1",
}
_SHEET = {
    "id": "sheet-1",
    "name": "Budget",
    "mimeType": "application/vnd.google-apps.spreadsheet",
    "webViewLink": "https://docs.google.com/spreadsheets/d/sheet-1",
}
_TEXT = {
    "id": "txt-1",
    "name": "notes.txt",
    "mimeType": "text/plain",
    "webViewLink": "https://drive.google.com/file/d/txt-1",
}
_IMAGE = {
    "id": "img-1",
    "name": "logo.png",
    "mimeType": "image/png",
    "webViewLink": "https://drive.google.com/file/d/img-1",
}


def make_credential(email="goran@topoteretes.com"):
    return SimpleNamespace(
        provider_account_id="110000000000000000001",
        user_id="user-1",
        provider_metadata={"email": email},
    )


@pytest.mark.parametrize(
    "email, readable",
    [
        ("goran@topoteretes.com", "google_drive_goran_topoteretes_com_"),
        ("Name.Surname+tag@Example.CO.UK", "google_drive_name_surname_tag_example_co_uk_"),
        ("", "google_drive_account_"),
    ],
)
def test_dataset_name_is_one_per_account_and_stays_readable(email, readable):
    # One dataset per account, never one per Workspace domain: cognee's
    # permissions are dataset-scoped, so a shared one would answer a
    # colleague's question from someone else's private files.
    name = dataset_name_for_account(email, "110000000000000000001")
    assert name.startswith(readable)
    # Same account, same name: a re-sync must not land in a second dataset.
    assert name == dataset_name_for_account(email, "110000000000000000001")


def test_two_google_accounts_whose_emails_slug_alike_get_different_datasets():
    # Gmail treats dots as insignificant and the slug folds dots, plus signs
    # and hyphens into one underscore, so these render identically. They are
    # different Google accounts and must not share a dataset.
    assert dataset_name_for_account("john.doe@gmail.com", "111") != dataset_name_for_account(
        "john_doe@gmail.com", "222"
    )


def test_account_ids_that_differ_only_at_the_end_still_get_different_datasets():
    # Google subjects are long numbers sharing a prefix, so any fixed-width
    # slice of the id is a collision waiting for the two accounts that differ
    # outside it. The digest has no such window.
    assert dataset_name_for_account("a@b.com", "1" * 20 + "1") != dataset_name_for_account(
        "a@b.com", "1" * 20 + "2"
    )


def test_format_file_is_stable_for_unchanged_content():
    # Nothing volatile in the rendering, so a re-sync of an untouched file
    # produces byte-identical text and does not churn the graph.
    assert format_file(_DOC, "body") == format_file(_DOC, "body")
    rendered = format_file(_DOC, "  body  ")
    assert "Q3 plan" in rendered
    assert "https://docs.google.com/document/d/doc-1" in rendered
    assert rendered.endswith("body")


@pytest.mark.asyncio
async def test_native_documents_are_exported_and_stored_files_downloaded():
    with (
        patch.object(sync_module.client, "export_file", AsyncMock(return_value="exported")),
        patch.object(sync_module.client, "download_file", AsyncMock(return_value="downloaded")),
    ):
        assert await sync_module._read_file("tok", _DOC) == "exported"
        assert await sync_module._read_file("tok", _SHEET) == "exported"
        assert await sync_module._read_file("tok", _TEXT) == "downloaded"
        # A type that needs a parser belongs to the loaders, not here.
        assert await sync_module._read_file("tok", _IMAGE) is None


@pytest.mark.asyncio
async def test_a_sheet_is_exported_as_csv_not_plain_text():
    with patch.object(sync_module.client, "export_file", AsyncMock(return_value="a,b")) as export:
        await sync_module._read_file("tok", _SHEET)

    assert export.await_args.args[2] == "text/csv"


def patched_sync(files=None, *, pages=None, read_side_effect=None, read_return="content"):
    """Patch everything sync_drive reaches out to.

    ``files`` is the shorthand for a single page; ``pages`` takes a list of
    whole page dicts (or exceptions) when the test is about the listing walk
    itself. Returns the unentered stack and a namespace of the mocks every
    assertion here is about.
    """
    remember = AsyncMock(return_value=SimpleNamespace(status="completed"))
    record = AsyncMock()
    if pages is None:
        pages = [{"files": files or []}]
    list_files = AsyncMock(side_effect=pages)
    read = (
        AsyncMock(side_effect=read_side_effect)
        if read_side_effect is not None
        else AsyncMock(return_value=read_return)
    )
    stack = ExitStack()
    for patcher in (
        patch(
            "cognee.modules.integrations.google_drive.adapter.access_token_for",
            AsyncMock(return_value="ya29.token"),
        ),
        patch("cognee.api.v1.remember.remember.remember", remember),
        patch("cognee.modules.integrations.credentials.record_sync_result", record),
        patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        patch.object(sync_module.client, "list_files", list_files),
        patch.object(sync_module, "_read_file", read),
    ):
        stack.enter_context(patcher)
    return stack, SimpleNamespace(remember=remember, record=record, list_files=list_files)


@pytest.mark.asyncio
async def test_one_sync_is_one_pipeline_run_over_every_file():
    stack, mocks = patched_sync([_DOC, _TEXT])
    with stack:
        await sync_drive(make_credential())

    # One remember() for the batch, not one per file.
    mocks.remember.assert_awaited_once()
    documents = mocks.remember.await_args.args[0]
    assert len(documents) == 2
    assert mocks.remember.await_args.kwargs["dataset_name"].startswith(
        "google_drive_goran_topoteretes_com_"
    )
    # improve() is a whole-graph pass with LLM cost; a sync must not spend it.
    assert mocks.remember.await_args.kwargs["self_improvement"] is False


@pytest.mark.asyncio
async def test_unreadable_and_empty_files_are_skipped_not_stored():
    stack, mocks = patched_sync([_DOC, _IMAGE, _TEXT], read_side_effect=["body", None, "   "])
    with stack:
        await sync_drive(make_credential())

    documents = mocks.remember.await_args.args[0]
    assert len(documents) == 1
    assert "body" in documents[0]


@pytest.mark.asyncio
async def test_one_failing_file_does_not_cost_the_whole_sync():
    stack, mocks = patched_sync(
        [_DOC, _TEXT], read_side_effect=[RuntimeError("Google file export failed: HTTP 403"), "ok"]
    )
    with stack:
        await sync_drive(make_credential())

    documents = mocks.remember.await_args.args[0]
    assert len(documents) == 1
    assert "ok" in documents[0]


@pytest.mark.asyncio
async def test_an_account_with_nothing_readable_runs_no_pipeline():
    stack, mocks = patched_sync([_IMAGE], read_return=None)
    with stack:
        await sync_drive(make_credential())

    mocks.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_file_limit_is_honoured():
    # This bounds rendered documents only, which is why _MAX_PAGES exists too.
    stack, mocks = patched_sync([dict(_TEXT, id=f"t{i}") for i in range(10)])
    with stack:
        await sync_drive(make_credential(), file_limit=3)

    assert len(mocks.remember.await_args.args[0]) == 3


@pytest.mark.asyncio
async def test_an_empty_page_carrying_a_next_token_does_not_end_the_walk():
    # Drive filters by permission after cutting a page, so a page of items
    # this account cannot see comes back empty with a live token. Treating
    # that as the end of the Drive silently indexes a prefix, or nothing.
    stack, mocks = patched_sync(pages=[{"files": [], "nextPageToken": "TOK2"}, {"files": [_TEXT]}])
    with stack:
        await sync_drive(make_credential())

    assert mocks.list_files.await_count == 2
    assert len(mocks.remember.await_args.args[0]) == 1


@pytest.mark.asyncio
async def test_a_listing_failure_keeps_what_the_earlier_pages_produced():
    # Nothing retries this sync: no webhook, no scheduler, no re-sync route.
    # Raising here would discard the documents already collected and leave the
    # account permanently empty.
    stack, mocks = patched_sync(
        pages=[
            {"files": [_TEXT], "nextPageToken": "TOK2"},
            RuntimeError("Google file listing failed: HTTP 401"),
        ]
    )
    with stack:
        await sync_drive(make_credential())

    mocks.remember.assert_awaited_once()
    assert len(mocks.remember.await_args.args[0]) == 1
    # A partial result is not a healthy one, and the credential is the only
    # place that can say so.
    assert mocks.record.await_args.kwargs["status"] == sync_module.SYNC_STATUS_DEGRADED


@pytest.mark.asyncio
async def test_a_drive_of_unsupported_files_stops_at_the_page_cap():
    # file_limit counts rendered documents, so a Drive full of images never
    # reaches it. Without a page cap the walk enumerates the whole Drive.
    stack, mocks = patched_sync(
        pages=[{"files": [_IMAGE], "nextPageToken": f"TOK{i}"} for i in range(200)],
        read_return=None,
    )
    with stack:
        await sync_drive(make_credential())

    assert mocks.list_files.await_count == sync_module._MAX_PAGES
    mocks.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_clean_sync_is_stamped_ok():
    stack, mocks = patched_sync([_TEXT])
    with stack:
        await sync_drive(make_credential())

    mocks.record.assert_awaited_once()
    assert mocks.record.await_args.kwargs["status"] == sync_module.SYNC_STATUS_OK


@pytest.mark.asyncio
async def test_a_sync_that_could_not_read_a_file_is_stamped_degraded():
    stack, mocks = patched_sync(
        [_DOC, _TEXT], read_side_effect=[RuntimeError("Google export failed: HTTP 401"), "ok"]
    )
    with stack:
        await sync_drive(make_credential())

    assert mocks.record.await_args.kwargs["status"] == sync_module.SYNC_STATUS_DEGRADED


@pytest.mark.asyncio
async def test_an_empty_account_is_still_stamped():
    # An account with nothing readable runs no pipeline, but it still has to
    # record that a sync happened, or the connection reads as never-synced.
    stack, mocks = patched_sync([_IMAGE], read_return=None)
    with stack:
        await sync_drive(make_credential())

    mocks.remember.assert_not_awaited()
    mocks.record.assert_awaited_once()
