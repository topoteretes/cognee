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
    "email, expected",
    [
        ("goran@topoteretes.com", "google_drive_goran_topoteretes_com"),
        ("Name.Surname+tag@Example.CO.UK", "google_drive_name_surname_tag_example_co_uk"),
        ("", "google_drive_account"),
    ],
)
def test_dataset_name_is_one_per_account_and_slugged(email, expected):
    # One dataset per account, never one per Workspace domain: cognee's
    # permissions are dataset-scoped, so a shared one would answer a
    # colleague's question from someone else's private files.
    assert dataset_name_for_account(email) == expected


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


def patched_sync(files, read_side_effect=None, read_return="content"):
    """Patch everything sync_drive reaches out to.

    Returns the unentered stack and the ``remember`` mock, which is what every
    assertion here is about: whether a file made it into the batch, and what
    the batch was run with.
    """
    remember = AsyncMock(return_value=SimpleNamespace(status="completed"))
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
        patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        patch.object(sync_module.client, "list_files", AsyncMock(return_value={"files": files})),
        patch.object(sync_module, "_read_file", read),
    ):
        stack.enter_context(patcher)
    return stack, remember


@pytest.mark.asyncio
async def test_one_sync_is_one_pipeline_run_over_every_file():
    stack, remember = patched_sync([_DOC, _TEXT])
    with stack:
        await sync_drive(make_credential())

    # One remember() for the batch, not one per file.
    remember.assert_awaited_once()
    documents = remember.await_args.args[0]
    assert len(documents) == 2
    assert remember.await_args.kwargs["dataset_name"] == "google_drive_goran_topoteretes_com"
    # improve() is a whole-graph pass with LLM cost; a sync must not spend it.
    assert remember.await_args.kwargs["self_improvement"] is False


@pytest.mark.asyncio
async def test_unreadable_and_empty_files_are_skipped_not_stored():
    stack, remember = patched_sync([_DOC, _IMAGE, _TEXT], read_side_effect=["body", None, "   "])
    with stack:
        await sync_drive(make_credential())

    documents = remember.await_args.args[0]
    assert len(documents) == 1
    assert "body" in documents[0]


@pytest.mark.asyncio
async def test_one_failing_file_does_not_cost_the_whole_sync():
    stack, remember = patched_sync(
        [_DOC, _TEXT], read_side_effect=[RuntimeError("Google file export failed: HTTP 403"), "ok"]
    )
    with stack:
        await sync_drive(make_credential())

    documents = remember.await_args.args[0]
    assert len(documents) == 1
    assert "ok" in documents[0]


@pytest.mark.asyncio
async def test_an_account_with_nothing_readable_runs_no_pipeline():
    stack, remember = patched_sync([_IMAGE], read_return=None)
    with stack:
        await sync_drive(make_credential())

    remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_file_limit_is_honoured():
    # The access token lives about an hour, so an unbounded first sync on a
    # large Drive would outlive it partway through.
    stack, remember = patched_sync([dict(_TEXT, id=f"t{i}") for i in range(10)])
    with stack:
        await sync_drive(make_credential(), file_limit=3)

    assert len(remember.await_args.args[0]) == 3
