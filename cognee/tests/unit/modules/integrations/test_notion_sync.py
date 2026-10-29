"""Notion sync delegates the walk and incremental state to ``notion_source``."""

import asyncio
from contextlib import ExitStack
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch
from uuid import UUID

import httpx
import pytest

from cognee.modules.integrations.notion import client
from cognee.modules.integrations.notion import sync as sync_module
from cognee.modules.integrations.notion.sync import sync_notion, without_nested_roots

WORKSPACE = "ws-1"


def nid(number: int) -> str:
    """A Notion id in its canonical spelling: 32 lowercase hex digits."""
    return f"{number:032x}"


def page(number: int) -> str:
    return f"page:{nid(number)}"


def source(number: int) -> str:
    return f"data_source:{nid(number)}"


def make_credential(*roots):
    return SimpleNamespace(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        provider="notion",
        workspace_id=None,
        provider_account_id=WORKSPACE,
        user_id=UUID("22222222-2222-2222-2222-222222222222"),
        account_label="Topoteretes",
        provider_metadata={"selected_roots": list(roots)},
    )


@pytest.fixture
def sync_mocks():
    with ExitStack() as stack:
        mocks = SimpleNamespace(
            source=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            remember=AsyncMock(return_value=SimpleNamespace(status="completed")),
            record=AsyncMock(),
            retire=AsyncMock(),
            parent=AsyncMock(return_value=None),
            verify=AsyncMock(),
            revoke=AsyncMock(),
        )
        for patcher in (
            patch.object(
                sync_module.ingestion,
                "require_active_credential",
                AsyncMock(side_effect=lambda credential: credential),
            ),
            patch.object(sync_module.ingestion, "retire_resources", mocks.retire),
            patch.object(sync_module.ingestion, "source_factory", return_value=mocks.source),
            patch.object(sync_module.client, "get_parent", mocks.parent),
            patch.object(sync_module.client, "verify_token", mocks.verify),
            patch.object(sync_module, "revoke_credential_if_current", mocks.revoke),
            patch(
                "cognee.modules.integrations.notion.adapter.access_token_for",
                AsyncMock(return_value="token"),
            ),
            patch.object(
                import_module("cognee.api.v1.remember.remember"), "remember", mocks.remember
            ),
            patch("cognee.modules.integrations.credentials.record_sync_result", mocks.record),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        ):
            stack.enter_context(patcher)
        yield mocks


@pytest.mark.asyncio
async def test_each_root_is_its_own_source_with_a_stable_name(sync_mocks):
    credential = make_credential(page(1), source(2), page(1))
    await sync_notion(credential)

    assert sync_mocks.source.call_count == 2
    page_call, source_call = sync_mocks.source.call_args_list
    assert page_call.kwargs == {
        "token": "token",
        "root_page_ids": [nid(1)],
        "root_data_source_ids": [],
        "workspace_id": WORKSPACE,
        "resource_name": sync_module.ingestion.resource_name("notion", credential, page(1)),
        "check_active": ANY,
    }
    assert source_call.kwargs["root_page_ids"] == []
    assert source_call.kwargs["root_data_source_ids"] == [nid(2)]
    assert page_call.kwargs["resource_name"].startswith("notion_pages_")
    for call in sync_mocks.remember.await_args_list:
        assert call.kwargs["dataset_name"] == sync_module.dataset_name_for_credential(credential)
        assert call.kwargs["write_disposition"] == "merge" and call.kwargs["user"] == "owner"


@pytest.mark.asyncio
async def test_two_spellings_of_one_page_are_one_source_and_one_table(sync_mocks):
    dashed = "-".join(
        [nid(255)[:8], nid(255)[8:12], nid(255)[12:16], nid(255)[16:20], nid(255)[20:]]
    )
    credential = make_credential(f"page:{nid(255)}", f"page:{dashed.upper()}")
    await sync_notion(credential)

    assert sync_mocks.source.call_count == 1
    assert sync_mocks.source.call_args.kwargs["root_page_ids"] == [nid(255)]
    assert sync_mocks.source.call_args.kwargs["resource_name"] == (
        sync_module.ingestion.resource_name("notion", credential, page(255))
    )


@pytest.mark.asyncio
async def test_an_empty_selection_retires_everything_and_syncs_nothing(sync_mocks):
    await sync_notion(make_credential())

    sync_mocks.source.assert_not_called()
    sync_mocks.remember.assert_not_awaited()
    sync_mocks.retire.assert_awaited_once_with(
        "notion", ANY, sync_module.dataset_name_for_credential(make_credential()), set()
    )
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [
        ["plain-id"],
        ["folder:" + nid(1)],
        ["page:"],
        [5],
        "page:x",
        ["page:not-a-notion-id"],
        [page(1) + " "],
        ["page:" + nid(1) + "ff"],
    ],
)
async def test_a_malformed_selection_stops_the_sync(sync_mocks, selection):
    credential = make_credential()
    credential.provider_metadata["selected_roots"] = selection

    with pytest.raises(ValueError):
        await sync_notion(credential)

    sync_mocks.remember.assert_not_awaited()
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
async def test_deselected_roots_are_retired_only_after_a_clean_run(sync_mocks):
    credential = make_credential(page(1))
    await sync_notion(credential)

    retained = {sync_module.ingestion.resource_name("notion", credential, page(1))}
    sync_mocks.retire.assert_awaited_once_with(
        "notion", ANY, sync_module.dataset_name_for_credential(credential), retained
    )


@pytest.mark.asyncio
async def test_a_failed_root_retires_nothing_and_degrades_the_sync(sync_mocks):
    sync_mocks.remember.side_effect = [
        SimpleNamespace(status="errored"),
        SimpleNamespace(status="completed"),
    ]
    await sync_notion(make_credential(page(1), page(2)))

    assert sync_mocks.remember.await_count == 2
    sync_mocks.retire.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"
    assert sync_mocks.record.await_args.kwargs["counts"]["failed"] == 1


@pytest.mark.asyncio
async def test_a_rate_limited_root_is_counted_as_a_rate_limit(sync_mocks):
    sync_mocks.remember.side_effect = client.NotionError("search", "rate_limited")

    await sync_notion(make_credential(page(1)))

    counts = sync_mocks.record.await_args.kwargs["counts"]
    assert counts["failed_rate_limit"] == 1 and "failed_ingestion" not in counts


@pytest.mark.asyncio
async def test_any_other_failure_is_counted_as_a_failed_ingestion(sync_mocks):
    sync_mocks.remember.side_effect = client.NotionError("search", "http_500")

    await sync_notion(make_credential(page(1)))

    counts = sync_mocks.record.await_args.kwargs["counts"]
    assert counts["failed_ingestion"] == 1 and "failed_rate_limit" not in counts


@pytest.mark.asyncio
async def test_a_token_notion_no_longer_accepts_revokes_the_connection(sync_mocks):
    sync_mocks.verify.side_effect = client.NotionError("token check", "unauthorized")
    credential = make_credential(page(1))

    with pytest.raises(client.NotionError):
        await sync_notion(credential)

    sync_mocks.revoke.assert_awaited_once_with(credential)
    sync_mocks.remember.assert_not_awaited()
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["network_error", "http_503", "rate_limited"])
async def test_a_failed_token_check_that_is_not_a_rejection_keeps_the_connection(sync_mocks, code):
    sync_mocks.verify.side_effect = client.NotionError("token check", code)

    with pytest.raises(client.NotionError):
        await sync_notion(make_credential(page(1)))

    sync_mocks.revoke.assert_not_awaited()
    sync_mocks.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_root_nested_in_another_selected_root_is_not_synced_twice(sync_mocks):
    # inner page -> block -> outer page, the outer one written in the other spelling
    chain = {
        ("page", nid(1)): ("block", nid(3)),
        ("block", nid(3)): ("page", nid(2).upper()),
        ("page", nid(2)): None,
    }
    sync_mocks.parent.side_effect = lambda _token, kind, object_id: chain[(kind, object_id)]

    await sync_notion(make_credential(page(1), page(2)))

    assert [call.kwargs["root_page_ids"] for call in sync_mocks.source.call_args_list] == [[nid(2)]]


@pytest.mark.asyncio
async def test_the_row_of_a_selected_data_source_is_nested_in_it():
    parents = {("page", nid(1)): ("data_source", nid(2)), ("data_source", nid(2)): None}

    async def parent(_token, kind, object_id):
        return parents[(kind, object_id)]

    with patch.object(client, "get_parent", parent):
        kept = await without_nested_roots("tok", [page(1), source(2)])

    assert kept == [source(2)]


@pytest.mark.asyncio
async def test_roots_in_different_trees_are_both_kept():
    async def parent(_token, kind, object_id):
        return None

    with patch.object(client, "get_parent", parent):
        kept = await without_nested_roots("tok", [page(1), page(2)])

    assert kept == [page(1), page(2)]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["object_not_found", "restricted_resource", "http_403"])
async def test_a_chain_that_cannot_be_followed_keeps_the_root(code):
    async def parent(_token, kind, object_id):
        raise client.NotionError("parent lookup", code)

    with patch.object(client, "get_parent", parent):
        kept = await without_nested_roots("tok", [page(1), page(2)])

    assert kept == [page(1), page(2)]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["http_503", "network_error", "rate_limited"])
async def test_any_other_lookup_failure_stops_the_sync_instead_of_changing_it(code):
    async def parent(_token, kind, object_id):
        raise client.NotionError("parent lookup", code)

    with patch.object(client, "get_parent", parent), pytest.raises(client.NotionError):
        await without_nested_roots("tok", [page(1), page(2)])


@pytest.mark.asyncio
async def test_a_chain_that_loops_ends_and_keeps_the_root():
    parents = {
        ("page", nid(1)): ("page", nid(8)),
        ("page", nid(8)): ("page", nid(9)),
        ("page", nid(9)): ("page", nid(8)),
        ("page", nid(2)): None,
    }

    async def parent(_token, kind, object_id):
        return parents[(kind, object_id)]

    with patch.object(client, "get_parent", parent):
        kept = await without_nested_roots("tok", [page(1), page(2)])

    assert kept == [page(1), page(2)]


@pytest.mark.asyncio
async def test_a_shared_ancestor_is_looked_up_once():
    calls = []

    async def parent(_token, kind, object_id):
        calls.append((kind, object_id))
        return ("page", nid(7)) if object_id != nid(7) else None

    with patch.object(client, "get_parent", parent):
        await without_nested_roots("tok", [page(1), page(2), page(3)])

    assert calls.count(("page", nid(7))) == 1


@pytest.mark.asyncio
async def test_an_end_to_end_walk_tags_pages_with_the_workspace_and_root():
    """The real ``notion_source`` over a fake Notion: rows land under ``notion:<ws>:<root>``."""
    from cognee.tasks.ingestion.connectors.notion import notion_source
    from cognee.tests.unit.tasks.connectors.test_notion_source import (
        FakeNotion,
        _page_object,
        _rich_text_block,
    )

    root, child = "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"
    fake = FakeNotion()
    fake.pages[root] = _page_object(root, "Root", parent={"type": "workspace", "workspace": True})
    fake.blocks[root] = [
        _rich_text_block("b1", "paragraph", "hello root"),
        {
            "id": child,
            "type": "child_page",
            "has_children": False,
            "child_page": {"title": "Child"},
        },
    ]
    fake.pages[child] = _page_object(child, "Child", parent={"type": "page_id", "page_id": root})
    fake.blocks[child] = [_rich_text_block("b2", "paragraph", "hello child")]
    http_client = httpx.Client(transport=httpx.MockTransport(fake.handler))

    def real_source(**kwargs):
        return notion_source(**kwargs, http_client=http_client)

    credential = make_credential(page(1))
    with ExitStack() as stack:
        remember = AsyncMock(return_value=SimpleNamespace(status="completed"))
        stack.enter_context(
            patch.object(sync_module.ingestion, "source_factory", return_value=real_source)
        )
        stack.enter_context(
            patch.object(
                sync_module.ingestion,
                "require_active_credential",
                AsyncMock(side_effect=lambda c: c),
            )
        )
        stack.enter_context(patch.object(sync_module.ingestion, "retire_resources", AsyncMock()))
        stack.enter_context(
            patch.object(sync_module.client, "get_parent", AsyncMock(return_value=None))
        )
        stack.enter_context(patch.object(sync_module.client, "verify_token", AsyncMock()))
        stack.enter_context(
            patch(
                "cognee.modules.integrations.notion.adapter.access_token_for",
                AsyncMock(return_value="token"),
            )
        )
        stack.enter_context(
            patch.object(import_module("cognee.api.v1.remember.remember"), "remember", remember)
        )
        stack.enter_context(
            patch("cognee.modules.integrations.credentials.record_sync_result", AsyncMock())
        )
        stack.enter_context(
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
        )
        await sync_notion(credential)

        source = remember.await_args.args[0]
        # dlt reads the source on a worker thread; its check_active hops back to this loop.
        rows = await asyncio.to_thread(list, source)

    assert {row["id"] for row in rows} == {root, child}
    assert all(row["cognee_node_set"] == [f"notion:{WORKSPACE}:{root}"] for row in rows)
    assert ("GET", "users/me") not in fake.requests
