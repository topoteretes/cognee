"""Unit tests for cognee_connector_slack.cognee_host.remember_message.

Copied from cognee/tests/unit/modules/integrations/test_remember_message.py (cognee dev, 2026-10-27).

Invariants: the enriched text carries channel/author provenance (falling
back gracefully when either is missing), the Slack dataset name is used,
the source node_set is stamped so the origin survives into the graph,
and the background mode is always requested (no LLM work should block the
3-second interactive-payload window).
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage

from cognee_connector_slack.cognee_host.remember_message import (
    SLACK_DATASET_NAME,
    SLACK_NODE_SET,
    remember_message,
    remember_note,
)

MODULE = "cognee_connector_slack.cognee_host.remember_message"


@pytest.mark.asyncio
async def test_remembers_with_channel_and_author():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(owner.id, text="ship it", channel_name="general", author_id="U1")

    remember.assert_awaited_once()
    (item,), kwargs = remember.call_args
    assert item.data == "In #general, <@U1> said: ship it"
    assert item.literal_text is True
    assert kwargs["dataset_name"] == SLACK_DATASET_NAME
    assert kwargs["user"] is owner
    assert kwargs["run_in_background"] is True
    # Origin stamp: without this, a Slack-sourced document reaches the graph
    # with no structured Slack marker at all (only the dataset name and the
    # prose prefix), so it can't be grouped, colored, or filtered by source.
    assert kwargs["node_set"] == SLACK_NODE_SET == ["slack"]


@pytest.mark.asyncio
async def test_falls_back_when_channel_and_author_are_missing():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(owner.id, text="ship it")

    (item,), _ = remember.call_args
    assert item.data == "In Slack, someone said: ship it"


@pytest.mark.asyncio
@pytest.mark.parametrize("remember_text", [remember_message, remember_note])
async def test_typed_text_is_never_read_as_a_local_file(tmp_path, monkeypatch, remember_text):
    """A path typed in Slack stays text, through cognee's real ingestion functions.

    The provenance prefix is no protection: resolving ``<prefix> ../../file`` collapses
    the ``..`` segments through the prefix and lands on the real file.
    """
    secret = tmp_path / "outside" / "canary.txt"
    secret.parent.mkdir()
    secret.write_text("CANARY")
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    # The first ".." joins the prefix into one segment, which the second one removes.
    typed = "../../" + os.path.relpath(secret, working_directory)

    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_text(owner.id, text=typed)
    (item,), _ = remember.call_args

    assert await resolve_data_directories([item]) == [item]
    stored = await save_data_item_to_storage(item)
    stored_path = Path(stored.removeprefix("file://"))
    assert stored_path.resolve() != secret.resolve()
    assert typed in stored_path.read_text()
    assert "CANARY" not in stored_path.read_text()
