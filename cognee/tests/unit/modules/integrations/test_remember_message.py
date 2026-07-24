"""Unit tests for cognee.modules.integrations.slack.remember_message.

Invariants: the enriched text carries channel/author provenance (falling
back gracefully when either is missing), the Slack dataset name is used,
and the background mode is always requested (no LLM work should block the
3-second interactive-payload window).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.integrations.slack.remember_message import (
    SLACK_DATASET_NAME,
    remember_message,
)

MODULE = "cognee.modules.integrations.slack.remember_message"


@pytest.mark.asyncio
async def test_remembers_with_channel_and_author():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(owner.id, text="ship it", channel_name="general", author_id="U1")

    remember.assert_awaited_once()
    (text,), kwargs = remember.call_args
    assert text == "In #general, <@U1> said: ship it"
    assert kwargs["dataset_name"] == SLACK_DATASET_NAME
    assert kwargs["user"] is owner
    assert kwargs["run_in_background"] is True


@pytest.mark.asyncio
async def test_falls_back_when_channel_and_author_are_missing():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(owner.id, text="ship it")

    (text,), _ = remember.call_args
    assert text == "In Slack, someone said: ship it"
