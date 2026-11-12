"""A message deleted in Slack is forgotten for everyone who remembered it.

Copied from cognee.tests.integration.tasks.connectors.test_slack_remembered_messages (cognee dev, 2026-11-12).

Real relational rows, a real ``add`` of each copy (as "Remember this" stores it) and the
real event handler and ``forget``; Slack is only the event body. The unit tests mock the
lookup of who remembered a message, so who is left out shows only here.
"""

import base64
import json
from uuid import uuid4

import cognee
import pytest
import pytest_asyncio
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup
from cognee.modules.integrations import credentials as store
from cognee.modules.users.methods import create_user
from cognee.tasks.ingestion.data_item import DataItem

from cognee_connector_slack.cognee_host.handle_slack_event import handle_slack_event
from cognee_connector_slack.cognee_host.remember_message import (
    CHANNEL_KEY,
    SLACK_DATASET_NAME,
    TEAM_KEY,
    TS_KEY,
    remembered_message_id,
)

TEAM = "T0REMEMBER"
CHANNEL = "C1"
TS = "1700000000.000100"


@pytest_asyncio.fixture
async def clean_cognee(tmp_path, monkeypatch):
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"k" * 32).decode())
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _user():
    return await create_user(f"slack-{uuid4().hex[:8]}@example.com", "pw-Test-123")


async def _remember(user) -> None:
    """The copy "Remember this" stores, without the cognify that follows it."""
    item = DataItem(
        data=f"In #general, <@U9> said: the launch moved to Friday ({user.id}).",
        literal_text=True,
        external_metadata={TEAM_KEY: TEAM, CHANNEL_KEY: CHANNEL, TS_KEY: TS},
        data_id=remembered_message_id(user.id, team_id=TEAM, channel_id=CHANNEL, message_ts=TS),
    )
    await cognee.add(item, dataset_name=SLACK_DATASET_NAME, user=user)


async def _copies(user) -> int:
    count = 0
    for dataset in await get_datasets_by_name(SLACK_DATASET_NAME, user.id):
        count += len(await get_dataset_data(dataset.id))
    return count


def _deleted() -> bytes:
    return json.dumps(
        {
            "type": "event_callback",
            "team_id": TEAM,
            "event": {
                "type": "message",
                "subtype": "message_deleted",
                "channel": CHANNEL,
                "deleted_ts": TS,
            },
        }
    ).encode()


@pytest.mark.asyncio
async def test_a_deleted_message_is_forgotten_for_a_previous_owner_too(clean_cognee):
    previous_owner, member, new_owner = await _user(), await _user(), await _user()
    # The previous owner connected the workspace and remembered the message, so did a
    # linked member; then the app was uninstalled and someone else connected it again.
    await store.upsert_credential(
        provider="slack",
        provider_account_id=TEAM,
        user_id=previous_owner.id,
        token_payload={"access_token": "xoxb-old"},
    )
    await store.upsert_credential(
        provider="slack_member",
        provider_account_id=f"{TEAM}:U2",
        user_id=member.id,
        token_payload={},
        auth_type="api_key",
    )
    await _remember(previous_owner)
    await _remember(member)
    await store.revoke_credential_by_account("slack", TEAM, revoked_by=store.REVOKED_BY_PROVIDER)
    await store.upsert_credential(
        provider="slack",
        provider_account_id=TEAM,
        user_id=new_owner.id,
        token_payload={"access_token": "xoxb-new"},
    )
    assert (await _copies(previous_owner), await _copies(member)) == (1, 1)

    assert await handle_slack_event(_deleted()) == {"ok": True}

    assert (await _copies(previous_owner), await _copies(member)) == (0, 0)


@pytest.mark.asyncio
async def test_a_member_forgets_their_own_copy_and_no_one_elses(clean_cognee):
    """``@cognee forget`` and "Forget this": only the asking member's copy goes."""
    from cognee_connector_slack.cognee_host.remember_message import forget_my_remembered_message

    member, other = await _user(), await _user()
    await _remember(member)
    await _remember(other)

    forgotten = await forget_my_remembered_message(
        member.id, team_id=TEAM, channel_id=CHANNEL, message_ts=TS
    )

    assert forgotten is True
    assert (await _copies(member), await _copies(other)) == (0, 1)
    assert (
        await forget_my_remembered_message(
            member.id, team_id=TEAM, channel_id=CHANNEL, message_ts=TS
        )
        is False
    ), "nothing left to forget"
