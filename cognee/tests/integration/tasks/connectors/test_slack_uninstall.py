"""An uninstalled Slack workspace's data is deleted 14 days later, unless it is connected again.

Slack's Developer Policy asks for the data of an uninstalled app to be deleted within 14
business days (G28). Real relational rows, a real ``add`` of the workspace's synced data
and of each member's remembered copies, the real event handler, reconnect and ``forget``;
Slack is only the event body and the token response.
"""

import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup
from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.connect import complete_installation
from cognee.modules.integrations.slack import sync as sync_module
from cognee.modules.integrations.slack.adapter import SlackIntegration
from cognee.modules.integrations.slack.handle_slack_event import handle_slack_event
from cognee.modules.integrations.slack.remember_message import (
    CHANNEL_KEY,
    SLACK_DATASET_NAME,
    TEAM_KEY,
    TS_KEY,
    remembered_message_id,
)
from cognee.modules.users.methods import create_user
from cognee.tasks.ingestion.connectors import slack_source
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tests.unit.tasks.connectors.fake_slack import CHANNEL, FakeSlack
from cognee.tests.unit.tasks.connectors.fake_slack import TEAM as FAKE_TEAM

TEAM = "T0UNINSTALL"
OTHER_TEAM = "T0OTHER"
INSTALLER = "U0INSTALLER"


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


async def _connect(owner) -> None:
    await store.upsert_credential(
        provider="slack",
        provider_account_id=TEAM,
        user_id=owner.id,
        token_payload={"access_token": "xoxb-test"},
        provider_metadata={"installed_by_slack_user_id": INSTALLER},
    )


async def _workspace_dataset_name() -> str:
    return sync_module.dataset_name_for_credential(
        await store.get_credential_by_account("slack", TEAM)
    )


async def _sync_a_document(owner) -> None:
    """What a channel sync leaves in the workspace dataset, without the cognify: a real
    ``slack_source`` run, so the row carries the connector's ``system_metadata``."""
    fake = FakeSlack()
    fake.add_user("U1", "ana")
    fake.post("1793872800.000100", "U1", "In #general: the launch moved to Friday.")
    await cognee.add(
        slack_source(
            service=fake,
            team_id=FAKE_TEAM,
            channel_id=CHANNEL,
            resource_name=f"slack_channel_{owner.id.hex}",
        ),
        dataset_name=await _workspace_dataset_name(),
        user=owner,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _remember(user, team: str, ts: str = "1700000000.000100") -> None:
    """The copy "Remember this" stores, without the cognify that follows it."""
    item = DataItem(
        data=f"Remembered from {team} by {user.id}.",
        literal_text=True,
        external_metadata={TEAM_KEY: team, CHANNEL_KEY: "C1", TS_KEY: ts},
        data_id=remembered_message_id(user.id, team_id=team, channel_id="C1", message_ts=ts),
    )
    await cognee.add(item, dataset_name=SLACK_DATASET_NAME, user=user)


async def _items(user, dataset_name: str) -> int:
    count = 0
    for dataset in await get_datasets_by_name(dataset_name, user.id):
        count += len(await get_dataset_data(dataset.id))
    return count


def _event(event_type: str) -> bytes:
    return json.dumps(
        {"type": "event_callback", "team_id": TEAM, "event": {"type": event_type}}
    ).encode()


def _days_later(days: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days, minutes=1)


async def _workspace_with_data():
    owner, member = await _user(), await _user()
    await _connect(owner)
    await _sync_a_document(owner)
    await _remember(owner, TEAM)
    await _remember(member, TEAM)
    await _remember(member, OTHER_TEAM)
    return owner, member


async def _state(owner, member) -> tuple[int, int, int]:
    """(workspace dataset items, owner's remembered copies, member's remembered copies)."""
    return (
        await _items(owner, await _workspace_dataset_name()),
        await _items(owner, SLACK_DATASET_NAME),
        await _items(member, SLACK_DATASET_NAME),
    )


@pytest.mark.asyncio
async def test_an_uninstalled_workspace_is_forgotten_fourteen_days_later(clean_cognee):
    owner, member = await _workspace_with_data()
    assert await _state(owner, member) == (1, 1, 2)

    await handle_slack_event(_event("app_uninstalled"))
    await sync_module.forget_uninstalled_workspaces(now=_days_later(13))
    assert await _state(owner, member) == (1, 1, 2), "nothing before 14 days"

    await sync_module.forget_uninstalled_workspaces(now=_days_later(14))
    # The workspace's dataset and every member's copies from it are gone; the member's
    # copy from another workspace stays.
    assert await _state(owner, member) == (0, 0, 1)

    await sync_module.forget_uninstalled_workspaces(now=_days_later(15))
    assert await _state(owner, member) == (0, 0, 1)


@pytest.mark.asyncio
async def test_a_workspace_connected_again_within_fourteen_days_keeps_its_data(
    clean_cognee, monkeypatch
):
    owner, member = await _workspace_with_data()
    await handle_slack_event(_event("app_uninstalled"))

    integration = SlackIntegration()
    monkeypatch.setattr(
        integration,
        "exchange_code",
        AsyncMock(
            return_value={
                "access_token": "xoxb-again",
                "team": {"id": TEAM},
                "authed_user": {"id": INSTALLER},
                "bot_user_id": "B1",
            }
        ),
    )
    await complete_installation(integration, code="again", user_id=owner.id)

    await sync_module.forget_uninstalled_workspaces(now=_days_later(15))
    assert await _state(owner, member) == (1, 1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["tokens_revoked", "owner_disconnect"])
async def test_a_connection_ended_without_an_uninstall_forgets_nothing(clean_cognee, end):
    owner, member = await _workspace_with_data()
    if end == "tokens_revoked":
        await handle_slack_event(_event("tokens_revoked"))
    else:
        await store.revoke_credential_by_account("slack", TEAM, revoked_by=store.REVOKED_BY_OWNER)

    await sync_module.forget_uninstalled_workspaces(now=_days_later(15))
    assert await _state(owner, member) == (1, 1, 2)


@pytest.mark.asyncio
async def test_a_previous_owners_copy_of_the_workspace_is_forgotten_too(clean_cognee):
    """A connected and synced, disconnected keeping the data, then B connected the same
    workspace and synced: an uninstall deletes both owners' copies (G44)."""
    previous, current = await _user(), await _user()
    await _connect(previous)
    await _sync_a_document(previous)
    await store.revoke_credential_by_account("slack", TEAM, revoked_by=store.REVOKED_BY_OWNER)
    await _connect(current)
    await _sync_a_document(current)
    name = await _workspace_dataset_name()
    assert (await _items(previous, name), await _items(current, name)) == (1, 1)

    await handle_slack_event(_event("app_uninstalled"))
    await sync_module.forget_uninstalled_workspaces(now=_days_later(14))

    assert (await _items(previous, name), await _items(current, name)) == (0, 0)


@pytest.mark.asyncio
async def test_a_deleted_workspace_still_says_when_its_data_was_deleted(clean_cognee):
    """G84: the deletion cleared the uninstall date, so the owner's card then said the data
    "stays in memory" about data already gone. The status keeps both dates, and the next
    tick deletes nothing again."""
    owner, member = await _workspace_with_data()
    await handle_slack_event(_event("app_uninstalled"))
    await sync_module.forget_uninstalled_workspaces(now=_days_later(14))
    assert await _state(owner, member) == (0, 0, 1)

    credential = await store.get_credential_by_account("slack", TEAM)
    integration = SlackIntegration()
    assert integration.data_deleted_after(credential) is not None
    assert integration.data_deleted_at(credential) is not None
    assert await sync_module.forget_uninstalled_workspaces(now=_days_later(15)) == 0


@pytest.mark.asyncio
async def test_an_owner_who_kept_the_data_is_warned_before_an_uninstall_deletes_it(clean_cognee):
    """G84: an owner disconnected keeping the data; the app was then removed from the
    workspace, which deletes it 14 days later. The status for that owner names the date."""
    owner, _ = await _workspace_with_data()
    await store.revoke_credential_by_account("slack", TEAM, revoked_by=store.REVOKED_BY_OWNER)
    await handle_slack_event(_event("app_uninstalled"))

    ended = await store.get_ended_connection(owner.id, "slack")
    assert ended is not None and ended.revoked_by == store.REVOKED_BY_OWNER
    assert SlackIntegration().data_deleted_after(ended) is not None


@pytest.mark.asyncio
async def test_a_workspace_connected_again_while_another_is_deleted_keeps_its_data(
    clean_cognee, monkeypatch
):
    """G75: the uninstalled workspaces were read once, then deleted one after another; one
    connected again while an earlier one was being deleted was deleted too."""
    owner = await _user()
    for team in (TEAM, OTHER_TEAM):
        await store.upsert_credential(
            provider="slack",
            provider_account_id=team,
            user_id=owner.id,
            token_payload={"access_token": "xoxb-test"},
            provider_metadata={"installed_by_slack_user_id": INSTALLER},
        )
        await store.revoke_credential_by_account("slack", team, revoked_by="provider")
        await sync_module.record_uninstall(team)

    deleted: list[str] = []
    original = sync_module._forget_workspace

    async def forget_then_reconnect_the_other(credential):
        deleted.append(credential.provider_account_id)
        await original(credential)
        if len(deleted) == 1:  # meanwhile the other workspace is connected again
            other = OTHER_TEAM if credential.provider_account_id == TEAM else TEAM
            await store.upsert_credential(
                provider="slack",
                provider_account_id=other,
                user_id=owner.id,
                token_payload={"access_token": "xoxb-again"},
                provider_metadata={sync_module.UNINSTALLED_AT_KEY: None},
            )

    monkeypatch.setattr(sync_module, "_forget_workspace", forget_then_reconnect_the_other)
    await sync_module.forget_uninstalled_workspaces(now=_days_later(14))

    assert len(deleted) == 1, deleted


@pytest.mark.asyncio
async def test_a_workspace_connected_again_during_its_own_deletion_is_deleted_after_the_next_uninstall(
    clean_cognee, monkeypatch
):
    """G90: the owner connected again while the workspace's deletion ran; the reconnect
    cleared the uninstall keys, then the finishing deletion stamped "deleted" on the active
    connection, so the next uninstall was skipped as deleted already and the card showed
    the old date."""
    import cognee.tasks.ingestion.ingest_dlt_source as ingest_module

    owner, member = await _workspace_with_data()
    await handle_slack_event(_event("app_uninstalled"))
    integration = SlackIntegration()
    monkeypatch.setattr(
        integration,
        "exchange_code",
        AsyncMock(
            return_value={
                "access_token": "xoxb-again",
                "team": {"id": TEAM},
                "authed_user": {"id": INSTALLER},
                "bot_user_id": "B1",
                "scope": "users:read",
            }
        ),
    )
    drop = ingest_module.drop_dlt_dataset

    async def drop_then_reconnect(name):
        await drop(name)
        await complete_installation(integration, code="again", user_id=owner.id)

    monkeypatch.setattr(ingest_module, "drop_dlt_dataset", drop_then_reconnect)
    await sync_module.forget_uninstalled_workspaces(now=_days_later(14))
    monkeypatch.setattr(ingest_module, "drop_dlt_dataset", drop)

    reconnected = await store.get_credential_by_account("slack", TEAM)
    assert integration.data_deleted_at(reconnected) is None, "no deletion stamp on it"

    await _sync_a_document(owner)
    await handle_slack_event(_event("app_uninstalled"))
    assert await sync_module.forget_uninstalled_workspaces(now=_days_later(15)) == 1
    assert (await _state(owner, member))[0] == 0


# -- Review 2026-11-11 (F8): red until fixed ---------------------------------------------


@pytest.mark.asyncio
async def test_the_deletion_leaves_a_dataset_the_connector_did_not_make(clean_cognee):
    """F8: the 14-day deletion forgot every dataset carrying the workspace's derived name,
    whoever owned it. A user who never connected the workspace can create a dataset of that
    name (it derives from the team id alone), and the uninstall deleted it."""
    owner, _ = await _workspace_with_data()
    outsider = await _user()
    await cognee.add(
        "The outsider's own notes.", dataset_name=await _workspace_dataset_name(), user=outsider
    )
    await handle_slack_event(_event("app_uninstalled"))

    await sync_module.forget_uninstalled_workspaces(now=_days_later(15))

    assert await _items(owner, await _workspace_dataset_name()) == 0
    assert await _items(outsider, await _workspace_dataset_name()) == 1
