"""What removes a synced Slack channel from memory: deselecting it does, the allowlist never.

Copied from cognee.tests.integration.tasks.connectors.test_slack_allowlist (cognee dev, 2026-11-12).

The channel allowlist limits where Cognee's commands work. End to end: a real credential row, the real ``sync_slack`` (remember, cognify over the
journeys tier's mock LLM, retirement), Slack faked in memory. The unit sync tests
replace the retirement, so a sync that removes a channel's documents shows only here.
"""

import asyncio
import base64
import contextlib
from unittest.mock import AsyncMock, patch

import cognee
import pytest
import pytest_asyncio
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup
from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import (
    get_credential_by_account,
    update_provider_metadata,
    upsert_credential,
)
from cognee.modules.users.methods import get_default_user
from fake_slack import TEAM, FakeSlack

from cognee_connector_slack import source as slack_source_module
from cognee_connector_slack.cognee_host import sync as sync_module
from cognee_connector_slack.source import SlackUnavailableError

INSTALLER = "U0INSTALLER"
PUBLIC = "C1"
PRIVATE = "C2"


@pytest_asyncio.fixture
async def connected_workspace(tmp_path, monkeypatch):
    """A clean cognee with the journeys tier's mock LLM and a Slack connection row."""
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"k" * 32).decode())
    for name, value in {
        "LLM_API_KEY": "mock-key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "openai/gpt-5-mini",
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "openai/text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "256",
        "EMBEDDING_API_KEY": "mock-key",
        "COGNEE_SKIP_PREFLIGHT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.tests.journeys import mock_ai as mocks

    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()
    mocks.install_all({})

    user = await get_default_user()
    await upsert_credential(
        provider="slack",
        user_id=user.id,
        provider_account_id=TEAM,
        token_payload={"access_token": "xoxb-test"},
        scopes=sync_module.REQUIRED_SCOPE,
        provider_metadata={
            "installed_by_slack_user_id": INSTALLER,
            sync_module.SELECTION_KEY: [PUBLIC, PRIVATE],
        },
    )
    yield
    mocks.uninstall_all()
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _workspace() -> FakeSlack:
    """A public and a private channel, three messages each on three days, the installer in
    both. Messages without replies are one document per day per channel (G7), so each
    channel holds three documents."""
    fake = FakeSlack()
    fake.add_user("U1", "ana")
    for channel, private in ((PUBLIC, False), (PRIVATE, True)):
        fake.add_channel(channel, channel.lower(), is_private=private)
        fake.members[channel] = ["U1", INSTALLER]
        for day in range(3):
            fake.post(
                f"{1793872860 + day * 86400}.000100",
                "U1",
                f"Note {day + 1} in {channel}.",
                channel=channel,
            )
    return fake


async def _sync(fake: FakeSlack) -> None:
    credential = await get_credential_by_account("slack", TEAM)
    with patch.object(slack_source_module, "build_slack_service", lambda token: fake):
        await sync_module.sync_slack(credential)


async def _documents_per_channel() -> dict[str, int]:
    credential = await get_credential_by_account("slack", TEAM)
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        [sync_module.dataset_name_for_credential(credential)], "read", user
    )
    channel_of = {
        ingestion.resource_name("slack", credential, channel): channel
        for channel in (PUBLIC, PRIVATE)
    }
    counts: dict[str, int] = {}
    for record in await get_dataset_data(datasets[0].id) if datasets else []:
        # Postgres staging names the table with its schema ("<schema>.<table>").
        channel = channel_of[record.system_metadata["table_name"].rpartition(".")[2]]
        counts[channel] = counts.get(channel, 0) + 1
    return counts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowlist",
    [[PUBLIC], ["general"]],
    ids=["commands_in_one_public_channel", "a_channel_name_instead_of_its_id"],
)
async def test_an_allowlist_never_removes_a_synced_channel(connected_workspace, allowlist):
    fake = _workspace()
    await _sync(fake)
    assert await _documents_per_channel() == {PUBLIC: 3, PRIVATE: 3}

    # The owner limits where @cognee and the shortcuts work. Even an id that names no
    # channel the sync reads (a name instead of an id) never removes a synced channel.
    await update_provider_metadata("slack", TEAM, {"allowed_channel_ids": allowlist})
    await _sync(fake)

    assert await _documents_per_channel() == {PUBLIC: 3, PRIVATE: 3}


async def _deselect_every_channel() -> None:
    """What ``PUT /integrations/slack/resources`` with an empty list does while a sync runs."""
    await update_provider_metadata("slack", TEAM, {sync_module.SELECTION_KEY: []})
    assert ingestion.request_rerun_if_running("slack", TEAM)


@pytest.mark.asyncio
async def test_channels_deselected_during_a_failing_sync_leave_memory(connected_workspace):
    """The owner deselects every channel while a sync runs, and that sync fails.

    The selection route asks the running sync to run once more; that run sees an empty
    selection and removes every channel, without a call to Slack.
    """
    fake = _workspace()
    await _sync(fake)
    assert await _documents_per_channel() == {PUBLIC: 3, PRIVATE: 3}

    loop = asyncio.get_running_loop()
    answer = fake.call

    def deselect_during_the_sync_then_fail(method, **params):
        if method == "auth.test":
            asyncio.run_coroutine_threadsafe(_deselect_every_channel(), loop).result()
            raise SlackUnavailableError("auth.test", "http_503")
        return answer(method, **params)

    fake.call = deselect_during_the_sync_then_fail
    with contextlib.suppress(SlackUnavailableError):
        await _sync(fake)

    assert await _documents_per_channel() == {}


async def _last_sync_counts() -> dict[str, int]:
    credential = await get_credential_by_account("slack", TEAM)
    return credential.provider_metadata["last_sync_counts"]


@pytest.mark.asyncio
async def test_a_channel_is_reported_retired_only_by_the_sync_that_removed_it(
    connected_workspace,
):
    """The installer leaves a selected private channel: the next sync removes it and says
    so, and the syncs after it, which find nothing left to remove, say nothing."""
    fake = _workspace()
    await _sync(fake)
    assert await _documents_per_channel() == {PUBLIC: 3, PRIVATE: 3}

    fake.members[PRIVATE] = ["U1"]
    await _sync(fake)
    assert await _documents_per_channel() == {PUBLIC: 3}
    assert (await _last_sync_counts()).get("channels_retired") == 1

    await _sync(fake)
    assert (await _last_sync_counts()).get("channels_retired", 0) == 0


def _files_holding(text: str, *roots) -> list[str]:
    """Every file under ``roots`` whose bytes, or gzip content, hold ``text``."""
    import gzip
    from pathlib import Path

    needle, found = text.encode(), []
    for root in roots:
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            data = path.read_bytes()
            if path.suffix == ".gz":
                data = gzip.decompress(data)
            if needle in data:
                found.append(str(path.relative_to(root)))
    return found


def _rows_holding(text: str, database) -> list[str]:
    """``table.column`` of every live row of a SQLite database holding ``text``."""
    import sqlite3

    found = []
    with sqlite3.connect(database) as connection:
        tables = [
            row[0]
            for row in connection.execute("select name from sqlite_master where type='table'")
        ]
        for table in tables:
            for column in [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]:
                query = f'select count(*) from "{table}" where cast("{column}" as text) like ?'
                if connection.execute(query, (f"%{text}%",)).fetchone()[0]:
                    found.append(f"{table}.{column}")
    return found


@pytest.mark.asyncio
async def test_the_uninstall_deletion_leaves_no_message_text_behind(connected_workspace, tmp_path):
    """Slack's Developer Policy asks for the data to be deleted (G28): the staging database,
    the dlt state and the completed load packages hold the messages' text too (G43)."""
    import json
    from datetime import datetime, timedelta, timezone

    from dlt.common.pipeline import get_dlt_pipelines_dir

    from cognee_connector_slack.cognee_host.handle_slack_event import handle_slack_event

    await _sync(_workspace())
    assert await _documents_per_channel() == {PUBLIC: 3, PRIVATE: 3}
    assert _files_holding("Note 1 in C1.", tmp_path, get_dlt_pipelines_dir())

    uninstall = {"type": "event_callback", "team_id": TEAM, "event": {"type": "app_uninstalled"}}
    await handle_slack_event(json.dumps(uninstall).encode())
    await sync_module.forget_uninstalled_workspaces(
        now=datetime.now(timezone.utc) + timedelta(days=14, minutes=1)
    )

    assert await _documents_per_channel() == {}
    # cognee's own relational database keeps no row with the text; SQLite leaves a
    # deleted row's bytes in free pages until VACUUM, as for every forget.
    relational = tmp_path / "system" / "databases" / "cognee_db"
    assert _rows_holding("Note 1 in C1.", relational) == []
    leftovers = _files_holding("Note 1 in C1.", tmp_path, get_dlt_pipelines_dir())
    assert [path for path in leftovers if path != str(relational.relative_to(tmp_path))] == []


@pytest.mark.asyncio
async def test_an_edited_reply_reaches_memory_with_the_next_sync(connected_workspace):
    """A reply edit leaves its root's fingerprint as it was; the message event marks the
    thread and the next sync reads it again (G6)."""
    import json

    from cognee_connector_slack.cognee_host.handle_slack_event import handle_slack_event

    root, reply = "1793872860.000100", "1793873100.000100"  # the first note, and a reply
    fake = _workspace()
    fake.reply(root, reply, "U1", "The owner is Ben.", channel=PUBLIC)
    await _sync(fake)
    assert "The owner is Ben." in await _content_of(PUBLIC, root)

    fake.edit(reply, "The owner is Carla.", edited_ts="1793873200.000100", channel=PUBLIC)
    edited = {
        "type": "message",
        "subtype": "message_changed",
        "channel": PUBLIC,
        "message": {
            "ts": reply,
            "thread_ts": root,
            "text": "The owner is Carla.",
            "edited": {"user": "U1", "ts": "1793873200.000100"},
        },
        "previous_message": {"ts": reply, "thread_ts": root, "text": "The owner is Ben."},
    }
    await handle_slack_event(
        json.dumps({"type": "event_callback", "team_id": TEAM, "event": edited}).encode()
    )
    await _sync(fake)

    content = await _content_of(PUBLIC, root)
    assert "The owner is Carla." in content and "The owner is Ben." not in content
    credential = await get_credential_by_account("slack", TEAM)
    assert not (credential.provider_metadata.get("changed_threads") or {}), "the mark is read"


async def _content_of(channel: str, root: str) -> str:
    """The staged text of one thread's document, through the dataset's data rows."""
    from cognee.infrastructure.files.storage import get_file_storage

    credential = await get_credential_by_account("slack", TEAM)
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        [sync_module.dataset_name_for_credential(credential)], "read", user
    )
    texts = []
    for record in await get_dataset_data(datasets[0].id):
        table = record.system_metadata.get("table_name", "").rpartition(".")[2]
        if table != ingestion.resource_name("slack", credential, channel):
            continue
        with open(record.raw_data_location.removeprefix("file://")) as handle:
            texts.append(handle.read())
    return "\n".join(texts)


@pytest.mark.asyncio
async def test_a_message_taken_out_leaves_memory_and_stays_out(connected_workspace):
    """ "Forget this" by the author or the owner (Goran, 2026-11-10): the message is gone
    from the workspace's memory after the next sync and no later sync brings it back,
    while it is still in Slack."""
    fake = _workspace()
    await _sync(fake)
    assert await _documents_per_channel() == {PUBLIC: 3, PRIVATE: 3}

    credential = await get_credential_by_account("slack", TEAM)
    note_ts = "1793872860.000100"  # "Note 1 in C1.", alone on its day
    with patch.object(sync_module, "start_sync", AsyncMock()):
        assert await sync_module.exclude_message(
            credential, channel_id=PUBLIC, message_ts=note_ts, root_ts=note_ts
        )
    await _sync(fake)

    assert await _documents_per_channel() == {PUBLIC: 2, PRIVATE: 3}
    assert "Note 1 in C1." not in await _content_of(PUBLIC, note_ts)
    await _sync(fake)
    assert "Note 1 in C1." not in await _content_of(PUBLIC, note_ts), "a later sync keeps it out"
