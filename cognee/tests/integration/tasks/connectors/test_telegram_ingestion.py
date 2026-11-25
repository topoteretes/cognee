# Adapted from topoteretes/cognee-community (Apache-2.0), PR #355, commit 76ff1a4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Telegram rows become tagged documents with their chat's node set and container.

The source runs through the real cognee.add() pipeline with the Bot API faked, and stops at the
Data records. A quiet second sync stores nothing new, an upgraded group keeps one node set, and
the bot token lands in no stored document.
"""

import subprocess
import sys

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import telegram_source
from cognee.tasks.ingestion.exceptions.exceptions import DLTIngestionError
from cognee.tests.unit.tasks.connectors.fake_telegram import (
    CHANNEL,
    GROUP,
    SUPERGROUP,
    FakeBot,
    msg,
    removed,
)

DATASET = "telegram_integration_test"
NAME = "telegram_e8ba1b716982492b069485f8"


async def _sync(bot: FakeBot, **kwargs):
    source = telegram_source(service=bot, resource_name=NAME, **kwargs)
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


async def _documents() -> dict[str, Data]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        data.system_metadata["external_id"]: data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
        and data.system_metadata.get("source") == "telegram"
    }


def _node_set(document: Data) -> list[str] | None:
    return (document.external_metadata or {}).get("node_set")


@pytest.mark.asyncio
async def test_messages_are_stored_with_their_chat_node_set_and_container(clean_environment):
    bot = FakeBot()
    bot.add(message=msg(GROUP, 5, "before upgrade"))
    bot.add(message=msg(GROUP, 6, None, migrate_to_chat_id=-1009))
    bot.add(message=msg(SUPERGROUP, 1, "after upgrade", migrate_from_chat_id=-100))
    bot.add(channel_post=msg(CHANNEL, 3, "Release notes", sender=None))

    source = await _sync(bot)
    documents = await _documents()

    assert set(documents) == {"-100:5", "-1009:1", "-1001:3"}
    assert {key: _node_set(doc) for key, doc in documents.items()} == {
        "-100:5": ["telegram:chat:-100"],
        "-1009:1": ["telegram:chat:-100"],
        "-1001:3": ["telegram:chat:-1001"],
    }
    assert documents["-1001:3"].system_metadata["structure"] == {
        "ancestors": [{"kind": "chat", "id": "-1001", "name": "News"}]
    }
    assert {doc.system_metadata["table_name"].rpartition(".")[2] for doc in documents.values()} == {
        NAME
    }
    assert source.cognee_sync_stats["scanned"] == 3 and source.cognee_sync_stats["skipped"] == 1
    # The upgrade announcement has no text: a service message.
    assert source.cognee_sync_stats["skipped_empty_content"] == 1

    ids = {key: doc.id for key, doc in documents.items()}
    quiet = await _sync(bot)
    assert {key: doc.id for key, doc in (await _documents()).items()} == ids
    assert quiet.cognee_sync_stats["scanned"] == 0


@pytest.mark.asyncio
async def test_a_removal_on_the_first_sync_finds_no_staging_table(clean_environment):
    """The first sync reads the staging table before any load created it (on Postgres the
    staging schema is missing too): nothing is stored, so only this page's rows are forgotten."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "gone with the bot"))
    bot.add(message=msg(CHANNEL, 2, "stays", sender=None))
    bot.add(my_chat_member=removed(GROUP))

    source = await _sync(bot)

    assert set(await _documents()) == {"-1001:2"}
    stats = source.cognee_sync_stats
    assert (stats["scanned"], stats["deleted"], stats["failed"]) == (1, 1, 0)


# Another process writing to SQLite files: dlt keeps staging in WAL mode, so a reader still
# gets through and only a write (the load) waits and then fails.
_HOLD_SQLITE_WRITE_LOCK = """
import sqlite3, sys, time
held = [sqlite3.connect(f, isolation_level=None) for f in sys.argv[1:]]
for c in held:
    c.execute("BEGIN EXCLUSIVE")
print("locked", flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SQLite locks files with LockFileEx on Windows, and an open file cannot be deleted "
    "there, so the holder process and tmp_path cleanup behave differently; this lock sequence "
    "was verified on macOS only",
)
@pytest.mark.asyncio
async def test_a_failed_load_is_loaded_by_the_next_sync_and_its_tombstone_applied(
    clean_environment, tmp_path
):
    """The run's rows and its cursor go to dlt together: when the load fails after the source
    emitted a removal, dlt keeps the package and loads it first on the next sync, so the
    removal is not lost although Telegram already counts the update as delivered."""
    from cognee.infrastructure.databases.relational import get_relational_config

    if get_relational_config().db_provider != "sqlite":
        pytest.skip("the failed load is made by locking SQLite staging files")
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "goes with the bot"))
    bot.add(channel_post=msg(CHANNEL, 2, "stays", sender=None))
    await _sync(bot)
    assert set(await _documents()) == {"-100:1", "-1001:2"}

    files = [
        str(path)
        for path in tmp_path.rglob("dlt_database_telegram_integration_test*")
        if path.is_file() and not path.name.endswith(("-wal", "-shm", "-journal"))
    ]
    assert files, "the SQLite staging files"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_SQLITE_WRITE_LOCK, *files], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        bot.add(my_chat_member=removed(GROUP))
        with pytest.raises(DLTIngestionError):
            await _sync(bot)
    finally:
        holder.kill()
        holder.wait()
    assert set(await _documents()) == {"-100:1", "-1001:2"}, "nothing applied yet"
    reads = len(bot.get_updates_calls())

    await _sync(bot)

    assert set(await _documents()) == {"-1001:2"}, "the kept package carried the tombstone"
    assert len(bot.get_updates_calls()) == reads, "dlt loads the kept package and reads nothing"
    await _sync(bot)
    assert bot.get_updates_calls()[-1]["offset"] == 1003, "the removal's cursor was kept"
    assert set(await _documents()) == {"-1001:2"}


@pytest.mark.asyncio
async def test_the_bot_token_is_in_no_stored_document(clean_environment):
    token = "111:" + "S" * 35
    bot = FakeBot(bot_id=111)
    bot.add(message=msg(GROUP, 1, "hello"))
    resource = telegram_source(token=token)
    # The real client is replaced by the fake after the name was derived from the token.
    resource = telegram_source(service=bot, resource_name=resource.name)
    await cognee.add(
        resource,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    documents = await _documents()
    assert set(documents) == {"-100:1"}
    for document in documents.values():
        with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
            assert "S" * 20 not in stored.read()
        assert "S" * 20 not in str(document.system_metadata)
