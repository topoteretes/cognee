# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK: one source per folder id, the httpx client over a fake.
"""Dropbox rows through the real cognee.add(): what a Data record holds and what it never holds.

Copied from cognee.tests.integration.tasks.connectors.test_dropbox_ingestion (cognee dev, 2026-11-12).

remember() runs this same add() step before cognify(), which needs an LLM, so this stops at the
Data records, like the OneNote and Linear tests. ``isolated_connector_state`` (conftest) keeps
the dlt state and the databases in the test's temporary directory.
"""

import cognee
import pytest
import pytest_asyncio
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from fake_dropbox import ACCOUNT, TOKEN, FakeDropbox

from cognee_connector_dropbox import dropbox_source

DATASET = "dropbox_integration_test"


@pytest_asyncio.fixture
async def clean(tmp_path):
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _add(fake, folder_id, name="dropbox_notes", account=ACCOUNT):
    source = dropbox_source(
        folder_id=folder_id,
        account_id=account,
        resource_name=name,
        service=fake.client(lambda: TOKEN, sleep=lambda _s: None),
    )
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


async def _records():
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return list(await get_dataset_data(datasets[0].id)) if datasets else []


@pytest.mark.asyncio
async def test_records_carry_the_file_name_the_path_line_and_no_token(clean):
    fake = FakeDropbox()
    notes = fake.mkdir("/Notes")
    a = fake.put("/Notes/2026/plan.md", "Ship it.")

    source = await _add(fake, notes)

    (record,) = await _records()
    assert record.system_metadata["source"] == "dropbox"
    assert record.system_metadata["external_id"] == a
    assert record.system_metadata["table_name"] == "dropbox_notes"
    assert record.label == "plan.md"
    ancestors = record.system_metadata["structure"]["ancestors"]
    assert [entry["name"] for entry in ancestors] == ["2026", "Notes"]
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(record.raw_data_location, mode="r", encoding="utf-8") as file:
        text = file.read()
    assert "Ship it." in text and "/Notes" not in text
    assert TOKEN not in text and TOKEN not in str(record.system_metadata)
    assert source.cognee_sync_stats["complete"] == 1


@pytest.mark.asyncio
async def test_an_unchanged_sync_downloads_nothing_and_keeps_the_records(clean):
    fake = FakeDropbox()
    notes = fake.mkdir("/Notes")
    fake.put("/Notes/a.md", "alpha")
    await _add(fake, notes)
    before = {r.id for r in await _records()}
    fake.downloads.clear()

    source = await _add(fake, notes)

    assert fake.downloads == [] and source.cognee_sync_stats["scanned"] == 0
    assert {r.id for r in await _records()} == before


@pytest.mark.asyncio
async def test_another_account_on_the_same_resource_name_is_refused(clean):
    fake = FakeDropbox()
    notes = fake.mkdir("/Notes")
    fake.put("/Notes/a.md", "alpha")
    await _add(fake, notes)
    before = {r.id for r in await _records()}

    with pytest.raises(Exception, match="another Dropbox account"):
        await _add(fake, notes, account="dbid:someone-else")

    assert {r.id for r in await _records()} == before
