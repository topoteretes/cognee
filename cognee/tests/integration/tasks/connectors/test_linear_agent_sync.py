"""The SDK Linear integration's sync through the real add() pipeline, Linear API faked.

``remember()`` would also cognify, which needs an LLM, so the ingestion step of
``sync_scopes`` is replaced by ``cognee.add()``, the same step ``remember()``
starts with. The rest is real: the DLT source, the dataset, the legacy cleanup
through ``forget()``, and the sync bookkeeping.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlparse

import pytest
import pytest_asyncio
from fake_linear import FakeLinear

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup
from cognee.modules.integrations import ingestion
from cognee.modules.integrations.linear import sync as sync_module
from cognee.modules.users.methods import get_default_user

DATASET = "linear_acme"
OLD_ISSUE = "Linear issue COG-9: Old title\nURL: https://linear.app/x\nState: Todo"
BY_HAND = "A note the user remembered into this dataset themselves"


async def _documents():
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return []
    return await get_dataset_data(datasets[0].id)


@pytest_asyncio.fixture
async def linear(clean_environment, monkeypatch):
    await setup()
    team = FakeLinear()
    team.add_issue("i1", 1)
    team.add_issue("i2", 2)
    team.add_comment("c1", "i1", 3, body="Alice owns the rollout.")
    team.add_project("p1", 4)

    user = await get_default_user()
    credential = SimpleNamespace(
        status="active",
        provider="linear",
        provider_account_id="org-1",
        provider_metadata={"organization_url_key": "acme"},
        user_id=user.id,
    )

    async def add_instead_of_remember(source, **kwargs):
        await cognee.add(
            source,
            dataset_name=kwargs["dataset_name"],
            user=kwargs["user"],
            write_disposition=kwargs["write_disposition"],
            primary_key=kwargs["primary_key"],
            max_rows_per_table=kwargs["max_rows_per_table"],
        )
        return SimpleNamespace(status="completed")

    import importlib

    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.remember"),
        "remember",
        add_instead_of_remember,
    )
    monkeypatch.setattr(ingestion, "require_active_credential", AsyncMock(return_value=credential))
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", AsyncMock())
    monkeypatch.setattr(
        "cognee.modules.integrations.credentials.update_provider_metadata", AsyncMock()
    )
    monkeypatch.setattr(sync_module, "list_teams", AsyncMock(return_value=[{"id": "team-1"}]))
    monkeypatch.setattr(sync_module, "_LinearService", lambda cred, loop: team)
    return SimpleNamespace(team=team, credential=credential)


@pytest.mark.asyncio
async def test_the_seed_replaces_the_old_text_documents_and_keeps_the_users_own(linear):
    await cognee.add([OLD_ISSUE, BY_HAND], dataset_name=DATASET)
    assert len(await _documents()) == 2

    await sync_module.sync_linear(linear.credential)

    documents = await _documents()
    tagged = {d.system_metadata["external_id"] for d in documents if d.system_metadata}
    assert tagged == {"issue:i1", "issue:i2", "project:p1"}
    untagged = [d for d in documents if not d.system_metadata]
    assert (
        len(untagged) == 1
        and "remembered" in Path(urlparse(untagged[0].raw_data_location).path).read_text()
    )


@pytest.mark.asyncio
async def test_a_second_sync_adds_nothing(linear):
    await sync_module.sync_linear(linear.credential)
    first = {d.id for d in await _documents()}

    await sync_module.sync_linear(linear.credential)

    assert {d.id for d in await _documents()} == first


@pytest.mark.asyncio
async def test_a_comment_edit_replaces_only_its_issue_after_the_seed(linear):
    await sync_module.sync_linear(linear.credential)
    before = {d.system_metadata["external_id"]: d.id for d in await _documents()}

    linear.team.add_comment("c2", "i2", 20, body="Bob found a blocker.")
    await sync_module.sync_linear(linear.credential, ["team-1"])

    after = {d.system_metadata["external_id"]: d.id for d in await _documents()}
    assert after["issue:i2"] != before["issue:i2"]
    assert after["issue:i1"] == before["issue:i1"]
    assert after["project:p1"] == before["project:p1"]
