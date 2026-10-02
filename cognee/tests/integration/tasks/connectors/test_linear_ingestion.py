"""Linear source through the real cognee.add() pipeline, with the API faked.

``remember()`` runs this same add() step first and then cognify(), which needs an
LLM, so this stops at the Data records like the Drive test. It checks what the
pod relies on: documents are tagged ``linear``, an unchanged re-sync creates no
Data ids and no new rows, a comment-only edit replaces one document, and the
token never lands in stored documents.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fake_linear import FakeLinear

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import linear as linear_module
from cognee.tasks.ingestion.connectors import linear_source

DATASET_NAME = "linear_integration_test"
TOKEN = "lin_oauth_SECRET_TOKEN_VALUE"
TABLE = "linear_team_one"


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    Container()[PipelineContext].deactivate()
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    monkeypatch.setenv("PIPELINES_DIR", str(tmp_path / "dlt" / "pipelines"))

    from cognee.context_global_variables import graph_db_config, vector_db_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    get_dlt_destination.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    # The fake's timestamps are on 2026-10-01; the source seeds its comment floor
    # from the wall clock, so pin it just before them.
    start = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(
        linear_module, "time", SimpleNamespace(time=lambda: start, sleep=lambda _: None)
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    Container()[PipelineContext].deactivate()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _linear_data():
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET_NAME]
    )
    if not datasets:
        return []
    return [
        data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "linear"
    ]


async def _sync(team: FakeLinear):
    source = linear_source(team_id="team-1", service=team, resource_name=TABLE)
    await cognee.add(
        source,
        dataset_name=DATASET_NAME,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


def _team() -> FakeLinear:
    team = FakeLinear()
    team.add_issue("i1", 1)
    team.add_issue("i2", 2)
    team.add_comment("c1", "i1", 3, body="Alice owns the rollout.")
    team.add_project("p1", 4)
    return team


@pytest.mark.asyncio
async def test_first_sync_stores_tagged_documents_and_the_second_creates_nothing(
    clean_environment,
):
    team = _team()
    source = await _sync(team)
    assert source.cognee_sync_stats["failed"] == 0

    first = await _linear_data()
    assert {d.system_metadata["external_id"] for d in first} == {
        "issue:i1",
        "issue:i2",
        "project:p1",
    }
    assert {d.system_metadata["table_name"] for d in first} == {TABLE}

    again = await _sync(team)
    assert again.cognee_sync_stats["scanned"] == 0
    assert {d.id for d in await _linear_data()} == {d.id for d in first}


@pytest.mark.asyncio
async def test_a_comment_only_edit_replaces_just_that_issue(clean_environment):
    team = _team()
    await _sync(team)
    before = {d.system_metadata["external_id"]: d.id for d in await _linear_data()}

    team.add_comment("c2", "i2", 20, body="Bob found a blocker.")  # i2.updatedAt unchanged
    await _sync(team)
    after = {d.system_metadata["external_id"]: d.id for d in await _linear_data()}

    assert set(after) == set(before)
    assert after["issue:i2"] != before["issue:i2"]
    assert after["issue:i1"] == before["issue:i1"]
    assert after["project:p1"] == before["project:p1"]


@pytest.mark.asyncio
async def test_a_rate_limited_first_sync_resumes_without_losing_documents(clean_environment):
    from cognee.tasks.ingestion.connectors.linear import LinearRateLimitedError

    team = _team()
    team.fail_after = 1
    team.fail_with = LinearRateLimitedError()
    cut = await _sync(team)
    assert cut.cognee_sync_stats["failed_rate_limit"] == 1

    team.fail_after = None
    done = await _sync(team)
    assert done.cognee_sync_stats["failed"] == 0
    assert {d.system_metadata["external_id"] for d in await _linear_data()} == {
        "issue:i1",
        "issue:i2",
        "project:p1",
    }


@pytest.mark.asyncio
async def test_the_token_is_not_in_any_stored_document(clean_environment):
    team = _team()
    source = linear_source(team_id="team-1", service=team, resource_name=TABLE)
    assert TOKEN not in repr(source)
    await _sync(team)
    for data in await _linear_data():
        assert TOKEN not in repr(data.system_metadata)
