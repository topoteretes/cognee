"""Linear source through the real cognee.add() pipeline, with the API faked.

``remember()`` runs this same add() step first and then cognify(), which needs an
LLM, so this stops at the Data records like the Drive test. It checks what the
pod relies on: documents are tagged ``linear``, an unchanged re-sync creates no
Data ids and no new rows, a comment-only edit replaces one document, and the
token never lands in stored documents.
"""

import pytest
from fake_linear import FakeLinear

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import linear_source

DATASET_NAME = "linear_integration_test"
TOKEN = "lin_oauth_SECRET_TOKEN_VALUE"
TABLE = "linear_team_one"


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
async def test_trashing_an_issue_or_project_forgets_its_document(clean_environment):
    team = _team()
    await _sync(team)
    before = {d.system_metadata["external_id"]: d.id for d in await _linear_data()}

    team.trash(team.issues, "i2", 30)  # updatedAt does not move, as in Linear
    team.trash(team.projects, "p1", 30)
    source = await _sync(team)

    assert source.cognee_sync_stats["deleted"] == 2
    after = {d.system_metadata["external_id"]: d.id for d in await _linear_data()}
    assert after == {"issue:i1": before["issue:i1"]}


@pytest.mark.asyncio
async def test_an_issue_already_trashed_on_the_first_sync_is_never_stored(clean_environment):
    team = _team()
    team.add_issue("i3", 5)
    team.trash(team.issues, "i3", 6)
    source = await _sync(team)

    assert source.cognee_sync_stats["failed"] == 0
    assert {d.system_metadata["external_id"] for d in await _linear_data()} == {
        "issue:i1",
        "issue:i2",
        "project:p1",
    }


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
