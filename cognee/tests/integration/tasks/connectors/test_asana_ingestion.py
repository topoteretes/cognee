# Adapted from topoteretes/cognee-community (Apache-2.0), PR #349, commit ef32699 (AbdulWasih05).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A stored Asana document is replaced only when its rendered text changes.

The source runs through the real cognee.add() pipeline with the API faked. ``remember()`` runs this
same add() step first and then cognify(), which needs an LLM, so this stops at the Data records
like the Linear test. A task deleted in Asana leaves the dataset, a task that cannot be read keeps
its document, a document's node set is its project gids, and the access token lands in no
stored document.
"""

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import asana_source
from cognee.tests.unit.tasks.connectors.fake_asana import FakeAsana, serve_over_http

DATASET = "asana_integration_test"
TABLE = "asana_test"
TOKEN = "SECRET-asana-token-123"


def _workspace(tasks: int = 3) -> FakeAsana:
    fake = FakeAsana()
    fake.add_project("P1")
    for gid in range(1, tasks + 1):
        fake.add_task(str(gid), f"Task {gid}", notes=f"Body {gid}.")
    return fake


async def _add(source) -> None:
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _sync(fake: FakeAsana, resource_name: str = TABLE, **kwargs):
    kwargs.setdefault("project_gids", ["P1"])
    source = asana_source(service=fake, resource_name=resource_name, **kwargs)
    await _add(source)
    return source


async def _rows() -> list[Data]:
    """Every Asana Data row of the dataset, with no collapsing of two rows for one record."""
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return []
    return [
        data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "asana"
    ]


async def _documents() -> dict[str, Data]:
    return {str(data.system_metadata["external_id"]): data for data in await _rows()}


async def _ids() -> dict[str, object]:
    return {key: doc.id for key, doc in (await _documents()).items()}


def _node_set(document: Data) -> list[str] | None:
    return (document.external_metadata or {}).get("node_set")


def _stored_text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_tagged_documents_with_their_project_node_set_and_a_quiet_one_changes_nothing(
    clean_environment,
):
    fake = _workspace()
    await _sync(fake)
    first = await _documents()

    assert set(first) == {"project:P1", "task:1", "task:2", "task:3"}
    assert {doc.system_metadata["table_name"] for doc in first.values()} == {TABLE}
    # Namespaced once: a second "asana:" prefix would mean core stopped seeing the name as
    # already namespaced.
    assert {key: _node_set(doc) for key, doc in first.items()} == {
        key: ["asana:P1"] for key in ("project:P1", "task:1", "task:2", "task:3")
    }
    assert "Body 1." in _stored_text(first["task:1"])

    before = await _ids()
    quiet = await _sync(fake)
    assert quiet.cognee_sync_stats["skipped"] == quiet.cognee_sync_stats["scanned"] == 3
    assert await _ids() == before


@pytest.mark.asyncio
async def test_an_edit_replaces_only_that_tasks_document(clean_environment):
    fake = _workspace()
    await _sync(fake)
    before = await _ids()

    fake.edit_task("2", notes="Rewritten.")
    await _sync(fake)
    after = await _ids()

    assert set(after) == set(before)
    assert after["task:2"] != before["task:2"]
    assert {k: v for k, v in after.items() if k != "task:2"} == {
        k: v for k, v in before.items() if k != "task:2"
    }
    assert "Rewritten." in _stored_text((await _documents())["task:2"])


@pytest.mark.asyncio
async def test_a_comment_edit_replaces_the_document_although_modified_at_did_not_move(
    clean_environment,
):
    fake = _workspace(tasks=2)
    fake.add_comment("1", "Grace", "first")
    await _sync(fake)
    before = await _ids()
    stamp = fake.tasks["1"]["modified_at"]

    fake.edit_comment("1", 0, "edited")
    await _sync(fake)
    after = await _ids()

    assert fake.tasks["1"]["modified_at"] == stamp
    assert after["task:1"] != before["task:1"] and after["task:2"] == before["task:2"]
    assert "- Grace: edited" in _stored_text((await _documents())["task:1"])


@pytest.mark.asyncio
async def test_a_task_deleted_in_asana_is_forgotten(clean_environment):
    fake = _workspace()
    await _sync(fake)

    fake.delete_task("2")
    await _sync(fake)

    assert set(await _documents()) == {"project:P1", "task:1", "task:3"}


@pytest.mark.asyncio
async def test_a_task_that_cannot_be_read_keeps_its_document_until_it_leaves(clean_environment):
    fake = _workspace()
    await _sync(fake)
    before = await _ids()
    fake.edit_task("2", notes="Rewritten.")
    fake.fail("/tasks/2", 403)

    for _ in range(3):
        stopped = await _sync(fake)
        assert stopped.cognee_sync_stats["failed"] == 1
        assert await _ids() == before, "the old document stays, untouched"

    fake.heal()
    await _sync(fake)
    assert "Rewritten." in _stored_text((await _documents())["task:2"])

    fake.edit_task("2")
    fake.fail("/tasks/2", 404)
    await _sync(fake)
    fake.delete_task("2")
    await _sync(fake)
    assert "task:2" not in await _documents()


@pytest.mark.asyncio
async def test_a_rate_limited_first_sync_resumes_without_losing_documents(clean_environment):
    fake = _workspace(tasks=4)
    fake.fail("/tasks/3", 429)
    cut = await _sync(fake)
    assert cut.cognee_sync_stats["failed_rate_limit"] == 1
    assert set(await _documents()) == {"project:P1", "task:1", "task:2"}

    fake.heal()
    done = await _sync(fake)
    assert done.cognee_sync_stats["failed"] == 0
    assert set(await _documents()) == {"project:P1", "task:1", "task:2", "task:3", "task:4"}


@pytest.mark.asyncio
async def test_the_real_client_syncs_and_the_token_never_reaches_stored_documents(
    clean_environment, monkeypatch
):
    fake, authorizations = _workspace(), []
    monkeypatch.setattr("requests.request", serve_over_http(fake, authorizations=authorizations))
    source = asana_source(access_token=TOKEN, resource_name=TABLE, project_gids=["P1"])
    assert TOKEN not in repr(source)

    await _add(source)
    fake.delete_task("2")
    await _add(asana_source(access_token=TOKEN, resource_name=TABLE, project_gids=["P1"]))

    documents = await _documents()
    assert set(documents) == {"project:P1", "task:1", "task:3"}
    assert authorizations and set(authorizations) == {f"Bearer {TOKEN}"}
    for doc in documents.values():
        assert TOKEN not in repr(doc)
        assert TOKEN not in _stored_text(doc)


@pytest.mark.asyncio
async def test_deselecting_a_project_rewrites_the_shared_task_and_forgets_the_rest(
    clean_environment,
):
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", projects=("P1", "P2"))
    fake.add_task("2", projects=("P2",))
    await _sync(fake, project_gids=["P1", "P2"])
    before = await _ids()
    assert "Project: Blog" in _stored_text((await _documents())["task:1"])

    await _sync(fake, project_gids=["P1"])
    after = await _ids()

    assert set(after) == {"project:P1", "task:1"}
    assert after["task:1"] != before["task:1"]
    assert "Blog" not in _stored_text((await _documents())["task:1"])


@pytest.mark.asyncio
async def test_a_task_in_two_projects_stores_both_node_sets_and_deselecting_one_changes_it(
    clean_environment,
):
    fake = _workspace(tasks=0)
    fake.add_project("P2", "Blog")
    fake.add_task("1", "In both", projects=("P2", "P1"))
    fake.add_task("2", "Only in P1", projects=("P1",))
    fake.add_task("3", "Only in P2", projects=("P2",))

    await _sync(fake, project_gids=["P1", "P2"])
    both = await _documents()

    assert {key: _node_set(doc) for key, doc in both.items()} == {
        "project:P1": ["asana:P1"],
        "project:P2": ["asana:P2"],
        "task:1": ["asana:P1", "asana:P2"],
        "task:2": ["asana:P1"],
        "task:3": ["asana:P2"],
    }

    await _sync(fake, project_gids=["P1"])
    only_p1 = await _documents()

    assert {key: _node_set(doc) for key, doc in only_p1.items()} == {
        "project:P1": ["asana:P1"],
        "task:1": ["asana:P1"],
        "task:2": ["asana:P1"],
    }
    assert only_p1["task:1"].id != both["task:1"].id
    assert only_p1["task:2"].id == both["task:2"].id


@pytest.mark.asyncio
async def test_two_asana_connections_in_one_dataset_stay_apart(clean_environment):
    team_a, team_b = _workspace(), FakeAsana()
    team_b.add_project("P1", "Other")
    team_b.add_task("1", "Elsewhere")
    team_b.add_task("2", "Elsewhere too")

    await _sync(team_a, "asana_a")
    await _sync(team_b, "asana_b")
    # the two connections share row ids ("task:1"): each keeps its own table, so each keeps its own
    assert len(await _rows()) == 4 + 3

    team_b.delete_task("2")
    await _sync(team_b, "asana_b")
    await _sync(team_a, "asana_a")

    assert len(await _rows()) == 4 + 2
    assert {d.system_metadata["table_name"] for d in await _rows()} == {"asana_a", "asana_b"}
