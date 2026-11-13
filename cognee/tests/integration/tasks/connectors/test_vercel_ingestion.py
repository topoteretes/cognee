# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A stored Vercel document is replaced only when its rendered text changes, and forgotten only
on Vercel's word.

The source runs through the real ``cognee.add()`` pipeline with the API faked; ``remember()``
runs this same add() step first. Covered here: the documents a first sync stores and what they
carry, a quiet re-sync, an edit and a rename, the deletions Vercel confirms and the ones it does
not, the cuts that keep documents, the account guard, and that no token, query string, secret
or email lands on disk, in the staging copy or in dlt's trace after a failed run.
"""

from types import SimpleNamespace

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import vercel as vercel_module
from cognee.tasks.ingestion.connectors import vercel_source
from cognee.tests.unit.tasks.connectors.fake_vercel import (
    CANARIES,
    CANARY_EMAIL,
    DAY_MS,
    NOW_MS,
    FakeVercel,
    serve_over_http,
)

DATASET = "vercel_integration_test"
TABLE = "vercel_web_test"
TOKEN = "SECRET-vercel-token-123"


def _team() -> FakeVercel:
    fake = FakeVercel()
    fake.add_project("prj_web", "web")
    fake.add_deployment("dpl_ready", minutes_ago=30, commit=1)
    fake.add_deployment(
        "dpl_fail",
        state="ERROR",
        minutes_ago=10,
        commit=2,
        errorCode="BUILD_FAILED",
        errorMessage="Command npm run build exited with 1",
    )
    fake.set_log("dpl_fail", "Error: Cannot find module 'x'")
    return fake


async def _add(source) -> None:
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _sync(fake, resource_name: str = TABLE, **kwargs):
    kwargs.setdefault("project_id", "prj_web")
    kwargs.setdefault("team_id", "team_acme")
    if "access_token" not in kwargs:
        kwargs["service"] = fake
    source = vercel_source(resource_name=resource_name, **kwargs)
    await _add(source)
    return source


async def _rows() -> list[Data]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return []
    return [
        data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "vercel"
    ]


async def _documents() -> dict[str, Data]:
    return {str(data.system_metadata["external_id"]): data for data in await _rows()}


async def _ids() -> dict[str, object]:
    return {key: doc.id for key, doc in (await _documents()).items()}


def _stored_text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_one_document_per_record_with_its_kind_and_place(
    clean_environment,
):
    await _sync(_team())
    documents = await _documents()

    assert set(documents) == {"project:prj_web", "deployment:dpl_ready", "deployment:dpl_fail"}
    # On Postgres the name is schema-qualified (``<dataset>.<table>``); the table is the same.
    tables = {doc.system_metadata["table_name"].rpartition(".")[2] for doc in documents.values()}
    assert tables == {TABLE}
    assert {key: doc.external_metadata["node_set"] for key, doc in documents.items()} == {
        "project:prj_web": ["vercel:project"],
        "deployment:dpl_ready": ["vercel:deployment"],
        "deployment:dpl_fail": ["vercel:deployment"],
    }
    assert documents["deployment:dpl_fail"].system_metadata["structure"] == {
        "ancestors": [{"kind": "project", "id": "project:prj_web", "name": "web", "document": True}]
    }
    assert documents["project:prj_web"].system_metadata["structure"] == {"ancestors": []}
    failed = _stored_text(documents["deployment:dpl_fail"])
    assert failed.startswith("# web: production deployment 2026-11-10 11:50 UTC (dpl_fail)")
    for fact in (
        "State: ERROR",
        "Error code: BUILD_FAILED",
        "Commit message: Commit 2",
        "Commit author: Ada Lovelace (ada)",
        f"Commit: {'2'.rjust(40, '0')}",
    ):
        assert fact in failed
    assert documents["deployment:dpl_fail"].system_metadata["url"] == (
        "https://vercel.com/acme/web/dpl_fail"
    )


@pytest.mark.asyncio
async def test_a_quiet_sync_changes_nothing(clean_environment):
    fake = _team()
    await _sync(fake)
    before = await _ids()

    quiet = await _sync(fake)

    assert quiet.cognee_sync_stats["skipped"] == quiet.cognee_sync_stats["scanned"] == 3
    assert await _ids() == before


@pytest.mark.asyncio
async def test_a_state_change_replaces_only_that_document_and_a_rename_only_the_project(
    clean_environment,
):
    fake = _team()
    fake.set_state("dpl_ready", "BUILDING")
    await _sync(fake)
    before = await _ids()

    fake.set_state("dpl_ready", "READY")
    fake.rename_project("prj_web", "website")
    await _sync(fake)
    after = await _ids()

    assert set(after) == set(before)
    assert after["deployment:dpl_fail"] == before["deployment:dpl_fail"]
    assert after["deployment:dpl_ready"] != before["deployment:dpl_ready"]
    assert after["project:prj_web"] != before["project:prj_web"]
    assert "State: READY" in _stored_text((await _documents())["deployment:dpl_ready"])


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["listed_deleted", "read_404", "read_410"])
async def test_a_deployment_vercel_says_is_gone_leaves_the_dataset(clean_environment, how):
    fake = _team()
    await _sync(fake)

    if how == "listed_deleted":
        fake.set_state("dpl_ready", "DELETED")
    elif how == "read_404":
        fake.delete("dpl_ready")
    else:
        fake.expire("dpl_ready")
    source = await _sync(fake)

    assert set(await _documents()) == {"project:prj_web", "deployment:dpl_fail"}
    assert source.cognee_sync_stats["deleted"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["full", "minimal"])
async def test_a_live_deployment_the_listing_missed_keeps_its_document_and_the_run_succeeds(
    clean_environment, variant
):
    """Its own read answers in the /v13 shape (``id``, ``createdAt``, maybe no ``projectId``);
    read with the list's shape it raised on every run and nothing else landed."""
    fake = _team()
    await _sync(fake)
    before = await _ids()
    fake.unlisted.add("dpl_ready")
    if variant == "minimal":
        fake.minimal_reads.add("dpl_ready")
    fake.set_state("dpl_fail", "CANCELED")

    source = await _sync(fake)

    after = await _ids()
    assert source.cognee_sync_stats["failed"] == 0
    assert after["deployment:dpl_ready"] == before["deployment:dpl_ready"]
    assert after["deployment:dpl_fail"] != before["deployment:dpl_fail"]


@pytest.mark.asyncio
async def test_a_deployment_that_left_the_window_keeps_its_document(clean_environment, monkeypatch):
    # The module's own clock only: time.time stays real for the rest of the process.
    monkeypatch.setattr(vercel_module, "time", SimpleNamespace(time=lambda: NOW_MS / 1000))
    fake = _team()
    fake.add_deployment("dpl_old", minutes_ago=20 * 24 * 60)
    await _sync(fake, lookback_days=30)

    later = (NOW_MS + 15 * DAY_MS) / 1000
    monkeypatch.setattr(vercel_module, "time", SimpleNamespace(time=lambda: later))
    fake.delete("dpl_old")
    await _sync(fake, lookback_days=30)

    assert "deployment:dpl_old" in await _documents()


@pytest.mark.asyncio
@pytest.mark.parametrize("trouble", ["project_404", "rate_limit", "auth_after_rows", "account"])
async def test_a_run_that_cannot_confirm_a_deletion_forgets_nothing(clean_environment, trouble):
    fake = _team()
    await _sync(fake)
    before = await _ids()
    fake.delete("dpl_ready")
    fake.set_state("dpl_fail", "CANCELED")
    if trouble == "project_404":
        del fake.projects["prj_web"]
    elif trouble == "rate_limit":
        fake.quota = 2  # the project and the listing; the check of dpl_ready answers 429
    elif trouble == "auth_after_rows":
        fake.fail("/v13/deployments", 401)
    else:
        fake.projects["prj_web"]["accountId"] = "team_other"

    source = await _sync(fake)
    after = await _ids()

    assert "deployment:dpl_ready" in after
    assert source.cognee_sync_stats["deleted"] == 0
    if trouble in ("rate_limit", "auth_after_rows"):
        assert after["deployment:dpl_fail"] != before["deployment:dpl_fail"], "rows are kept"
    else:
        assert after == before, "nothing was read, nothing was written"


@pytest.mark.asyncio
async def test_two_projects_in_one_dataset_keep_their_documents_apart(clean_environment):
    fake = _team()
    fake.add_project("prj_api", "api")
    fake.add_deployment("dpl_api", "prj_api")
    await _sync(fake, "vercel_web_test")
    await _sync(fake, "vercel_api_test", project_id="prj_api")
    fake.delete("dpl_api")

    await _sync(fake, "vercel_web_test")

    assert set(await _documents()) == {
        "project:prj_web",
        "deployment:dpl_ready",
        "deployment:dpl_fail",
        "project:prj_api",
        "deployment:dpl_api",
    }, "a sync of web confirms nothing about api"


@pytest.mark.asyncio
async def test_no_token_secret_or_email_reaches_a_document_or_the_disk(
    clean_environment, monkeypatch, tmp_path
):
    fake = _team()
    serve_over_http(monkeypatch, fake, TOKEN)
    await _sync(fake, access_token=TOKEN, include_build_logs=True)

    documents = await _documents()
    assert "build_log:dpl_fail" in documents
    for document in documents.values():
        text = _stored_text(document)
        assert TOKEN not in text and "@" not in text
        for canary in CANARIES:
            assert canary not in text
    for path in tmp_path.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            assert TOKEN.encode() not in raw, path
            assert CANARY_EMAIL.encode() not in raw, path


@pytest.mark.asyncio
async def test_a_failed_run_leaves_no_token_or_query_in_dlts_trace(
    clean_environment, monkeypatch, tmp_path
):
    """The community source's error kept the request URL, team id and project id in its chain,
    and dlt pickles the chain into its trace file."""
    fake = _team()
    fake.fail("/v7/deployments", 400)
    serve_over_http(monkeypatch, fake, TOKEN)

    with pytest.raises(Exception, match="HTTP 400") as raised:
        await _sync(fake, access_token=TOKEN)

    chain, error = [], raised.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(e, vercel_module.VercelAPIError) for e in chain)
    traces = [path for path in tmp_path.rglob("trace.pickle")]
    assert traces, "dlt wrote its trace"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            assert TOKEN.encode() not in raw, path
            assert b"teamId=" not in raw, path
            assert b"api.vercel.com/v7" not in raw, path
