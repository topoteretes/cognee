"""Unit tests for cognee.modules.integrations.linear.sync.

The Linear API, remember() and the credential store are mocked. What is under
test is the orchestration: which teams a sync walks, when deselected tables and
the former text-path documents go, how webhooks queue behind a running sync,
and how the DLT worker thread gets a fresh token. The legacy cleanup runs on a
real SQLite table, and test_linear_agent_sync.py covers the whole path.
"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data, Dataset
from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.linear.client import LinearUnauthorizedError

sync_module = importlib.import_module("cognee.modules.integrations.linear.sync")
adapter_module = importlib.import_module("cognee.modules.integrations.linear.adapter")
source_module = importlib.import_module("cognee.tasks.ingestion.connectors.linear")
# Patched by module object: the string target resolves to the function on Python 3.10.
_forget_module = importlib.import_module("cognee.api.v1.forget.forget")

_USER_ID = uuid4()
_DATASET = "linear_acme_co"


def _credential(**metadata):
    return SimpleNamespace(
        status="active",
        provider_account_id="org-1",
        provider_metadata={"organization_url_key": "Acme-Co", **metadata},
        user_id=_USER_ID,
    )


def _team(team_id, **extra):
    return {"id": team_id, "key": team_id.upper(), "name": f"Team {team_id}", **extra}


@pytest.fixture
def run(monkeypatch):
    """A sync with every collaborator mocked; `sync_scopes` reports all scopes retained."""
    calls = SimpleNamespace(
        scopes=[],
        retired=[],
        forgotten=AsyncMock(return_value=0),
        marker=AsyncMock(),
        teams=AsyncMock(return_value=[_team("t1"), _team("t2")]),
    )

    async def sync_scopes(provider, credential, counts, *, scopes, dataset_name, **kwargs):
        calls.scopes.append(list(scopes))
        return {f"linear_{scope}" for scope in scopes}

    async def retire(provider, credential, dataset_name, retained):
        calls.retired.append(set(retained))

    monkeypatch.setattr(ingestion, "sync_scopes", sync_scopes)
    monkeypatch.setattr(ingestion, "retire_resources", retire)
    monkeypatch.setattr(ingestion, "source_factory", lambda provider: lambda **kwargs: kwargs)
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    monkeypatch.setattr(sync_module, "forget_legacy_documents", calls.forgotten)
    monkeypatch.setattr(sync_module, "_dataset_is_shared", AsyncMock(return_value=False))
    monkeypatch.setattr(sync_module, "list_teams", calls.teams)
    monkeypatch.setattr(
        "cognee.modules.integrations.credentials.update_provider_metadata", calls.marker
    )
    return calls


def _written(run):
    return [call.args[2] for call in run.marker.await_args_list]


def test_dataset_name_is_one_per_workspace_and_identifier_safe():
    assert sync_module.dataset_name_for_org("Acme-Co") == "linear_acme_co"
    assert sync_module.dataset_name_for_org("my.workspace") == "linear_my_workspace"
    assert sync_module.dataset_name_for_org("---") == "linear_workspace"
    credential = _credential()
    credential.provider_metadata = None
    assert sync_module.dataset_name_for_credential(credential) == "linear_org_1"


# -- a full sync -------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_full_sync_walks_every_granted_team_then_cleans_up_and_marks_the_seed(run):
    status, _ = await sync_module._sync_source(_credential(), {"failed": 0}, None)

    assert status == "ok"
    assert run.scopes == [["t1", "t2"]]
    run.forgotten.assert_awaited_once()
    assert _written(run) == [{"dlt_seeded": True}, {"legacy_cleaned": True}]
    # Nothing is selected, so nothing is retired: a team missing from one listing
    # must not lose its data.
    assert run.retired == []


@pytest.mark.asyncio
async def test_a_selection_syncs_only_those_teams_and_retires_the_rest(run):
    await sync_module._sync_source(_credential(selected_team_ids=["t2", "t9"]), {"failed": 0}, None)

    assert run.scopes == [["t2", "t9"]]
    run.teams.assert_not_awaited()
    assert run.retired == [{"linear_t2", "linear_t9"}]


@pytest.mark.asyncio
async def test_an_empty_selection_retires_everything_and_still_seeds(run):
    """A full pass with nothing selected is complete: webhooks for teams chosen later can seed them."""
    status, _ = await sync_module._sync_source(
        _credential(selected_team_ids=[]), {"failed": 0}, None
    )

    assert status == "ok"
    assert run.scopes == [] and run.retired == [set()]
    run.forgotten.assert_not_awaited()
    assert _written(run) == [{"dlt_seeded": True}]


@pytest.mark.asyncio
async def test_a_failed_team_skips_retirement_cleanup_and_the_seed(run, monkeypatch):
    async def failing(provider, credential, counts, **kwargs):
        counts["failed"] += 1
        return {"linear_t1"}

    monkeypatch.setattr(ingestion, "sync_scopes", failing)

    status, _ = await sync_module._sync_source(
        _credential(selected_team_ids=["t1"]), {"failed": 0}, None
    )

    assert status == "degraded"
    assert run.retired == []
    run.forgotten.assert_not_awaited()
    run.marker.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_missing_team_is_reported_but_does_not_keep_the_connection_unseeded(
    run, monkeypatch
):
    """A selected team that is gone fails every run; it must not block the other teams' cleanup."""

    async def one_team_gone(provider, credential, counts, **kwargs):
        counts["failed"] += 1
        counts["failed_team_not_found"] = 1
        return {"linear_t1"}

    monkeypatch.setattr(ingestion, "sync_scopes", one_team_gone)

    status, counts = await sync_module._sync_source(
        _credential(selected_team_ids=["t1", "gone"]), {"failed": 0}, None
    )

    assert status == "degraded" and counts["failed_team_not_found"] == 1
    assert run.retired == [{"linear_t1"}]
    assert {"dlt_seeded": True} in _written(run)


@pytest.mark.asyncio
async def test_a_failing_cleanup_is_not_fatal_and_is_retried_next_time(run):
    """The seed marker is already written, so a cleanup error cannot keep webhooks off."""
    run.forgotten.side_effect = RuntimeError("graph backend down")

    status, _ = await sync_module._sync_source(_credential(), {"failed": 0}, None)

    assert status == "ok"
    assert _written(run) == [{"dlt_seeded": True}]


@pytest.mark.asyncio
async def test_a_connection_already_cleaned_does_not_scan_its_documents_again(run):
    await sync_module._sync_source(
        _credential(dlt_seeded=True, legacy_cleaned=True), {"failed": 0}, None
    )

    run.forgotten.assert_not_awaited()
    run.marker.assert_not_awaited()


# -- webhook-sized syncs -----------------------------------------------------
@pytest.mark.asyncio
async def test_a_partial_sync_touches_only_the_named_teams_and_never_cleans_up(run):
    await sync_module._sync_source(_credential(dlt_seeded=True), {"failed": 0}, ["t2"])

    assert run.scopes == [["t2"]]
    assert run.retired == []
    run.forgotten.assert_not_awaited()
    run.marker.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "named"),
    [
        ({}, ["foreign"]),  # a team the app was never granted
        ({"selected_team_ids": ["t1"]}, ["t2"]),  # a team the user did not select
    ],
)
async def test_a_partial_sync_ignores_teams_that_are_not_synced_for_this_connection(
    run, metadata, named
):
    await sync_module._sync_source(_credential(dlt_seeded=True, **metadata), {"failed": 0}, named)

    assert run.scopes == []


# -- resume bookkeeping ------------------------------------------------------
@pytest.fixture
def cut_short(run, monkeypatch):
    async def cut(provider, credential, counts, **kwargs):
        counts["failed"] += 1
        counts["failed_rate_limit"] = 1
        return {"linear_t1"}

    monkeypatch.setattr(ingestion, "sync_scopes", cut)
    return run


@pytest.mark.asyncio
async def test_a_full_pass_cut_short_by_the_quota_records_when_to_resume(cut_short):
    status, _ = await sync_module._sync_source(_credential(), {"failed": 0}, None)

    assert status == "degraded"
    (written,) = _written(cut_short)
    assert list(written) == ["resume_needed_at"] and written["resume_needed_at"]
    cut_short.forgotten.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_partial_run_cut_short_leaves_the_resume_marker_alone(cut_short):
    """Only a full pass writes it: last_sync_counts is replaced by every run, partial ones too."""
    await sync_module._sync_source(
        _credential(dlt_seeded=True, resume_needed_at="2026-10-02T10:00:00+00:00"),
        {"failed": 0},
        ["t1"],
    )

    cut_short.marker.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_clean_full_pass_clears_the_resume_marker(run):
    await sync_module._sync_source(
        _credential(resume_needed_at="2026-10-02T10:00:00+00:00"), {"failed": 0}, None
    )

    assert _written(run)[0] == {"dlt_seeded": True, "resume_needed_at": None}


@pytest.fixture
def failing_source(monkeypatch):
    marker = AsyncMock()
    monkeypatch.setattr(sync_module, "_mark", marker)
    monkeypatch.setattr(sync_module, "_sync_source", AsyncMock(side_effect=RuntimeError("boom")))

    async def run_sync(provider, credential, sync_source):
        await sync_source(credential, {"failed": 0})

    monkeypatch.setattr(ingestion, "run_sync", run_sync)
    return marker


@pytest.mark.asyncio
async def test_a_first_full_pass_that_raises_is_stamped_for_the_resume_worker(failing_source):
    with pytest.raises(RuntimeError, match="boom"):
        await sync_module.sync_linear(_credential())

    (call,) = failing_source.await_args_list
    assert list(call.args[1]) == ["resume_needed_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential", "team_ids"),
    [
        (_credential(dlt_seeded=True), None),  # seeded: a permanent failure must not loop
        (_credential(), ["t1"]),  # a webhook's partial run
    ],
)
async def test_a_failure_is_not_stamped_once_seeded_or_for_a_partial_run(
    failing_source, credential, team_ids
):
    with pytest.raises(RuntimeError, match="boom"):
        await sync_module.sync_linear(credential, team_ids)

    failing_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_marker_is_compared_with_the_stored_row_not_the_runs_snapshot(run):
    """A reinstall can reset a marker that the snapshot the run started with still shows as set."""
    stale_snapshot = _credential(dlt_seeded=True, resume_needed_at="2026-10-02T10:00:00+00:00")
    ingestion.require_active_credential.side_effect = None
    ingestion.require_active_credential.return_value = _credential(dlt_seeded=False)

    await sync_module._mark(stale_snapshot, {"dlt_seeded": True, "resume_needed_at": None})

    run.marker.assert_awaited_once_with("linear", "org-1", {"dlt_seeded": True})


@pytest.mark.asyncio
async def test_a_late_sync_cannot_write_markers_onto_a_connection_that_is_gone(run):
    ingestion.require_active_credential.side_effect = CredentialInactiveError()

    with pytest.raises(CredentialInactiveError):
        await sync_module._mark(_credential(), {"dlt_seeded": True})

    run.marker.assert_not_awaited()


def test_a_reinstall_starts_without_the_previous_installs_markers():
    installation = adapter_module.LinearIntegration().parse_installation(
        {"access_token": "t", "viewer": {}, "organization": {"id": "org-1", "urlKey": "acme"}}
    )

    assert installation.provider_metadata["dlt_seeded"] is False
    assert installation.provider_metadata["legacy_cleaned"] is False
    assert installation.provider_metadata["resume_needed_at"] is None


# -- queued webhooks ---------------------------------------------------------
@pytest.mark.asyncio
async def test_a_team_named_during_a_running_sync_is_synced_right_after_it(monkeypatch):
    seen = []
    release = asyncio.Event()

    async def fake_sync_linear(credential, team_ids=None):
        seen.append(team_ids)
        key = ("linear", "org-1")
        ingestion._running_syncs.add(key)
        try:
            if len(seen) == 1:
                await release.wait()
        finally:
            ingestion._running_syncs.discard(key)

    monkeypatch.setattr(sync_module, "sync_linear", fake_sync_linear)
    sync_module._pending_teams.clear()
    credential = _credential()

    first = asyncio.create_task(sync_module.request_sync(credential, ["t1"]))
    await asyncio.sleep(0)
    assert await sync_module.request_sync(credential, ["t2"]) is True  # queued
    assert await sync_module.request_sync(credential) is False  # a full request is dropped
    release.set()
    await first

    assert seen == [["t1"], ["t2"]]
    assert sync_module._pending_teams == {}


@pytest.mark.asyncio
async def test_teams_a_failing_run_had_taken_are_queued_again(monkeypatch):
    sync_module._pending_teams.clear()
    monkeypatch.setattr(sync_module, "sync_linear", AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        await sync_module.request_sync(_credential(), ["t1", "t2"])

    assert sync_module._pending_teams == {"org-1": {"t1", "t2"}}
    sync_module._pending_teams.clear()


@pytest.mark.asyncio
async def test_an_inactive_connection_drops_its_queue(monkeypatch):
    sync_module._pending_teams.clear()
    sync_module._pending_teams["org-1"] = {"t9"}
    monkeypatch.setattr(
        sync_module, "sync_linear", AsyncMock(side_effect=CredentialInactiveError())
    )

    with pytest.raises(CredentialInactiveError):
        await sync_module.request_sync(_credential(), ["t1"])

    assert sync_module._pending_teams == {}


# -- teams -------------------------------------------------------------------
@pytest.fixture
def graphql(monkeypatch):
    mocked = AsyncMock()
    monkeypatch.setattr(sync_module, "graphql", mocked)
    monkeypatch.setattr(adapter_module, "access_token_for", AsyncMock(return_value="lin_tok"))
    return mocked


@pytest.mark.asyncio
async def test_list_teams_pages_and_deduplicates(graphql):
    page = {"hasNextPage": True, "endCursor": "c"}
    graphql.side_effect = [
        {"teams": {"nodes": [_team("t1"), _team("t2")], "pageInfo": page}},
        {"teams": {"nodes": [_team("t2"), _team("t3")], "pageInfo": {"hasNextPage": False}}},
    ]

    teams = await sync_module.list_teams(_credential())

    assert [team["id"] for team in teams] == ["t1", "t2", "t3"]
    assert graphql.await_args_list[1].args[2] == {"first": 100, "after": "c"}


@pytest.mark.asyncio
async def test_a_401_on_the_teams_query_refreshes_the_token_and_asks_again(graphql, monkeypatch):
    graphql.side_effect = [
        LinearUnauthorizedError("Linear LinearTeams failed: HTTP 401"),
        {"teams": {"nodes": [_team("t1")], "pageInfo": {"hasNextPage": False}}},
    ]
    monkeypatch.setattr(
        adapter_module, "access_token_for", AsyncMock(side_effect=["lin_tok", "lin_tok_2"])
    )

    await sync_module.list_teams(_credential())

    assert [call.args[0] for call in graphql.await_args_list] == ["lin_tok", "lin_tok_2"]


@pytest.mark.asyncio
async def test_the_adapter_describes_teams_for_the_generic_picker_and_names_its_dataset(
    monkeypatch,
):
    monkeypatch.setattr(
        adapter_module,
        "list_teams",
        AsyncMock(
            return_value=[_team("t1", visibility="public"), _team("t2", visibility="private")]
        ),
    )
    integration = adapter_module.LinearIntegration()

    resources = await integration.list_resources(_credential())

    assert [(r["id"], r["attributes"]["private"]) for r in resources] == [
        ("t1", False),
        ("t2", True),
    ]
    assert integration.resource_selection_key == "selected_team_ids"
    assert integration.dataset_name(_credential()) == _DATASET


@pytest.mark.asyncio
async def test_install_and_manual_sync_both_start_a_full_sync(monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(adapter_module, "request_sync", request)
    integration = adapter_module.LinearIntegration()
    credential = _credential()

    await integration.on_installed(credential)
    await integration.sync_now(credential)

    assert [call.args for call in request.await_args_list] == [(credential,), (credential,)]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "Issue", "data": {"teamId": "t1"}}, ["t1"]),
        ({"type": "Comment", "data": {"issue": {"teamId": "t2"}}}, ["t2"]),
        ({"type": "Project", "data": {"teamIds": ["t1", "t2", "t1"]}}, ["t1", "t2"]),
        ({"type": "Cycle", "data": {"teamId": "t1"}}, []),
    ],
)
def test_team_ids_from_event(payload, expected):
    assert sync_module.team_ids_from_event(payload) == expected


# -- the worker thread's client ---------------------------------------------
class _FakeClient:
    tokens: list[str] = []

    def __init__(self, token):
        self.token = token
        self.rate_limit = {"requests": 4000, "complexity": 1_000_000}
        type(self).tokens.append(token)

    def execute(self, query, variables=None):
        if type(self).tokens == ["old"]:
            raise source_module.LinearAuthError("rejected")
        return {"token": self.token}


@pytest.mark.asyncio
async def test_the_service_takes_a_fresh_token_and_retries_a_401_once(monkeypatch):
    _FakeClient.tokens = []
    access = AsyncMock(side_effect=["old", "new"])
    monkeypatch.setattr(adapter_module, "access_token_for", access)
    monkeypatch.setattr(source_module, "LinearClient", _FakeClient)
    service = sync_module._LinearService(_credential(), asyncio.get_running_loop())

    result = await asyncio.to_thread(service.execute, "query A { x }")

    assert result == {"token": "new"}
    assert access.await_args_list[1].kwargs == {"rejected": "old"}
    assert service.rate_limit == {"requests": 4000, "complexity": 1_000_000}


def test_typed_source_errors_get_their_own_count_keys():
    """dlt and remember() wrap the source's error, so the type is found down the chain."""
    wrapped = RuntimeError("DLT ingestion failed")
    wrapped.__cause__ = source_module.LinearTeamNotFoundError("gone")

    assert sync_module._classify_error(wrapped) == "failed_team_not_found"
    assert sync_module._classify_error(source_module.LinearAuthError("x")) == "failed_auth"
    assert sync_module._classify_error(RuntimeError("other")) == "failed_ingestion"


# -- documents the former text path wrote ----------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Linear issue COG-1: Fix login\nURL: https://linear.app/x\nState: In Progress",
        "Linear issue COG-1: Fix login\nURL: u\nState: Todo\nDescription: a\nb",
        "Linear issue COG-2:\nURL: \nState: Unknown",  # no title: the old writer right-stripped
        "Linear issue unknown: Ünïcode títle\nURL: \nState: Unknown",
    ],
)
def test_the_legacy_shape_matches_what_the_old_writer_produced(text):
    assert sync_module._LEGACY_TEXT.match(text.encode())


@pytest.mark.parametrize(
    "text",
    [
        "Linear issue tracking notes for Q3 planning",
        "Linear issue COG-1: looks like one\nbut has no URL line",
        "Notes\nLinear issue COG-1: Fix login\nURL: u\nState: Todo",
        "Linear issue COG-1: Fix login\nURL: u\nState: s\nmy own note after",
    ],
)
def test_a_note_that_only_looks_like_an_old_issue_does_not_match(text):
    assert not sync_module._LEGACY_TEXT.match(text.encode())


_OLD = "Linear issue COG-1: Fix login\nURL: \nState: Todo"


async def _legacy_dataset(tmp_path, rows, collaborator):
    """Rows on a real in-memory SQLite table: returns (engine, sessions, ids, dataset id)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    other_owner = uuid4()
    dataset, other_dataset = uuid4(), uuid4()
    async with engine.begin() as conn:
        await conn.run_sync(Dataset.__table__.create)
        await conn.run_sync(Data.__table__.create)
    ids = {}
    async with sessions() as db:
        db.add_all(
            [
                Dataset(id=dataset, name=_DATASET, owner_id=_USER_ID),
                Dataset(id=other_dataset, name=_DATASET, owner_id=other_owner),
            ]
        )
        for key, (where, metadata, text, owner, extension) in rows.items():
            path = tmp_path / key
            path.write_text(text)
            ids[key] = uuid4()
            db.add(
                Data(
                    id=ids[key],
                    dataset_id=dataset if where == "mine" else other_dataset,
                    name=key,
                    extension=extension,
                    raw_data_location=str(path),
                    system_metadata=metadata,
                    owner_id={"me": _USER_ID, "collab": collaborator, "other": other_owner}[owner],
                )
            )
        await db.commit()
    return engine, sessions, ids, dataset


def _cleanup_patches(sessions, forget, active=None):
    return (
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            lambda: SimpleNamespace(get_async_session=sessions),
        ),
        patch.object(_forget_module, "forget", forget),
        patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        patch.object(ingestion, "require_active_credential", active or AsyncMock()),
    )


@pytest.mark.asyncio
async def test_only_the_owners_untagged_text_documents_of_the_old_shape_are_forgotten(tmp_path):
    rows = {
        "legacy": ("mine", None, _OLD, "me", "txt"),
        "legacy_empty_metadata": ("mine", {}, _OLD, "me", "txt"),
        "same_first_words": ("mine", None, "Linear issue tracking notes", "me", "txt"),
        "tagged": ("mine", {"source": "linear", "table_name": "t"}, _OLD, "me", "txt"),
        "not_text": ("mine", None, _OLD, "me", "md"),
        "collaborators": ("mine", None, _OLD, "collab", "txt"),
        "other_dataset": ("theirs", None, _OLD, "other", "txt"),
        "unreadable": ("mine", None, _OLD, "me", "txt"),
    }
    engine, sessions, ids, dataset = await _legacy_dataset(tmp_path, rows, uuid4())
    (tmp_path / "unreadable").unlink()  # skipped, never deleted
    try:
        forget = AsyncMock()
        patches = _cleanup_patches(sessions, forget)
        with patches[0], patches[1], patches[2], patches[3]:
            count = await sync_module.forget_legacy_documents(_credential(), _DATASET)

        assert count == 2
        assert {call.kwargs["data_id"] for call in forget.await_args_list} == {
            ids["legacy"],
            ids["legacy_empty_metadata"],
        }
        assert all(call.kwargs["dataset_id"] == dataset for call in forget.await_args_list)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_connection_that_goes_inactive_mid_cleanup_stops_the_deletes(tmp_path):
    rows = {f"legacy{n}": ("mine", None, _OLD, "me", "txt") for n in range(3)}
    engine, sessions, _, _ = await _legacy_dataset(tmp_path, rows, uuid4())
    try:
        forget = AsyncMock()
        active = AsyncMock(side_effect=[None, CredentialInactiveError(), None])
        patches = _cleanup_patches(sessions, forget, active)
        with patches[0], patches[1], patches[2], patches[3], pytest.raises(CredentialInactiveError):
            await sync_module.forget_legacy_documents(_credential(), _DATASET)

        assert forget.await_count == 1
    finally:
        await engine.dispose()


# -- two workspaces that normalise to one dataset ---------------------------
@pytest.mark.asyncio
async def test_a_dataset_shared_with_another_active_workspace_of_the_same_user_is_detected(
    credential_db, monkeypatch
):
    from cognee.modules.integrations import credentials as store

    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine",
        store.get_relational_engine,
    )

    async def connect(account, url_key, user, revoked=False):
        await store.upsert_credential(
            provider="linear",
            user_id=user,
            provider_account_id=account,
            token_payload={"access_token": "t"},
            provider_metadata={"organization_url_key": url_key},
        )
        if revoked:
            await store.revoke_credential_by_account("linear", account)

    me, someone_else = uuid4(), uuid4()
    await connect("org-a", "Acme-Co", me)
    await connect("org-b", "acme_co", me)  # same slug, same user
    await connect("org-c", "other-co", me)
    await connect("org-d", "acme.co", someone_else)  # same slug, another user
    await connect("org-e", "acme co", me, revoked=True)  # same slug, revoked

    mine = SimpleNamespace(provider_account_id="org-a", user_id=me)
    alone = SimpleNamespace(provider_account_id="org-c", user_id=me)

    assert await sync_module._dataset_is_shared(mine, "linear_acme_co") is True
    assert await sync_module._dataset_is_shared(alone, "linear_other_co") is False


@pytest.mark.asyncio
async def test_retirement_is_skipped_on_a_shared_dataset_and_runs_otherwise(monkeypatch):
    retire = AsyncMock()
    monkeypatch.setattr(ingestion, "retire_resources", retire)
    shared = AsyncMock(return_value=True)
    monkeypatch.setattr(sync_module, "_dataset_is_shared", shared)

    await sync_module._retire(_credential(), _DATASET, {"linear_x"})
    retire.assert_not_awaited()

    shared.return_value = False
    await sync_module._retire(_credential(), _DATASET, {"linear_x"})
    retire.assert_awaited_once()
