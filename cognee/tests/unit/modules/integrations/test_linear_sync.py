"""Unit tests for cognee.modules.integrations.linear.sync.

The Linear API, remember() and the credential store are mocked. What is under
test is the orchestration: which teams a sync walks, when deselected tables and
the former text-path documents go, how webhooks queue behind a running sync,
and how the DLT worker thread gets a fresh token.
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
from cognee.modules.integrations.linear.client import LinearUnauthorizedError

sync_module = importlib.import_module("cognee.modules.integrations.linear.sync")
adapter_module = importlib.import_module("cognee.modules.integrations.linear.adapter")
source_module = importlib.import_module("cognee.tasks.ingestion.connectors.linear")

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
        calls.dataset = dataset_name
        calls.classify = kwargs["classify_error"]
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


def test_dataset_name_is_one_per_workspace_and_identifier_safe():
    assert sync_module.dataset_name_for_org("Acme-Co") == "linear_acme_co"
    assert sync_module.dataset_name_for_org("my.workspace") == "linear_my_workspace"
    assert sync_module.dataset_name_for_org("---") == "linear_workspace"


def test_dataset_name_falls_back_to_the_organization_id():
    credential = _credential()
    credential.provider_metadata = None
    assert sync_module.dataset_name_for_credential(credential) == "linear_org_1"


# -- a full sync -------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_full_sync_walks_every_granted_team_then_cleans_up_and_marks_the_seed(run):
    status, _ = await sync_module._sync_source(_credential(), {"failed": 0}, None)

    assert status == "ok"
    assert run.scopes == [["t1", "t2"]]
    assert run.dataset == _DATASET
    # Nothing is selected, so nothing is retired: a team missing from one
    # listing must not lose its data.
    assert run.retired == []
    run.forgotten.assert_awaited_once()
    assert [call.args[2] for call in run.marker.await_args_list] == [
        {"dlt_seeded": True},
        {"legacy_cleaned": True},
    ]


@pytest.mark.asyncio
async def test_a_selection_syncs_only_those_teams_and_retires_the_rest(run):
    credential = _credential(selected_team_ids=["t2", "t2", "t9"])

    await sync_module._sync_source(credential, {"failed": 0}, None)

    assert run.scopes == [["t2", "t9"]]
    run.teams.assert_not_awaited()
    assert run.retired == [{"linear_t2", "linear_t9"}]
    run.forgotten.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_empty_selection_retires_everything_and_keeps_the_old_documents(run):
    status, _ = await sync_module._sync_source(
        _credential(selected_team_ids=[]), {"failed": 0}, None
    )

    assert status == "ok"
    assert run.scopes == []
    assert run.retired == [set()]
    run.forgotten.assert_not_awaited()
    # Nothing selected is a complete pass: webhooks for teams chosen later can seed them.
    run.marker.assert_awaited_once_with("linear", "org-1", {"dlt_seeded": True})


@pytest.mark.asyncio
async def test_a_failed_team_skips_retirement_cleanup_and_the_seed_marker(run, monkeypatch):
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
async def test_a_malformed_selection_is_refused(run):
    with pytest.raises(ValueError):
        await sync_module._sync_source(_credential(selected_team_ids="t1"), {"failed": 0}, None)


# -- a webhook-sized sync ----------------------------------------------------
@pytest.mark.asyncio
async def test_a_partial_sync_touches_only_the_named_teams_and_never_cleans_up(run):
    await sync_module._sync_source(_credential(dlt_seeded=True), {"failed": 0}, ["t2"])

    assert run.scopes == [["t2"]]
    assert run.retired == []
    run.forgotten.assert_not_awaited()
    run.marker.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_partial_sync_for_a_team_the_app_was_never_granted_does_nothing(run):
    status, counts = await sync_module._sync_source(
        _credential(dlt_seeded=True), {"failed": 0}, ["foreign"]
    )

    assert status == "ok" and counts["failed"] == 0
    assert run.scopes == []


@pytest.mark.asyncio
async def test_a_partial_sync_drops_teams_outside_the_selection(run):
    await sync_module._sync_source(
        _credential(dlt_seeded=True, selected_team_ids=["t1"]), {"failed": 0}, ["t1", "t2"]
    )

    assert run.scopes == [["t1"]]


@pytest.mark.asyncio
async def test_a_partial_sync_for_no_selected_team_does_nothing(run):
    await sync_module._sync_source(
        _credential(dlt_seeded=True, selected_team_ids=["t1"]), {"failed": 0}, ["t2"]
    )

    assert run.scopes == []


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
    await sync_module.request_sync(credential, ["t2"])  # queued, returns at once
    await sync_module.request_sync(credential, ["t3"])
    assert seen == [["t1"]]
    release.set()
    await first

    assert seen == [["t1"], ["t2", "t3"]]
    assert sync_module._pending_teams == {}


@pytest.mark.asyncio
async def test_a_full_sync_that_finds_one_running_is_dropped(monkeypatch):
    monkeypatch.setattr(sync_module, "sync_linear", AsyncMock())
    ingestion._running_syncs.add(("linear", "org-1"))
    try:
        await sync_module.request_sync(_credential())
    finally:
        ingestion._running_syncs.discard(("linear", "org-1"))

    sync_module.sync_linear.assert_not_awaited()


# -- teams -------------------------------------------------------------------
@pytest.fixture
def graphql(monkeypatch):
    mocked = AsyncMock()
    monkeypatch.setattr(sync_module, "graphql", mocked)
    monkeypatch.setattr(adapter_module, "access_token_for", AsyncMock(return_value="lin_tok"))
    return mocked


@pytest.mark.asyncio
async def test_list_teams_pages_and_deduplicates(graphql):
    graphql.side_effect = [
        {
            "teams": {
                "nodes": [_team("t1"), _team("t2")],
                "pageInfo": {"hasNextPage": True, "endCursor": "c"},
            }
        },
        {"teams": {"nodes": [_team("t2"), _team("t3")], "pageInfo": {"hasNextPage": False}}},
    ]

    teams = await sync_module.list_teams(_credential())

    assert [team["id"] for team in teams] == ["t1", "t2", "t3"]
    assert graphql.await_args_list[1].args[2] == {"first": 100, "after": "c"}


@pytest.mark.asyncio
async def test_list_teams_refuses_a_response_without_a_team_list(graphql):
    graphql.return_value = {"teams": None}

    with pytest.raises(RuntimeError):
        await sync_module.list_teams(_credential())


@pytest.mark.asyncio
async def test_a_401_on_the_teams_query_refreshes_the_token_and_asks_again(graphql, monkeypatch):
    graphql.side_effect = [
        LinearUnauthorizedError("Linear LinearTeams failed: HTTP 401"),
        {"teams": {"nodes": [_team("t1")], "pageInfo": {"hasNextPage": False}}},
    ]
    monkeypatch.setattr(
        adapter_module, "access_token_for", AsyncMock(side_effect=["lin_tok", "lin_tok_2"])
    )

    teams = await sync_module.list_teams(_credential())

    assert [team["id"] for team in teams] == ["t1"]
    assert [call.args[0] for call in graphql.await_args_list] == ["lin_tok", "lin_tok_2"]


@pytest.mark.asyncio
async def test_list_resources_describes_teams_for_the_generic_picker(monkeypatch):
    monkeypatch.setattr(
        adapter_module,
        "list_teams",
        AsyncMock(
            return_value=[_team("t1", visibility="public"), _team("t2", visibility="private")]
        ),
    )

    resources = await adapter_module.LinearIntegration().list_resources(_credential())

    assert [(r["id"], r["attributes"]["private"]) for r in resources] == [
        ("t1", False),
        ("t2", True),
    ]
    assert adapter_module.LinearIntegration.resource_selection_key == "selected_team_ids"
    assert adapter_module.LinearIntegration().dataset_name(_credential()) == _DATASET


@pytest.mark.asyncio
async def test_install_and_manual_sync_both_start_a_full_sync(monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(adapter_module, "request_sync", request)
    integration = adapter_module.LinearIntegration()
    credential = _credential()

    await integration.on_installed(credential)
    await integration.sync_now(credential)

    assert [call.args for call in request.await_args_list] == [(credential,), (credential,)]


# -- webhook payloads --------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "Issue", "data": {"teamId": "t1"}}, ["t1"]),
        ({"type": "Comment", "data": {"issue": {"teamId": "t2"}}}, ["t2"]),
        ({"type": "Comment", "data": {"teamId": "t3"}}, ["t3"]),
        ({"type": "Project", "data": {"teamIds": ["t1", "t2", "t1"]}}, ["t1", "t2"]),
        ({"type": "Issue", "data": {}}, []),
        ({"type": "Issue", "data": "x"}, []),
        ({"type": "Cycle", "data": {"teamId": "t1"}}, []),
    ],
)
def test_team_ids_from_event(payload, expected):
    assert sync_module.team_ids_from_event(payload) == expected


# -- the worker thread's client ---------------------------------------------
class _FakeClient:
    tokens: list[str] = []
    fail_first = False

    def __init__(self, token):
        self.token = token
        self.rate_limit = {"requests": 4000, "complexity": 1_000_000}
        type(self).tokens.append(token)

    def execute(self, query, variables=None):
        if type(self).fail_first and len(type(self).tokens) == 1:
            raise source_module.LinearAuthError("rejected")
        return {"token": self.token}


@pytest.mark.asyncio
async def test_the_service_takes_a_fresh_token_per_request_and_retries_a_401_once(monkeypatch):
    _FakeClient.tokens = []
    _FakeClient.fail_first = True
    access = AsyncMock(side_effect=["old", "new"])
    monkeypatch.setattr(adapter_module, "access_token_for", access)
    monkeypatch.setattr(source_module, "LinearClient", _FakeClient)
    service = sync_module._LinearService(_credential(), asyncio.get_running_loop())

    result = await asyncio.to_thread(service.execute, "query A { x }")

    assert result == {"token": "new"}
    assert _FakeClient.tokens == ["old", "new"]
    assert access.await_args_list[1].kwargs == {"rejected": "old"}
    assert service.rate_limit == {"requests": 4000, "complexity": 1_000_000}


@pytest.mark.asyncio
async def test_the_service_does_not_retry_when_no_new_token_exists(monkeypatch):
    _FakeClient.tokens = []
    _FakeClient.fail_first = True
    monkeypatch.setattr(adapter_module, "access_token_for", AsyncMock(return_value="same"))
    monkeypatch.setattr(source_module, "LinearClient", _FakeClient)
    service = sync_module._LinearService(_credential(), asyncio.get_running_loop())

    with pytest.raises(source_module.LinearAuthError):
        await asyncio.to_thread(service.execute, "query A { x }")


def test_typed_source_errors_get_their_own_count_keys():
    team = source_module.LinearTeamNotFoundError("gone")
    wrapped = RuntimeError("DLT ingestion failed")
    wrapped.__cause__ = team
    assert sync_module._classify_error(wrapped) == "failed_team_not_found"
    assert sync_module._classify_error(source_module.LinearAuthError("x")) == "failed_auth"
    assert sync_module._classify_error(RuntimeError("other")) == "failed_ingestion"


# -- documents the former text path wrote ----------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Linear issue COG-1: Fix login\nURL: https://linear.app/x\nState: In Progress",
        "Linear issue COG-1: Fix login\nURL: https://linear.app/x\nState: Todo\nDescription: a\nb",
        # no title: the old writer right-stripped the first line
        "Linear issue COG-2:\nURL: \nState: Unknown",
        "Linear issue unknown: Ünïcode títle\nURL: \nState: Unknown",
        "Linear issue 0b1c-uuid: Title\nURL: u\nState: Done",
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
        "linear issue COG-1: Fix login\nURL: u\nState: Todo",
        "Linear issue COG-1: Fix login\nURL: u\nState: s\nmy own note after",
    ],
)
def test_a_note_that_only_looks_like_an_old_issue_does_not_match(text):
    assert not sync_module._LEGACY_TEXT.match(text.encode())


async def _legacy_dataset(tmp_path, rows, collaborator):
    """Dataset rows on a real in-memory SQLite table; returns (engine, sessions, ids, dataset)."""
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
        for key, (where, metadata, text, owner) in rows.items():
            path = tmp_path / key
            path.write_text(text)
            ids[key] = uuid4()
            db.add(
                Data(
                    id=ids[key],
                    dataset_id=dataset if where == "mine" else other_dataset,
                    name=key,
                    raw_data_location=str(path),
                    system_metadata=metadata,
                    owner_id={"me": _USER_ID, "collab": collaborator, "other": other_owner}[owner],
                )
            )
        await db.commit()
    return engine, sessions, ids, dataset


_OLD = "Linear issue COG-1: Fix login\nURL: \nState: Todo"


@pytest.mark.asyncio
async def test_only_the_owners_untagged_documents_of_the_old_shape_are_forgotten(tmp_path):
    rows = {
        "legacy": ("mine", None, _OLD, "me"),
        "legacy_no_title": ("mine", None, "Linear issue COG-5:\nURL: \nState: Todo", "me"),
        "legacy_empty_metadata": ("mine", {}, _OLD, "me"),
        "note_with_the_same_first_words": (
            "mine",
            None,
            "Linear issue tracking notes for Q3 planning",
            "me",
        ),
        "by_hand": ("mine", None, "Meeting notes the user remembered themselves", "me"),
        "tagged": ("mine", {"source": "linear", "table_name": "linear_x"}, _OLD, "me"),
        "collaborators": ("mine", None, _OLD, "collab"),
        "other_dataset": ("theirs", None, _OLD, "other"),
    }
    engine, sessions, ids, dataset = await _legacy_dataset(tmp_path, rows, uuid4())
    try:
        forget = AsyncMock()
        with (
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                lambda: SimpleNamespace(get_async_session=sessions),
            ),
            patch("cognee.api.v1.forget.forget.forget", forget),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
            patch.object(ingestion, "require_active_credential", AsyncMock()),
        ):
            count = await sync_module.forget_legacy_documents(_credential(), _DATASET)

        assert count == 3
        assert {call.kwargs["data_id"] for call in forget.await_args_list} == {
            ids["legacy"],
            ids["legacy_no_title"],
            ids["legacy_empty_metadata"],
        }
        assert {call.kwargs["dataset_id"] for call in forget.await_args_list} == {dataset}
        assert all(call.kwargs["user"] == "owner" for call in forget.await_args_list)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_connection_that_goes_inactive_mid_cleanup_stops_the_deletes(tmp_path):
    from cognee.modules.integrations.credentials import CredentialInactiveError

    rows = {f"legacy{n}": ("mine", None, _OLD, "me") for n in range(3)}
    engine, sessions, _, _ = await _legacy_dataset(tmp_path, rows, uuid4())
    try:
        forget = AsyncMock()
        active = AsyncMock(side_effect=[None, CredentialInactiveError(), None])
        with (
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                lambda: SimpleNamespace(get_async_session=sessions),
            ),
            patch("cognee.api.v1.forget.forget.forget", forget),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
            patch.object(ingestion, "require_active_credential", active),
            pytest.raises(CredentialInactiveError),
        ):
            await sync_module.forget_legacy_documents(_credential(), _DATASET)

        assert forget.await_count == 1
    finally:
        await engine.dispose()


# -- bookkeeping a full pass leaves ------------------------------------------
@pytest.mark.asyncio
async def test_the_seed_marker_is_written_before_retirement_and_cleanup(run, monkeypatch):
    order = []
    run.marker.side_effect = lambda *args: order.append(("marker", args[2]))
    run.forgotten.side_effect = lambda *args: order.append(("cleanup", None))

    async def retire(provider, credential, dataset_name, retained):
        order.append(("retire", None))

    monkeypatch.setattr(ingestion, "retire_resources", retire)

    await sync_module._sync_source(_credential(selected_team_ids=["t1"]), {"failed": 0}, None)

    assert order[0] == ("marker", {"dlt_seeded": True})
    assert [name for name, _ in order[1:3]] == ["retire", "cleanup"]


@pytest.mark.asyncio
async def test_a_failing_cleanup_is_not_fatal_and_is_retried_next_time(run, caplog):
    run.forgotten.side_effect = RuntimeError("graph backend down")

    status, _ = await sync_module._sync_source(_credential(), {"failed": 0}, None)

    assert status == "ok"
    assert [call.args[2] for call in run.marker.await_args_list] == [{"dlt_seeded": True}]


@pytest.mark.asyncio
async def test_a_connection_already_cleaned_does_not_scan_its_documents_again(run):
    await sync_module._sync_source(
        _credential(dlt_seeded=True, legacy_cleaned=True), {"failed": 0}, None
    )

    run.forgotten.assert_not_awaited()
    run.marker.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_missing_team_is_reported_but_does_not_keep_the_connection_unseeded(
    run, monkeypatch
):
    async def one_team_gone(provider, credential, counts, **kwargs):
        counts["failed"] += 1
        counts["failed_team_not_found"] = 1
        return {"linear_t1"}

    monkeypatch.setattr(ingestion, "sync_scopes", one_team_gone)

    status, counts = await sync_module._sync_source(
        _credential(selected_team_ids=["t1", "gone"]), {"failed": 0}, None
    )

    assert status == "degraded"
    assert counts["failed_team_not_found"] == 1
    assert run.retired == [{"linear_t1"}]
    run.forgotten.assert_awaited_once()
    assert {"dlt_seeded": True} in [call.args[2] for call in run.marker.await_args_list]


@pytest.mark.asyncio
async def test_a_full_pass_cut_short_by_the_quota_records_when_to_resume(run, monkeypatch):
    async def cut(provider, credential, counts, **kwargs):
        counts["failed"] += 1
        counts["failed_rate_limit"] = 1
        return {"linear_t1"}

    monkeypatch.setattr(ingestion, "sync_scopes", cut)

    status, _ = await sync_module._sync_source(_credential(), {"failed": 0}, None)

    assert status == "degraded"
    (call,) = run.marker.await_args_list
    assert list(call.args[2]) == ["resume_needed_at"] and call.args[2]["resume_needed_at"]
    run.forgotten.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_clean_full_pass_clears_the_resume_marker(run):
    await sync_module._sync_source(
        _credential(resume_needed_at="2026-10-02T10:00:00+00:00"), {"failed": 0}, None
    )

    assert run.marker.await_args_list[0].args[2] == {"dlt_seeded": True, "resume_needed_at": None}


@pytest.mark.asyncio
async def test_a_partial_run_cut_short_does_not_touch_the_resume_marker(run, monkeypatch):
    async def cut(provider, credential, counts, **kwargs):
        counts["failed"] += 1
        counts["failed_rate_limit"] = 1
        return {"linear_t1"}

    monkeypatch.setattr(ingestion, "sync_scopes", cut)

    await sync_module._sync_source(
        _credential(dlt_seeded=True, resume_needed_at="2026-10-02T10:00:00+00:00"),
        {"failed": 0},
        ["t1"],
    )

    run.marker.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_late_sync_cannot_write_markers_onto_a_connection_that_is_gone(run):
    from cognee.modules.integrations.credentials import CredentialInactiveError

    ingestion.require_active_credential.side_effect = CredentialInactiveError()

    with pytest.raises(CredentialInactiveError):
        await sync_module._mark(_credential(), {"dlt_seeded": True})

    run.marker.assert_not_awaited()


def test_a_reinstall_starts_without_the_previous_installs_markers():
    installation = adapter_module.LinearIntegration().parse_installation(
        {"access_token": "t", "viewer": {}, "organization": {"id": "org-1", "urlKey": "acme"}}
    )

    metadata = installation.provider_metadata
    assert metadata["dlt_seeded"] is False
    assert metadata["legacy_cleaned"] is False
    assert metadata["resume_needed_at"] is None


# -- queued webhooks after a failure ----------------------------------------
@pytest.mark.asyncio
async def test_teams_a_failing_run_had_taken_are_queued_again(monkeypatch):
    sync_module._pending_teams.clear()

    async def failing(credential, team_ids=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(sync_module, "sync_linear", failing)

    with pytest.raises(RuntimeError):
        await sync_module.request_sync(_credential(), ["t1", "t2"])

    assert sync_module._pending_teams == {"org-1": {"t1", "t2"}}
    sync_module._pending_teams.clear()


@pytest.mark.asyncio
async def test_an_inactive_connection_drops_its_queue(monkeypatch):
    from cognee.modules.integrations.credentials import CredentialInactiveError

    sync_module._pending_teams.clear()
    sync_module._pending_teams["org-1"] = {"t9"}
    monkeypatch.setattr(
        sync_module, "sync_linear", AsyncMock(side_effect=CredentialInactiveError())
    )

    with pytest.raises(CredentialInactiveError):
        await sync_module.request_sync(_credential(), ["t1"])

    assert sync_module._pending_teams == {}


@pytest.mark.asyncio
async def test_request_sync_says_whether_it_ran_or_was_queued_or_dropped(monkeypatch):
    sync_module._pending_teams.clear()
    monkeypatch.setattr(sync_module, "sync_linear", AsyncMock())
    assert await sync_module.request_sync(_credential()) is True

    ingestion._running_syncs.add(("linear", "org-1"))
    try:
        assert await sync_module.request_sync(_credential()) is False
        assert await sync_module.request_sync(_credential(), ["t1"]) is True
    finally:
        ingestion._running_syncs.discard(("linear", "org-1"))
        sync_module._pending_teams.clear()


def test_a_delivery_naming_a_crowd_of_teams_is_cut_to_a_bounded_list():
    payload = {"type": "Project", "data": {"teamIds": [f"t{n}" for n in range(500)]}}
    assert len(sync_module.team_ids_from_event(payload)) == 50


# -- the read window --------------------------------------------------------
def test_a_state_line_cut_off_by_the_window_is_not_taken_for_a_whole_document():
    window = sync_module._LEGACY_READ_BYTES
    state = "x" * (window + 100)
    text = f"Linear issue COG-1: t\nURL: u\nState: {state}\nMy own trailing notes".encode()

    assert not sync_module._is_legacy_text(text[: window + 1])


def test_a_short_legacy_document_and_one_longer_than_the_window_with_a_description_match():
    window = sync_module._LEGACY_READ_BYTES
    short = b"Linear issue COG-1: t\nURL: u\nState: Todo"
    described = b"Linear issue COG-1: t\nURL: u\nState: Todo\nDescription: " + b"d" * (window * 2)

    assert sync_module._is_legacy_text(short)
    assert sync_module._is_legacy_text(described[: window + 1])


@pytest.mark.asyncio
async def test_a_long_title_and_a_cut_off_note_are_judged_by_the_window_not_by_the_first_bytes(
    tmp_path,
):
    window = sync_module._LEGACY_READ_BYTES
    rows = {
        "long_title": ("mine", None, f"Linear issue COG-1: {'t' * 500}\nURL: u\nState: Todo", "me"),
        "cut_off_note": (
            "mine",
            None,
            f"Linear issue COG-2: t\nURL: u\nState: {'s' * (window + 50)}\nmy own notes",
            "me",
        ),
    }
    engine, sessions, ids, _ = await _legacy_dataset(tmp_path, rows, uuid4())
    try:
        forget = AsyncMock()
        with (
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                lambda: SimpleNamespace(get_async_session=sessions),
            ),
            patch("cognee.api.v1.forget.forget.forget", forget),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
            patch.object(ingestion, "require_active_credential", AsyncMock()),
        ):
            await sync_module.forget_legacy_documents(_credential(), _DATASET)

        assert {call.kwargs["data_id"] for call in forget.await_args_list} == {ids["long_title"]}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_document_whose_file_cannot_be_read_is_skipped_not_deleted(tmp_path):
    rows = {
        "legacy": ("mine", None, _OLD, "me"),
        "unreadable": ("mine", None, _OLD, "me"),
    }
    engine, sessions, ids, _ = await _legacy_dataset(tmp_path, rows, uuid4())
    (tmp_path / "unreadable").unlink()
    try:
        forget = AsyncMock()
        with (
            patch(
                "cognee.infrastructure.databases.relational.get_relational_engine",
                lambda: SimpleNamespace(get_async_session=sessions),
            ),
            patch("cognee.api.v1.forget.forget.forget", forget),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
            patch.object(ingestion, "require_active_credential", AsyncMock()),
        ):
            count = await sync_module.forget_legacy_documents(_credential(), _DATASET)

        assert count == 1
        assert [call.kwargs["data_id"] for call in forget.await_args_list] == [ids["legacy"]]
    finally:
        await engine.dispose()


def test_a_typed_error_is_found_through_the_context_chain_too():
    inner = source_module.LinearTeamNotFoundError("gone")
    outer = RuntimeError("DLT ingestion failed")
    try:
        try:
            raise inner
        except source_module.LinearTeamNotFoundError:
            raise outer from None
    except RuntimeError as raised:
        raised.__context__ = inner
        assert raised.__cause__ is None
        assert sync_module._classify_error(raised) == "failed_team_not_found"


# -- markers are compared with what is stored ---------------------------------
@pytest.mark.asyncio
async def test_a_marker_is_written_when_the_stored_row_lacks_it_even_if_the_snapshot_has_it(run):
    # The run started after a seed; a reinstall has since reset the stored marker.
    stale_snapshot = _credential(dlt_seeded=True, resume_needed_at="2026-10-02T10:00:00+00:00")
    stored = _credential(dlt_seeded=False, resume_needed_at=None)
    ingestion.require_active_credential.side_effect = None
    ingestion.require_active_credential.return_value = stored

    await sync_module._mark(stale_snapshot, {"dlt_seeded": True, "resume_needed_at": None})

    run.marker.assert_awaited_once_with("linear", "org-1", {"dlt_seeded": True})


@pytest.mark.asyncio
async def test_nothing_is_written_when_the_stored_row_already_has_the_values(run):
    stored = _credential(dlt_seeded=True)
    ingestion.require_active_credential.side_effect = None
    ingestion.require_active_credential.return_value = stored

    await sync_module._mark(_credential(), {"dlt_seeded": True})

    run.marker.assert_not_awaited()


# -- a first full pass that fails outright -----------------------------------
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
async def test_an_unseeded_first_full_pass_that_raises_is_stamped_for_the_resume_worker(
    failing_source,
):
    with pytest.raises(RuntimeError, match="boom"):
        await sync_module.sync_linear(_credential())

    (call,) = failing_source.await_args_list
    assert list(call.args[1]) == ["resume_needed_at"] and call.args[1]["resume_needed_at"]


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
async def test_a_failing_stamp_never_replaces_the_error_in_flight(failing_source):
    failing_source.side_effect = OSError("db down")

    with pytest.raises(RuntimeError, match="boom"):
        await sync_module.sync_linear(_credential())


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
    only_other_users = SimpleNamespace(provider_account_id="org-z", user_id=someone_else)
    assert await sync_module._dataset_is_shared(only_other_users, "linear_acme_co") is True


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
