"""The per-scope loop Drive and Linear share."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import CredentialInactiveError

CREDENTIAL = SimpleNamespace(user_id="user-1", provider_account_id="acct-1")


@pytest.fixture
def remember(monkeypatch):
    mocked = AsyncMock(return_value=SimpleNamespace(status="completed"))
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.remember"), "remember", mocked
    )
    monkeypatch.setattr("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    return mocked


async def _run(scopes, counts=None, **kwargs):
    counts = counts if counts is not None else {"scanned": 0, "skipped": 0, "failed": 0}
    built = []

    def make_source(scope, name, check_active):
        built.append((scope, name))
        return SimpleNamespace(cognee_sync_stats={"scanned": 1})

    retained = await ingestion.sync_scopes(
        "linear",
        CREDENTIAL,
        counts,
        scopes=scopes,
        dataset_name="linear_acme",
        make_source=make_source,
        **kwargs,
    )
    return retained, counts, built


@pytest.mark.asyncio
async def test_every_scope_is_ingested_into_its_own_named_table(remember):
    retained, counts, built = await _run(["t1", "t2"])

    assert [scope for scope, _ in built] == ["t1", "t2"]
    assert retained == {name for _, name in built}
    assert remember.await_count == 2
    assert counts["scanned"] == 2 and counts["failed"] == 0
    for call in remember.await_args_list:
        assert call.kwargs["write_disposition"] == "merge"
        assert call.kwargs["self_improvement"] is False


@pytest.mark.asyncio
async def test_a_failing_scope_is_counted_under_the_key_the_caller_picks_and_the_rest_continue(
    remember,
):
    remember.side_effect = [RuntimeError("boom"), SimpleNamespace(status="completed")]

    _, counts, _ = await _run(["t1", "t2"], classify_error=lambda error: "failed_custom")

    assert counts["failed"] == 1 and counts["failed_custom"] == 1
    assert remember.await_count == 2


@pytest.mark.asyncio
async def test_an_unclassified_failure_is_an_ingestion_failure(remember):
    remember.return_value = SimpleNamespace(status="errored")
    remember.side_effect = RuntimeError("boom")

    _, counts, _ = await _run(["t1"])

    assert counts["failed_ingestion"] == 1


@pytest.mark.asyncio
async def test_a_connection_that_goes_inactive_stops_the_loop_even_on_the_last_scope(remember):
    remember.side_effect = [SimpleNamespace(status="completed"), CredentialInactiveError()]

    with pytest.raises(CredentialInactiveError):
        await _run(["t1", "t2"])


@pytest.mark.asyncio
async def test_a_failure_that_names_the_linear_rate_limit_is_recorded_as_one(monkeypatch):
    recorded = AsyncMock()
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", recorded)

    async def sync_source(credential, counts):
        raise RuntimeError("Linear query LinearTeams failed: HTTP 400 RATELIMITED")

    with pytest.raises(RuntimeError):
        await ingestion.run_sync("linear", CREDENTIAL, sync_source)

    assert recorded.await_args.kwargs["counts"]["failed_rate_limit"] == 1
