"""The per-scope loop Drive and Linear share."""

import contextlib
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import CredentialInactiveError

CREDENTIAL = SimpleNamespace(
    id=UUID("11111111-1111-1111-1111-111111111111"),
    user_id=UUID("22222222-2222-2222-2222-222222222222"),
    provider="linear",
    workspace_id=None,
    provider_account_id="acct-1",
)


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
    """Not counted as a failed scope: the sync must stop and report the connection as gone."""
    remember.side_effect = [SimpleNamespace(status="completed"), CredentialInactiveError()]

    with pytest.raises(CredentialInactiveError):
        await _run(["t1", "t2"])


@pytest.mark.asyncio
async def test_a_sync_requested_while_one_runs_runs_once_more_after_it(monkeypatch):
    """A selection saved mid-sync is applied by one more run, not dropped."""
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", AsyncMock())
    runs = []

    async def sync_source(credential, counts):
        runs.append(credential)
        if len(runs) == 1:
            assert ingestion.request_rerun_if_running("linear", CREDENTIAL.provider_account_id)
            assert ingestion.request_rerun_if_running("linear", CREDENTIAL.provider_account_id)
        return "ok", counts

    await ingestion.run_sync("linear", CREDENTIAL, sync_source)

    # Requested twice during the first run: one more run applies both.
    assert len(runs) == 2
    assert not ingestion.request_rerun_if_running("linear", CREDENTIAL.provider_account_id)


@pytest.mark.asyncio
async def test_a_sync_requested_while_one_fails_still_runs_after_it(monkeypatch):
    """A selection saved during a sync that fails is applied, not dropped with the failure."""
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", AsyncMock())
    runs = []

    async def sync_source(credential, counts):
        runs.append(credential)
        if len(runs) == 1:
            assert ingestion.request_rerun_if_running("linear", CREDENTIAL.provider_account_id)
            raise RuntimeError("the provider is unreachable")
        return "ok", counts

    with contextlib.suppress(RuntimeError):
        await ingestion.run_sync("linear", CREDENTIAL, sync_source)

    assert len(runs) == 2


@pytest.mark.asyncio
async def test_a_rate_limit_error_is_recorded_as_one_whatever_its_text(monkeypatch):
    """Typed, not sniffed from the message: Linear's wording is not Google's."""
    recorded = AsyncMock()
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", recorded)

    async def sync_source(credential, counts):
        raise ingestion.RateLimitedError("slow down")

    with pytest.raises(ingestion.RateLimitedError):
        await ingestion.run_sync("linear", CREDENTIAL, sync_source)

    assert recorded.await_args.kwargs["counts"]["failed_rate_limit"] == 1


@pytest.mark.asyncio
async def test_a_failing_scope_is_logged_by_its_resource_name_not_its_id(remember, caplog):
    """A scope can be a channel or folder id; the hashed resource name names it in the log."""
    remember.side_effect = RuntimeError("boom")

    _, _, built = await _run(["C0SECRETCHAN"])

    assert "C0SECRETCHAN" not in caplog.text
    assert built[0][1] in caplog.text


@pytest.mark.asyncio
async def test_the_failure_log_call_carries_no_scope(remember, monkeypatch):
    """Read off the log call itself, so it holds whatever logging configuration the run has."""
    log = Mock()
    monkeypatch.setattr(ingestion, "logger", log)
    remember.side_effect = RuntimeError("boom")

    _, _, built = await _run(["C0SECRETCHAN"])

    (call,) = log.exception.call_args_list
    assert "C0SECRETCHAN" not in repr(call)
    assert built[0][1] in call.args


@pytest.mark.parametrize(
    ("counts", "gone_keys", "blocked"),
    [
        ({"failed": 0}, (), False),
        ({"failed": 1, "failed_ingestion": 1}, (), True),
        # A backfill cut short may not have replaced a legacy table yet.
        ({"failed": 1, "failed_budget": 1}, (), True),
        ({"failed": 2, "failed_team_not_found": 2}, ("failed_team_not_found",), False),
        (
            {"failed": 2, "failed_team_not_found": 1, "failed_rate_limit": 1},
            ("failed_team_not_found",),
            True,
        ),
    ],
)
def test_only_a_scope_gone_at_the_provider_does_not_block_retirement(counts, gone_keys, blocked):
    assert ingestion.retirement_blocked(counts, gone_keys=gone_keys) is blocked


@pytest.mark.asyncio
async def test_a_failure_followed_by_a_rerun_keeps_its_traceback_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    monkeypatch.setattr("cognee.modules.integrations.credentials.record_sync_result", AsyncMock())
    runs = []

    async def sync_source(credential, counts):
        runs.append(credential)
        if len(runs) == 1:
            ingestion.request_rerun_if_running("linear", CREDENTIAL.provider_account_id)
            raise RuntimeError("the provider is unreachable")
        return "ok", counts

    with caplog.at_level("WARNING"):
        await ingestion.run_sync("linear", CREDENTIAL, sync_source)

    rerun = [record for record in caplog.records if "running it again" in record.getMessage()]
    assert rerun and "the provider is unreachable" in rerun[0].getMessage()


def test_a_rate_limit_carries_the_wait_the_provider_named():
    assert ingestion.RateLimitedError("slow down", retry_after=30).retry_after == 30
    assert ingestion.RateLimitedError("slow down").retry_after is None
