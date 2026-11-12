"""Slack sync: guards per channel, one source per channel, retirement, and the tick.

The walk itself is ``slack_source``'s (tested next to it); here the source factory and
``remember`` are stand-ins, and the guard calls go to the in-memory Slack.
"""

import asyncio
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch
from uuid import UUID

import pytest
import pytest_asyncio

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.slack import sync as sync_module
from cognee.modules.integrations.slack.sync import sync_slack
from cognee.tasks.ingestion.connectors.slack import (
    SlackAPIError,
    SlackAuthError,
    SlackMissingScopeError,
    SlackRateLimitedError,
    SlackUnavailableError,
)
from cognee.tests.unit.tasks.connectors.fake_slack import TEAM, FakeSlack

INSTALLER = "U0INSTALLER"
SCOPES = "commands,channels:read,channels:history,groups:read,groups:history,users:read"


def make_credential(*channels, **overrides):
    metadata = {"installed_by_slack_user_id": INSTALLER, "selected_channels": list(channels)}
    metadata.update(overrides.pop("metadata", {}))
    return SimpleNamespace(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        provider="slack",
        workspace_id=None,
        provider_account_id=TEAM,
        user_id=UUID("22222222-2222-2222-2222-222222222222"),
        scopes=overrides.pop("scopes", SCOPES),
        provider_metadata=metadata,
        last_synced_at=overrides.pop("last_synced_at", None),
        sync_status=overrides.pop("sync_status", None),
        **overrides,
    )


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    """No connection is paused unless a test says so (the pause lives in the database)."""
    monkeypatch.setattr(sync_module, "is_paused", AsyncMock(return_value=False))


def workspace(*channels: str) -> FakeSlack:
    fake = FakeSlack()
    for channel in channels:
        fake.add_channel(channel, channel.lower())
        fake.members[channel] = ["U0SOMEONE", INSTALLER]
    return fake


@pytest.fixture
def sync_mocks():
    with ExitStack() as stack:
        mocks = SimpleNamespace(
            fake=workspace("C1", "C2", "C3"),
            source=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            remember=AsyncMock(return_value=SimpleNamespace(status="completed")),
            record=AsyncMock(),
            # Like the real one: every selected channel has a table, and the ones outside
            # ``retained`` are emptied and returned.
            retire=AsyncMock(
                side_effect=lambda provider, credential, dataset_name, retained: (
                    {
                        ingestion.resource_name(provider, credential, channel)
                        for channel in sync_module.selected_channels(credential)
                    }
                    - retained
                )
            ),
        )
        for patcher in (
            patch.object(
                ingestion,
                "require_active_credential",
                AsyncMock(side_effect=lambda credential: credential),
            ),
            patch.object(ingestion, "retire_resources", mocks.retire),
            patch.object(ingestion, "source_factory", return_value=mocks.source),
            patch.object(ingestion, "build_service", lambda provider, token: mocks.fake),
            patch(
                "cognee.modules.integrations.slack.adapter.access_token_for",
                AsyncMock(return_value="xoxb-token"),
            ),
            patch.object(
                import_module("cognee.api.v1.remember.remember"), "remember", mocks.remember
            ),
            patch("cognee.modules.integrations.credentials.record_sync_result", mocks.record),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        ):
            stack.enter_context(patcher)
        yield mocks


def wrapped(error: Exception) -> RuntimeError:
    """The source's error as remember() raises it: wrapped by dlt and ingestion."""
    outer = RuntimeError("dlt step failed")
    outer.__cause__ = error
    return outer


def recorded_counts(mocks) -> dict[str, int]:
    return mocks.record.await_args.kwargs["counts"]


@pytest.mark.asyncio
async def test_each_channel_is_its_own_source_in_the_workspace_dataset(sync_mocks):
    credential = make_credential("C2", "C1", "C2")
    await sync_slack(credential)

    assert [call.kwargs for call in sync_mocks.source.call_args_list] == [
        {
            "service": ANY,  # a fresh-token client per channel, see below
            "team_id": TEAM,
            "channel_id": channel,
            "resource_name": ingestion.resource_name("slack", credential, channel),
            "check_active": ANY,
            "changed_roots": set(),  # no thread was marked changed (G6)
            "excluded_messages": set(),  # nothing taken out of the workspace's memory
        }
        for channel in ("C1", "C2")
    ]
    dataset = ingestion.dataset_name_for_account("slack", "", TEAM)
    assert sync_module.dataset_name_for_credential(credential) == dataset
    for call in sync_mocks.remember.await_args_list:
        assert call.kwargs["dataset_name"] == dataset
        assert call.kwargs["write_disposition"] == "merge" and call.kwargs["user"] == "owner"
    assert sync_mocks.fake.methods().count("auth.test") == 1, "one auth.test per sync"


def synced_channels(mocks) -> list[str]:
    return [call.kwargs["channel_id"] for call in mocks.source.call_args_list]


def table(credential, channel: str) -> str:
    return ingestion.resource_name("slack", credential, channel)


@pytest.mark.asyncio
async def test_a_public_channel_syncs_without_the_installer_in_it(sync_mocks):
    """Any member of the workspace may read a public channel, the installer included."""
    sync_mocks.fake.members["C1"] = ["U0SOMEONE"]
    await sync_slack(make_credential("C1"))

    assert synced_channels(sync_mocks) == ["C1"]
    assert recorded_counts(sync_mocks)["failed"] == 0


@pytest.mark.asyncio
async def test_a_private_channel_the_installer_left_is_retired(sync_mocks):
    """The installer reads the dataset: what Slack hides from them leaves their memory."""
    fake = sync_mocks.fake
    fake.channels["C2"]["is_private"] = True
    fake.members["C2"] = ["U0SOMEONE"]
    credential = make_credential("C1", "C2")

    await sync_slack(credential)

    assert synced_channels(sync_mocks) == ["C1"]
    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 0 and counts["channels_retired"] == 1
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"
    sync_mocks.retire.assert_awaited_once_with("slack", credential, ANY, {table(credential, "C1")})


@pytest.mark.asyncio
async def test_a_deleted_channel_or_one_the_app_left_keeps_its_data_and_is_reported(sync_mocks):
    """Unavailable channels fail on every sync but never block the others' retirement."""
    sync_mocks.fake.channels["C1"]["is_member"] = False  # the app was removed
    credential = make_credential("C1", "C3", "C9")  # C9 was deleted

    await sync_slack(credential)

    assert synced_channels(sync_mocks) == ["C3"]
    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 2 and counts["failed_channel_unavailable"] == 2
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"
    sync_mocks.retire.assert_awaited_once_with(
        "slack", credential, ANY, {table(credential, c) for c in ("C1", "C3", "C9")}
    )


@pytest.mark.asyncio
async def test_the_owner_can_forget_unavailable_channels(sync_mocks):
    sync_mocks.fake.channels["C1"]["is_member"] = False
    credential = make_credential(
        "C1", "C3", "C9", metadata={sync_module.FORGET_UNAVAILABLE_KEY: True}
    )

    await sync_slack(credential)

    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 0 and counts["channels_retired"] == 2
    sync_mocks.retire.assert_awaited_once_with("slack", credential, ANY, {table(credential, "C3")})


@pytest.mark.asyncio
async def test_an_archived_channel_keeps_its_data_unread_until_it_is_unarchived(sync_mocks):
    """Slack keeps an archived channel's history and its members still read it. Whether the
    app stays a member is not said, so membership is not asked of it."""
    fake = sync_mocks.fake
    fake.channels["C2"].update(is_archived=True, is_member=False)
    credential = make_credential("C1", "C2", "C3")

    await sync_slack(credential)

    assert synced_channels(sync_mocks) == ["C1", "C3"]
    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 0 and counts["channels_archived"] == 1
    assert "channels_retired" not in counts
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"
    sync_mocks.retire.assert_awaited_once_with(
        "slack", credential, ANY, {table(credential, c) for c in ("C1", "C2", "C3")}
    )

    fake.channels["C2"].update(is_archived=False, is_member=True)
    sync_mocks.source.reset_mock()
    await sync_slack(credential)

    assert synced_channels(sync_mocks) == ["C1", "C2", "C3"]
    assert "channels_archived" not in recorded_counts(sync_mocks)


@pytest.mark.asyncio
async def test_an_archived_channel_does_not_hold_back_a_deselected_one(sync_mocks):
    sync_mocks.fake.channels["C2"]["is_archived"] = True
    credential = make_credential("C2")  # C1 and C3 were deselected

    await sync_slack(credential)

    sync_mocks.source.assert_not_called()
    sync_mocks.retire.assert_awaited_once_with("slack", credential, ANY, {table(credential, "C2")})


@pytest.mark.asyncio
async def test_an_archived_private_channel_the_installer_left_is_retired(sync_mocks):
    fake = sync_mocks.fake
    fake.channels["C2"].update(is_archived=True, is_private=True)
    fake.members["C2"] = ["U0SOMEONE"]
    credential = make_credential("C1", "C2")

    await sync_slack(credential)

    counts = recorded_counts(sync_mocks)
    assert counts["channels_retired"] == 1 and "channels_archived" not in counts
    sync_mocks.retire.assert_awaited_once_with("slack", credential, ANY, {table(credential, "C1")})


@pytest.mark.asyncio
async def test_a_deleted_workspace_ends_the_connection_and_keeps_the_data(sync_mocks):
    """For a bot token ``account_inactive`` means a deleted workspace or bot user: the
    connection is ended like an uninstall instead of failing every six hours."""
    from cognee.modules.integrations.credentials import CredentialInactiveError

    sync_mocks.fake.fail("auth.test", SlackAuthError("auth.test", "account_inactive"))
    revoke = AsyncMock(return_value=True)

    with (
        patch("cognee.modules.integrations.slack.persistence.revoke_by_team", revoke),
        pytest.raises(CredentialInactiveError),
    ):
        await sync_slack(make_credential("C1", "C2"))

    revoke.assert_awaited_once_with(TEAM)
    sync_mocks.source.assert_not_called()
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["invalid_auth", "token_revoked", "token_expired", "http_401"])
async def test_any_other_auth_error_leaves_the_connection_as_it_is(sync_mocks, code):
    """``invalid_auth`` may be an IP allowlist or a configuration; only Slack's answer for a
    deleted workspace ends a connection here."""
    sync_mocks.fake.fail("auth.test", SlackAuthError("auth.test", code))
    revoke = AsyncMock(return_value=True)

    with (
        patch("cognee.modules.integrations.slack.persistence.revoke_by_team", revoke),
        pytest.raises(SlackAuthError),
    ):
        await sync_slack(make_credential("C1"))

    revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unclear_answer_keeps_every_channel_and_blocks_retirement(sync_mocks):
    """A member list Slack does not return is not proof the installer left."""
    fake = sync_mocks.fake
    fake.channels["C2"]["is_private"] = True
    fake.fail("conversations.members", SlackAPIError("conversations.members", "internal_error"))

    await sync_slack(make_credential("C1", "C2"))

    assert synced_channels(sync_mocks) == ["C1"]
    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 1 and counts["failed_channel_access"] == 1
    assert "channels_retired" not in counts
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_command_allowlist_never_narrows_the_sync(sync_mocks):
    # The allowlist decides where Cognee's commands work; the selection decides the sync.
    credential = make_credential("C1", "C2", metadata={"allowed_channel_ids": ["C2"]})
    await sync_slack(credential)

    assert synced_channels(sync_mocks) == ["C1", "C2"]
    assert recorded_counts(sync_mocks).get("channels_retired", 0) == 0
    sync_mocks.retire.assert_awaited_once_with(
        "slack", credential, ANY, {table(credential, "C1"), table(credential, "C2")}
    )


@pytest.mark.asyncio
async def test_a_deselected_channel_is_retired_after_a_clean_sync(sync_mocks):
    credential = make_credential("C1", "C3")
    await sync_slack(credential)

    sync_mocks.retire.assert_awaited_once_with(
        "slack",
        credential,
        ingestion.dataset_name_for_account("slack", "", TEAM),
        {ingestion.resource_name("slack", credential, c) for c in ("C1", "C3")},
    )


@pytest.mark.asyncio
async def test_a_channel_its_own_walk_cut_short_does_not_hold_back_a_deselected_one(sync_mocks):
    """Under Tier 1 nearly every sync cuts a walk short, and the cut counted as a failure
    blocked every retirement: a deselected channel stayed in memory on 700 of 714 syncs
    (measured). A cut channel was stored and stays retained; Slack's tables are one per
    channel, so it holds nothing of the deselected one (Goran, 2026-11-10)."""

    def source(**kwargs):
        stats = {"failed": 1, "failed_rate_limit": 1} if kwargs["channel_id"] == "C1" else {}
        return SimpleNamespace(cognee_sync_stats=stats, **kwargs)

    sync_mocks.source.side_effect = source
    credential = make_credential("C1", "C3")

    await sync_slack(credential)

    assert recorded_counts(sync_mocks)["failed_rate_limit"] == 1
    sync_mocks.retire.assert_awaited_once_with(
        "slack",
        credential,
        ingestion.dataset_name_for_account("slack", "", TEAM),
        {ingestion.resource_name("slack", credential, c) for c in ("C1", "C3")},
    )


@pytest.mark.asyncio
async def test_a_retire_verdict_is_reported_only_when_the_retirement_runs(sync_mocks):
    # The installer left private C2, so its verdict is "retire"; C1's walk is rate limited,
    # which blocks every retirement on this sync. C2's documents stay, and so must the report.
    sync_mocks.fake.channels["C2"]["is_private"] = True
    sync_mocks.fake.members["C2"] = ["U0SOMEONE"]
    sync_mocks.remember.side_effect = wrapped(SlackRateLimitedError("conversations.history"))

    await sync_slack(make_credential("C1", "C2"))

    counts = recorded_counts(sync_mocks)
    assert counts["failed_rate_limit"] == 1
    sync_mocks.retire.assert_not_awaited()
    assert counts.get("channels_retired", 0) == 0


@pytest.mark.parametrize(
    "answer",
    [{}, {"channel": None}, {"channel": {"name": "general"}}, {"channel": {"id": "C9"}}],
    ids=["no_channel", "null_channel", "no_id", "another_channel"],
)
def test_an_answer_that_does_not_describe_the_channel_never_retires_it(answer):
    """An uncertain answer never retires anything: only a clear one may."""
    client = SimpleNamespace(call=lambda method, **params: {"ok": True, **answer})

    with pytest.raises(sync_module.SlackChannelAccessError):
        sync_module.channel_verdict(client, "C1", INSTALLER)


@pytest.mark.asyncio
async def test_a_failed_channel_source_keeps_the_deselected_tables(sync_mocks):
    sync_mocks.remember.side_effect = [
        SimpleNamespace(status="completed"),
        wrapped(SlackRateLimitedError("conversations.history")),
    ]
    await sync_slack(make_credential("C1", "C2"))

    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 1 and counts["failed_rate_limit"] == 1
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [[], None])
async def test_an_empty_selection_syncs_nothing_and_retires_everything(sync_mocks, selection):
    credential = make_credential()
    credential.provider_metadata["selected_channels"] = selection
    await sync_slack(credential)

    sync_mocks.source.assert_not_called()
    assert sync_mocks.fake.calls == []
    sync_mocks.retire.assert_awaited_once_with(
        "slack", credential, ingestion.dataset_name_for_account("slack", "", TEAM), set()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", ["commands,channels:history", None])
async def test_a_connection_without_users_read_stops_with_the_reconnect(sync_mocks, scopes):
    with pytest.raises(SlackMissingScopeError, match="Reconnect Slack to grant users:read"):
        await sync_slack(make_credential("C1", "C2", scopes=scopes))

    sync_mocks.source.assert_not_called()
    assert sync_mocks.fake.calls == [], "nothing is read from Slack"
    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 1 and counts["failed_missing_scope"] == 1


@pytest.mark.asyncio
async def test_a_connection_that_does_not_name_its_installer_is_refused(sync_mocks):
    credential = make_credential("C1", metadata={"installed_by_slack_user_id": None})
    with pytest.raises(sync_module.SlackSyncError, match="Reconnect Slack"):
        await sync_slack(credential)
    sync_mocks.source.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["token_revoked", "invalid_auth"])
async def test_a_token_slack_no_longer_accepts_tells_the_owner_to_reconnect(sync_mocks, code):
    """G76: a token Slack rejects at ``auth.test`` was raised before any channel, so no
    ``failed_auth`` was counted and the card said "Refresh to retry", which never helps."""
    from cognee.tasks.ingestion.connectors.slack import SlackAuthError

    sync_mocks.fake.fail("auth.test", SlackAuthError("auth.test", code))
    with pytest.raises(SlackAuthError):
        await sync_slack(make_credential("C1"))

    counts = recorded_counts(sync_mocks)
    assert counts["failed_auth"] == 1
    sync_mocks.source.assert_not_called()


@pytest.mark.asyncio
async def test_a_token_rejected_after_auth_test_also_tells_the_owner_to_reconnect(sync_mocks):
    """G85: a token Slack rejects on a channel's ``conversations.info``, after ``auth.test``
    passed, re-raised uncounted, and the card said "Refresh to retry"."""
    from cognee.tasks.ingestion.connectors.slack import SlackAuthError

    sync_mocks.fake.fail(
        "conversations.info", SlackAuthError("conversations.info", "token_revoked")
    )
    with pytest.raises(SlackAuthError):
        await sync_slack(make_credential("C1"))

    assert recorded_counts(sync_mocks)["failed_auth"] == 1


@pytest.mark.asyncio
async def test_a_connection_without_a_usable_token_asks_for_a_reconnect(sync_mocks):
    """G85: no access token stored, or a refresh Slack refuses, can only be fixed by
    connecting again; it was raised before any count."""
    from cognee.modules.integrations.slack import adapter

    with (
        patch.object(
            adapter,
            "access_token_for",
            AsyncMock(side_effect=adapter.SlackReconnectError("holds no access token")),
        ),
        pytest.raises(adapter.SlackReconnectError),
    ):
        await sync_slack(make_credential("C1"))

    assert recorded_counts(sync_mocks)[sync_module.RECONNECT_KEY] == 1


@pytest.mark.asyncio
async def test_a_connection_that_no_longer_matches_its_workspace_asks_for_a_reconnect(
    sync_mocks,
):
    """G76: a connection without its installer, or a token of another workspace, says
    "Reconnect Slack" in the error, and the owner's card has to say it too."""
    credential = make_credential("C1", metadata={"installed_by_slack_user_id": None})
    with pytest.raises(sync_module.SlackSyncError):
        await sync_slack(credential)
    assert recorded_counts(sync_mocks)[sync_module.RECONNECT_KEY] == 1


@pytest.mark.asyncio
async def test_a_token_of_another_workspace_is_refused_before_any_channel(sync_mocks):
    credential = make_credential("C1")
    credential.provider_account_id = "T0OTHER"
    with pytest.raises(sync_module.SlackSyncError, match="does not belong"):
        await sync_slack(credential)
    assert sync_mocks.fake.methods() == ["auth.test"]


@pytest.mark.asyncio
async def test_a_dead_token_during_the_guards_stops_the_whole_sync(sync_mocks):
    sync_mocks.fake.fail(
        "conversations.info", SlackAuthError("conversations.info", "token_revoked")
    )
    with pytest.raises(SlackAuthError):
        await sync_slack(make_credential("C1", "C2"))
    sync_mocks.source.assert_not_called()


@pytest.mark.asyncio
async def test_a_selection_that_is_not_channel_ids_is_refused(sync_mocks):
    with pytest.raises(ValueError, match="channel ids"):
        await sync_slack(make_credential("general"))


@pytest.mark.asyncio
async def test_a_connection_revoked_during_the_guards_stops_before_the_next_channel(sync_mocks):
    from cognee.modules.integrations.credentials import CredentialInactiveError

    checks = {"n": 0}

    async def require(credential):
        checks["n"] += 1
        # run_sync, the re-read after the sync lock, the first channel's guard, the second's
        if checks["n"] == 4:
            raise CredentialInactiveError()
        return credential

    with (
        patch.object(ingestion, "require_active_credential", AsyncMock(side_effect=require)),
        pytest.raises(CredentialInactiveError),
    ):
        await sync_slack(make_credential("C1", "C2"))

    infos = [p["channel"] for m, p in sync_mocks.fake.calls if m == "conversations.info"]
    assert infos == ["C1"]


@pytest.mark.asyncio
async def test_a_workspace_address_outside_slack_com_is_refused(sync_mocks):
    sync_mocks.fake.url = "https://acme.slack.com.evil.example/"
    with pytest.raises(sync_module.SlackSyncError, match="does not belong"):
        await sync_slack(make_credential("C1"))
    sync_mocks.source.assert_not_called()


@pytest.mark.asyncio
async def test_a_group_dm_is_never_synced(sync_mocks):
    """Legacy group DM ids start with G, like private channels, and pass the selection check."""
    sync_mocks.fake.add_channel("G0GROUPDM", "mpdm-ana--ben-1")
    sync_mocks.fake.channels["G0GROUPDM"]["is_mpim"] = True
    sync_mocks.fake.members["G0GROUPDM"] = [INSTALLER]

    await sync_slack(make_credential("C1", "G0GROUPDM"))

    assert [call.kwargs["channel_id"] for call in sync_mocks.source.call_args_list] == ["C1"]
    counts = recorded_counts(sync_mocks)
    assert counts["failed"] == 0 and counts["channels_retired"] == 1


@pytest.mark.asyncio
async def test_each_channel_takes_a_fresh_token_when_its_walk_starts(sync_mocks):
    tokens = AsyncMock(side_effect=["guard-token", "c1-token", "c2-token"])
    built = []

    def build_service(provider, token):
        built.append(token)
        return sync_mocks.fake

    with (
        patch("cognee.modules.integrations.slack.adapter.access_token_for", tokens),
        patch.object(ingestion, "build_service", build_service),
    ):
        await sync_slack(make_credential("C1", "C2"))
        services = [call.kwargs["service"] for call in sync_mocks.source.call_args_list]
        assert built == ["guard-token"], "no channel token before its walk starts"
        for service in services:
            # The walk runs on dlt's worker thread.
            await asyncio.to_thread(service.call, "auth.test")
            await asyncio.to_thread(service.call, "auth.test")

    assert built == ["guard-token", "c1-token", "c2-token"], "one token per channel"


@pytest.mark.parametrize(
    ("error", "key"),
    [
        (SlackMissingScopeError("users.info", "users:read"), "failed_missing_scope"),
        (SlackAuthError("auth.test", "token_revoked"), "failed_auth"),
        (SlackRateLimitedError("conversations.history"), "failed_rate_limit"),
        # Slack down before the walk read anything: resumed like a cut, not parked for hours.
        (
            SlackUnavailableError("conversations.history", "service_unavailable"),
            "failed_unavailable",
        ),
        (SlackAPIError("conversations.history", "not_in_channel"), "failed_ingestion"),
    ],
)
def test_a_wrapped_source_error_is_counted_under_its_own_key(error, key):
    try:
        try:
            raise error
        except Exception as inner:
            raise RuntimeError("dlt step failed") from inner
    except RuntimeError as outer:
        assert sync_module._classify_error(outer) == key


@pytest.mark.asyncio
async def test_a_channel_that_fails_before_any_row_is_counted_once(monkeypatch):
    """The host counts a raised channel under its key; the source must not count it again."""
    from cognee.tasks.ingestion.connectors.slack import slack_source

    fake = workspace("C1")
    fake.fail("conversations.history", SlackUnavailableError("conversations.history", "http_503"))

    async def remember(source, **kwargs):
        await asyncio.to_thread(list, source)  # what dlt does, on its worker thread

    monkeypatch.setattr(import_module("cognee.api.v1.remember.remember"), "remember", remember)
    monkeypatch.setattr("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
    monkeypatch.setattr(
        ingestion, "require_active_credential", AsyncMock(side_effect=lambda credential: credential)
    )
    counts = {"scanned": 0, "skipped": 0, "failed": 0}

    await ingestion.sync_scopes(
        "slack",
        make_credential("C1"),
        counts,
        scopes=["C1"],
        dataset_name="slack_test",
        make_source=lambda channel, name, check_active: slack_source(
            service=fake,
            team_id=TEAM,
            channel_id=channel,
            resource_name=name,
            check_active=check_active,
        ),
        classify_error=sync_module._classify_error,
    )

    assert counts["failed"] == 1
    assert counts["failed_unavailable"] == 1


# --- the tick ---


@pytest.mark.asyncio
async def test_taking_a_message_out_records_it_marks_its_root_and_starts_a_sync(monkeypatch):
    credential = make_credential("C1")
    credential.status = "active"
    writes = []

    async def change(provider, account, compute):
        patch = compute(dict(credential.provider_metadata))
        writes.append(patch)
        credential.provider_metadata = {**credential.provider_metadata, **(patch or {})}
        return credential

    started = []
    monkeypatch.setattr("cognee.modules.integrations.credentials.change_provider_metadata", change)
    monkeypatch.setattr(
        "cognee.modules.integrations.credentials.get_credential_by_account",
        AsyncMock(side_effect=lambda provider, account: credential),
    )

    async def start(credential):
        started.append(credential)

    monkeypatch.setattr(sync_module, "start_sync", start)

    taken = await sync_module.exclude_message(
        credential, channel_id="C1", message_ts="1.2", root_ts="1.1"
    )
    assert taken is True
    meta = credential.provider_metadata
    assert meta[sync_module.EXCLUDED_MESSAGES_KEY] == {"C1": ["1.2"]}
    assert list(meta[sync_module.CHANGED_THREADS_KEY]["C1"]) == ["1.1"]
    assert started == [credential]

    assert not await sync_module.exclude_message(
        credential, channel_id="C9", message_ts="9.9", root_ts="9.9"
    ), "a channel that is not synced has nothing to take out"


def test_a_connection_is_due_once_its_interval_has_passed():
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    assert sync_module.is_due(make_credential("C1"), now)
    assert not sync_module.is_due(make_credential(), now), "nothing selected"
    recent = make_credential("C1", last_synced_at=now - timedelta(hours=5))
    assert not sync_module.is_due(recent, now)
    # SQLite gives the timestamp back without a timezone; it was written as UTC.
    naive = make_credential("C1", last_synced_at=(now - timedelta(hours=6)).replace(tzinfo=None))
    assert sync_module.is_due(naive, now)


def test_a_connection_whose_sync_runs_is_not_due():
    credential = make_credential("C1")
    ingestion._running_syncs.add(("slack", TEAM))
    try:
        assert not sync_module.is_due(credential, datetime.now(timezone.utc))
    finally:
        ingestion._running_syncs.discard(("slack", TEAM))


def credentials_table(rows: list):
    """A relational engine whose credential query answers the current ``rows``."""

    class _Session:
        async def __aenter__(self):
            result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: list(rows)))
            return SimpleNamespace(execute=AsyncMock(return_value=result))

        async def __aexit__(self, *exc):
            return False

    return SimpleNamespace(get_async_session=_Session)


@pytest.mark.asyncio
async def test_a_workspace_that_becomes_due_starts_while_another_sync_still_runs():
    big, late = make_credential("C1"), make_credential("C2")
    late.provider_account_id = "T0LATE"
    rows = [big]
    late_started = asyncio.Event()

    async def fake_sync(credential):
        if credential is late:
            late_started.set()
            return
        ingestion._running_syncs.add(("slack", TEAM))
        try:
            await late_started.wait()  # the long first sync ends only after the other starts
        finally:
            ingestion._running_syncs.discard(("slack", TEAM))

    with (
        patch.object(sync_module, "get_relational_engine", return_value=credentials_table(rows)),
        patch.object(sync_module, "sync_slack", fake_sync),
        patch.object(sync_module, "_STARTUP_DELAY_SECONDS", 0),
        patch.object(sync_module, "_TICK_SECONDS", 0.01),
    ):
        worker = asyncio.create_task(sync_module._worker())
        await asyncio.sleep(0.05)
        rows.append(late)
        try:
            await asyncio.wait_for(late_started.wait(), 2)
        finally:
            worker.cancel()
            await asyncio.gather(worker, *sync_module._scheduled, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_failed_scheduled_sync_is_logged_and_stops_no_other(caplog):
    first, second = make_credential("C1"), make_credential("C2")
    second.provider_account_id = "T0SECOND"
    done = []

    async def fake_sync(credential):
        if credential is first:
            raise RuntimeError("provider down")
        done.append(credential.provider_account_id)

    with (
        patch.object(
            sync_module, "get_relational_engine", return_value=credentials_table([first, second])
        ),
        patch.object(sync_module, "sync_slack", fake_sync),
    ):
        assert await sync_module.sync_due_workspaces() == 2
        await asyncio.gather(*sync_module._scheduled, return_exceptions=True)
        await asyncio.sleep(0)  # let the done callbacks run

    assert done == ["T0SECOND"]
    assert not sync_module._scheduled
    assert f"workspace {TEAM} failed" in caplog.text


@pytest.mark.asyncio
async def test_a_paused_connection_is_not_started_by_the_tick(monkeypatch):
    """G62: a pause (SDK-1434) holds the Slack tick, as it holds every other sync path."""
    paused, running = make_credential("C1"), make_credential("C2")
    running.provider_account_id = "T0SECOND"
    started = []

    async def fake_sync(credential):
        started.append(credential.provider_account_id)

    monkeypatch.setattr(
        sync_module, "is_paused", AsyncMock(side_effect=lambda credential: credential is paused)
    )
    with (
        patch.object(
            sync_module, "get_relational_engine", return_value=credentials_table([paused, running])
        ),
        patch.object(sync_module, "sync_slack", fake_sync),
    ):
        assert await sync_module.sync_due_workspaces() == 1
        await asyncio.gather(*sync_module._scheduled, return_exceptions=True)

    assert started == ["T0SECOND"]


@pytest.mark.asyncio
async def test_a_message_taken_out_of_a_paused_connection_waits_for_the_resume(monkeypatch):
    spawned = []
    monkeypatch.setattr(sync_module, "is_paused", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "cognee.modules.integrations.background.spawn_background",
        lambda coro, **kwargs: (spawned.append(kwargs), coro.close()),
    )

    await sync_module.start_sync(make_credential("C1"))

    assert spawned == [], "recorded, and applied by the first sync after the resume"


@asynccontextmanager
async def no_progress(provider, credential, **_kwargs):
    yield SimpleNamespace(report=AsyncMock(), finish=AsyncMock())


@pytest.mark.asyncio
async def test_two_workspaces_due_at_once_sync_side_by_side():
    """Each workspace stages into its own dlt dataset, so nothing makes one wait for the
    other. A queued one counted as running, and its owner's disconnect got a 409 until the
    sync ahead finished (second review of #6683)."""
    first, second = make_credential("C1"), make_credential("C2")
    second.provider_account_id = "T0SECOND"
    both_syncing = asyncio.Event()
    syncing: set[str] = set()

    async def walk(credential, counts):
        syncing.add(credential.provider_account_id)
        if len(syncing) == 2:
            both_syncing.set()
        await both_syncing.wait()
        return sync_module.SYNC_STATUS_OK, counts

    with (
        patch.object(
            sync_module, "get_relational_engine", return_value=credentials_table([first, second])
        ),
        patch.object(sync_module, "_sync_channels", walk),
        patch.object(ingestion, "require_active_credential", AsyncMock(side_effect=lambda c: c)),
        patch("cognee.modules.integrations.credentials.record_sync_result", AsyncMock()),
        patch("cognee.modules.integrations.sync_progress.track_sync", no_progress),
        patch("cognee.modules.integrations.sync_progress.report_sync_progress", AsyncMock()),
    ):
        assert await sync_module.sync_due_workspaces() == 2
        await asyncio.wait_for(asyncio.gather(*sync_module._scheduled), 2)

    assert syncing == {TEAM, "T0SECOND"}


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_the_scheduled_syncs(monkeypatch):
    monkeypatch.setattr(sync_module.slack_settings, "client_id", "")
    cancelled = asyncio.Event()

    async def long_sync():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with sync_module.slack_sync_lifespan(None):
        task = asyncio.create_task(long_sync())
        sync_module._scheduled.add(task)
        task.add_done_callback(sync_module._scheduled_finished)
        await asyncio.sleep(0)

    assert cancelled.is_set() and task.done()


def test_a_run_cut_short_resumes_after_ten_minutes_and_other_failures_wait():
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    eleven_minutes_ago = now - timedelta(minutes=11)

    def degraded(**counts):
        return make_credential(
            "C1",
            last_synced_at=eleven_minutes_ago,
            sync_status="degraded",
            metadata={"last_sync_counts": {"failed": 1, **counts}},
        )

    assert sync_module.is_due(degraded(failed_budget=1), now)
    assert sync_module.is_due(degraded(failed_rate_limit=1), now)
    assert sync_module.is_due(degraded(failed_unavailable=1), now)
    assert not sync_module.is_due(degraded(failed_channel_access=1), now)
    nine_minutes_ago = degraded(failed_budget=1)
    nine_minutes_ago.last_synced_at = now - timedelta(minutes=9)
    assert not sync_module.is_due(nine_minutes_ago, now)
    # A clean run that happens to carry an old count waits the full interval.
    clean = degraded(failed_budget=1)
    clean.sync_status = "ok"
    assert not sync_module.is_due(clean, now)


def test_a_channel_stuck_on_one_day_waits_the_interval_while_a_real_cut_resumes():
    """G56, option 1: a run that cannot move past one day would hold the staging lock for
    its 600 s every ten minutes for nothing; it waits the interval. Another channel that
    was cut and moved still resumes after ten minutes."""
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)

    def degraded(**counts):
        return make_credential(
            "C1",
            "C2",
            last_synced_at=now - timedelta(minutes=11),
            sync_status="degraded",
            metadata={"last_sync_counts": {"failed": 1, **counts}},
        )

    stuck = degraded(failed_budget=1, failed_day_too_large=1)
    assert not sync_module.is_due(stuck, now)
    stuck.last_synced_at = now - timedelta(hours=6)
    assert sync_module.is_due(stuck, now)
    assert sync_module.is_due(
        degraded(failed_budget=1, failed_rate_limit=1, failed_day_too_large=1), now
    )


@pytest.mark.asyncio
async def test_the_worker_does_not_start_without_slack(monkeypatch):
    monkeypatch.setattr(sync_module.slack_settings, "client_id", "")
    with patch.object(sync_module.asyncio, "create_task") as create_task:
        async with sync_module.slack_sync_lifespan(None):
            pass
    create_task.assert_not_called()


@pytest.mark.asyncio
async def test_with_the_history_sync_off_the_worker_runs_only_the_uninstall_cleanup(monkeypatch):
    """Remembered copies exist without the history sync, so their uninstall deletion
    (Slack's Developer Policy) runs whatever the sync setting says."""
    monkeypatch.setattr(sync_module.slack_settings, "history_sync_enabled", False)
    sync_due = AsyncMock()
    cleanup = AsyncMock()
    monkeypatch.setattr(sync_module, "sync_due_workspaces", sync_due)
    monkeypatch.setattr(sync_module, "forget_uninstalled_workspaces", cleanup)
    # The startup delay passes, then the first tick's sleep ends the loop.
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError()])
    monkeypatch.setattr(sync_module.asyncio, "sleep", sleep)

    with pytest.raises(asyncio.CancelledError):
        await sync_module._worker()

    sync_due.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_channel_whose_run_failed_keeps_its_thread_marks(sync_mocks):
    """The walker drops a mark once it reads the thread, but a run that then fails is not
    stored: only a channel whose run was stored clears its marks (G57)."""

    def reading_source(**kwargs):
        kwargs["changed_roots"].clear()  # the walker read every marked thread
        return SimpleNamespace(**kwargs)

    async def remember(source, **kwargs):
        if source.channel_id == "C1":
            raise wrapped(SlackAPIError("conversations.history", "internal_error"))
        return SimpleNamespace(status="completed")

    sync_mocks.source.side_effect = reading_source
    sync_mocks.remember.side_effect = remember
    marks = {"C1": ["1.000100"], "C2": ["2.000100"]}
    credential = make_credential("C1", "C2", metadata={"changed_threads": marks})
    with patch.object(sync_module, "_clear_read_threads", AsyncMock()) as clear:
        await sync_slack(credential)

    assert clear.await_args.args[1] == {"C2": {"2.000100": ""}}


@pytest_asyncio.fixture
async def credential_file(tmp_path, monkeypatch):
    """The credential store on a SQLite file: each session has its own connection, as in a
    deployment (an in-memory database shares one, so a rollback undoes another's write)."""
    import base64

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from cognee.modules.integrations import credentials as store
    from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'credentials.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(IntegrationCredential.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        store, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    yield store
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_marks_and_an_uninstall_on_one_connection_are_all_kept(
    credential_file,
):
    """G60: every writer of ``provider_metadata`` read the row, merged in Python and wrote it
    back, so of writes that overlapped only the last one stayed: a lost mark leaves an edit
    out of memory, a lost ``uninstalled_at`` cancels the 14-day deletion."""
    from uuid import uuid4

    store = credential_file

    await store.upsert_credential(
        provider="slack",
        user_id=uuid4(),
        provider_account_id=TEAM,
        token_payload={"access_token": "xoxb"},
        provider_metadata={sync_module.SELECTION_KEY: ["C1"]},
    )
    roots = [f"17000000{i:02d}.000100" for i in range(10)]

    await asyncio.gather(
        *(sync_module.mark_thread_changed(TEAM, "C1", root) for root in roots),
        store.update_provider_metadata(
            "slack", TEAM, {sync_module.UNINSTALLED_AT_KEY: "2026-11-10T00:00:00+00:00"}
        ),
    )

    metadata = (await store.get_credential_by_account("slack", TEAM)).provider_metadata
    assert sorted(metadata[sync_module.CHANGED_THREADS_KEY]["C1"]) == roots
    assert metadata[sync_module.UNINSTALLED_AT_KEY] == "2026-11-10T00:00:00+00:00"


@pytest.mark.asyncio
async def test_a_thread_changed_again_while_a_sync_reads_it_stays_marked(
    credential_file, sync_mocks
):
    """G72: a thread marked, read by a running sync, then edited again (or a message in it
    taken out) before that sync ended: the second mark found the first and wrote nothing,
    and the sync then cleared it, so the second change was never read."""
    store = credential_file
    root = "1700000000.000100"
    credential = make_credential("C1")
    await store.upsert_credential(
        provider="slack",
        user_id=credential.user_id,
        provider_account_id=TEAM,
        token_payload={"access_token": "xoxb"},
        scopes=SCOPES,
        provider_metadata=credential.provider_metadata,
    )
    await sync_module.mark_thread_changed(TEAM, "C1", root)
    credential = await store.get_credential_by_account("slack", TEAM)

    async def read_then_edited_again(source, **_kwargs):
        source.changed_roots.discard(root)  # the walk read the thread
        await sync_module.mark_thread_changed(TEAM, "C1", root)  # edited again meanwhile
        return SimpleNamespace(status="completed")

    sync_mocks.remember.side_effect = read_then_edited_again
    await sync_slack(credential)

    marks = (await store.get_credential_by_account("slack", TEAM)).provider_metadata
    assert root in (marks.get(sync_module.CHANGED_THREADS_KEY) or {}).get("C1", {})


@pytest.mark.parametrize("wrapped_in_dlt", [False, True])
def test_a_refused_refresh_inside_a_channel_walk_is_a_reconnect(wrapped_in_dlt):
    """G91: each channel's walk takes its own token; a refresh Slack refuses there raises
    ``SlackReconnectError`` inside dlt, and it was counted as ``failed_ingestion``, so the
    card said "Refresh to retry"."""
    from cognee.modules.integrations.slack.adapter import SlackReconnectError

    error = SlackReconnectError("Slack token refresh failed: invalid_refresh_token")
    raised = wrapped(error) if wrapped_in_dlt else error
    assert sync_module._classify_error(raised) == sync_module.RECONNECT_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "reconnect"),
    [("invalid_refresh_token", True), ("token_revoked", True), ("internal_error", False)],
)
async def test_only_a_refresh_token_slack_rejects_asks_for_a_reconnect(code, reconnect):
    """G95: a refresh Slack answers ``ok: false`` for a passing reason (``internal_error``)
    was reported as reconnect, which does not help; only a rejected refresh token is."""
    from cognee.modules.integrations.slack import adapter

    class Response:
        status = 200

        async def json(self):
            return {"ok": False, "error": code}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    class Session:
        def post(self, *_args, **_kwargs):
            return Response()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    credential = make_credential("C1")
    with (
        patch.object(adapter, "client_session", lambda *_a, **_k: Session(), create=True),
        patch.object(adapter, "decrypt_token_payload", return_value={"refresh_token": "r"}),
        patch.object(adapter, "require", lambda name: "x", create=True),
        pytest.raises(RuntimeError) as raised,
    ):
        await adapter.SlackIntegration().refresh(credential)
    assert isinstance(raised.value, adapter.SlackReconnectError) is reconnect
