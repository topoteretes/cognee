"""Unit tests for cognee.modules.integrations.slack.adapter.SlackIntegration.

Covers the field-extraction logic that used to live in persistence.py's
save_installation() — team vs. enterprise (Grid) routing id, secret/metadata
split, and expires_in -> token_expires_at conversion.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.slack.adapter import SlackIntegration

integration = SlackIntegration()


def test_extracts_team_id_as_account_id():
    installation = integration.parse_installation(
        {"team": {"id": "T123", "name": "Acme"}, "access_token": "xoxb-secret"}
    )
    assert installation.provider_account_id == "T123"
    assert installation.account_label == "Acme"


def test_falls_back_to_enterprise_id_for_grid_installs():
    installation = integration.parse_installation(
        {"enterprise": {"id": "E456"}, "access_token": "xoxb-secret"}
    )
    assert installation.provider_account_id == "E456"


def test_missing_team_and_enterprise_raises():
    with pytest.raises(ValueError, match="neither team.id nor enterprise.id"):
        integration.parse_installation({"access_token": "xoxb-secret"})


def test_splits_secret_from_metadata():
    installation = integration.parse_installation(
        {
            "team": {"id": "T123"},
            "access_token": "xoxb-secret",
            "refresh_token": "xoxe-refresh",
            "bot_user_id": "U999",
            "enterprise": {"id": "E456"},
            "authed_user": {"id": "U111"},
        }
    )
    assert installation.token_payload == {
        "access_token": "xoxb-secret",
        "refresh_token": "xoxe-refresh",
    }
    assert installation.provider_metadata == {
        "bot_user_id": "U999",
        "enterprise_id": "E456",
        "installed_by_slack_user_id": "U111",
        "selected_channels": [],
        # A (re)connect cancels a pending uninstall deletion.
        "uninstalled_at": None,
        "uninstall_data_deleted_at": None,
    }
    # Secret material never ends up in the clear-text metadata.
    assert "xoxb-secret" not in installation.provider_metadata.values()


def test_missing_authed_user_leaves_installed_by_none():
    installation = integration.parse_installation(
        {"team": {"id": "T123"}, "access_token": "xoxb-secret"}
    )
    assert installation.provider_metadata["installed_by_slack_user_id"] is None


def test_expires_in_becomes_token_expires_at():
    before = datetime.now(timezone.utc)
    installation = integration.parse_installation(
        {"team": {"id": "T123"}, "access_token": "x", "expires_in": 3600}
    )
    assert installation.token_expires_at is not None
    assert installation.token_expires_at - before >= timedelta(seconds=3599)


def test_no_expires_in_leaves_token_expires_at_none():
    installation = integration.parse_installation({"team": {"id": "T123"}, "access_token": "x"})
    assert installation.token_expires_at is None


def test_scope_is_passed_through_as_scopes():
    installation = integration.parse_installation(
        {"team": {"id": "T123"}, "access_token": "x", "scope": "commands,chat:write"}
    )
    assert installation.scopes == "commands,chat:write"


def test_the_only_write_scope_is_the_one_answers_need():
    """SDK-832: no write scope beyond what a call needs. ``@cognee`` answers with
    ``chat.postMessage`` and replies with ``chat.postEphemeral`` (both ``chat:write``);
    everything else posts through ``response_url`` or ``views.*``, which need none."""
    from cognee.modules.integrations.slack import oauth

    scopes = oauth._BOT_SCOPES.split(",")
    assert [scope for scope in scopes if scope.endswith(":write")] == ["chat:write"]
    assert "commands" in scopes and "channels:read" in scopes


@pytest.mark.asyncio
async def test_forget_related_data_forgets_what_the_user_remembered_from_this_workspace():
    """And the answers recorded for rating under that user."""
    user = SimpleNamespace(id="user-1")
    credential = SimpleNamespace(provider_account_id="T123")
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.forget_remembered", new=AsyncMock()
        ) as forget_remembered,
        patch(
            "cognee.modules.integrations.slack.handle_cognee_ask.forget_answer_records",
            new=AsyncMock(),
        ) as forget_answer_records,
    ):
        await integration.forget_related_data(credential, user)

    forget_remembered.assert_awaited_once_with(user, team_id="T123")
    forget_answer_records.assert_awaited_once_with("T123", "user-1")
