"""Unit tests for cognee.modules.integrations.slack.adapter.SlackIntegration.

Covers the field-extraction logic that used to live in persistence.py's
save_installation() — team vs. enterprise (Grid) routing id, secret/metadata
split, and expires_in -> token_expires_at conversion.
"""

from datetime import datetime, timedelta, timezone

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
