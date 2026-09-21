"""Exercise Gmail's provider-specific calls through the OAuth adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.gmail import adapter, client


@pytest.mark.asyncio
async def test_label_picker_uses_gmail_client_with_current_access_token():
    credential = SimpleNamespace(token_expires_at=None)
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "token"}),
        patch.object(
            client,
            "list_labels",
            AsyncMock(
                return_value={
                    "labels": [
                        {"id": "INBOX", "name": "Inbox", "type": "system"},
                        {"id": "Label_1", "name": "Project", "type": "user"},
                        {"name": "missing ID"},
                    ]
                }
            ),
        ) as list_labels,
    ):
        resources = await adapter.GoogleGmailIntegration().list_resources(credential)

    list_labels.assert_awaited_once_with("token")
    assert [(resource["id"], resource["name"]) for resource in resources] == [
        ("INBOX", "Inbox"),
        ("Label_1", "Project"),
    ]
    assert resources[1]["attributes"]["type"] == "user"


@pytest.mark.asyncio
async def test_expired_token_is_refreshed_before_listing_labels():
    from datetime import datetime, timedelta, timezone

    credential = SimpleNamespace(
        token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_account_id="subject",
    )
    refreshed = SimpleNamespace(token_expires_at=None)
    with (
        patch.object(adapter.GoogleGmailIntegration, "refresh", AsyncMock()) as refresh,
        patch.object(adapter, "get_credential_by_account", AsyncMock(return_value=refreshed)),
        patch.object(
            adapter, "decrypt_token_payload", return_value={"access_token": "fresh-token"}
        ) as decrypt,
        patch.object(client, "list_labels", AsyncMock(return_value={"labels": []})) as labels,
    ):
        assert await adapter.GoogleGmailIntegration().list_resources(credential) == []

    refresh.assert_awaited_once_with(credential)
    decrypt.assert_called_once_with(refreshed)
    labels.assert_awaited_once_with("fresh-token")
