from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.api.v1.serve.serve import _serve_cloud
from cognee.api.v1.serve.credentials import CloudCredentials
from cognee.api.v1.serve.cloud_client import CloudClient

@pytest.mark.asyncio
async def test_serve_cloud_immediate_health_check_success():
    """If saved credentials exist and immediate health check succeeds, reuse the connection
    without requiring is_token_expired to be false and without refreshing.
    """
    creds = CloudCredentials(
        access_token="expired_access_token",
        refresh_token="some_refresh_token",
        expires_at=time.time() - 3600, # expired 1 hour ago
        service_url="https://cloud.cognee.test",
        api_key="test_api_key",
        email="test@cognee.test",
    )

    with patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds), \
         patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_client, \
         patch.object(CloudClient, "_health_check", new_callable=AsyncMock) as mock_health_check, \
         patch("cognee.api.v1.serve.device_auth.refresh_access_token", new_callable=AsyncMock) as mock_refresh:
        
        mock_health_check.return_value = True

        client = await _serve_cloud()

        assert client is not None
        assert client.service_url == "https://cloud.cognee.test"
        assert client.api_key == "test_api_key"
        
        mock_health_check.assert_called()
        mock_set_client.assert_called_once_with(client)
        mock_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_serve_cloud_immediate_health_check_fail_token_valid():
    """If immediate health check fails, but token is not expired, attempt standard health check."""
    creds = CloudCredentials(
        access_token="valid_access_token",
        refresh_token="some_refresh_token",
        expires_at=time.time() + 3600, # expires in 1 hour
        service_url="https://cloud.cognee.test",
        api_key="test_api_key",
        email="test@cognee.test",
    )

    # We want health check to fail on the fast check (3 calls), but succeed on the standard check
    health_check_returns = [False, False, False, True]

    with patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds), \
         patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_client, \
         patch.object(CloudClient, "_health_check", new_callable=AsyncMock) as mock_health_check, \
         patch.object(CloudClient, "close", new_callable=AsyncMock) as mock_close, \
         patch("cognee.api.v1.serve.device_auth.refresh_access_token", new_callable=AsyncMock) as mock_refresh:
        
        mock_health_check.side_effect = health_check_returns

        client = await _serve_cloud()

        assert client is not None
        assert mock_health_check.call_count == 4
        mock_set_client.assert_called_once()
        mock_close.assert_called_once()
        mock_refresh.assert_not_called()


@pytest.mark.asyncio
async def test_serve_cloud_immediate_health_check_fail_token_expired_refresh_success():
    """If immediate health check fails, token is expired, but refresh succeeds."""
    creds = CloudCredentials(
        access_token="expired_access_token",
        refresh_token="some_refresh_token",
        expires_at=time.time() - 3600, # expired
        service_url="https://cloud.cognee.test",
        api_key="test_api_key",
        email="test@cognee.test",
    )

    from cognee.api.v1.serve.device_auth import TokenResponse
    refreshed_token = TokenResponse(
        access_token="new_access_token",
        id_token="new_id_token",
        refresh_token="new_refresh_token",
        expires_in=3600,
        token_type="Bearer",
    )

    health_check_returns = [False, False, False, True]

    with patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds), \
         patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_client, \
         patch("cognee.api.v1.serve.credentials.save_credentials") as mock_save, \
         patch.object(CloudClient, "_health_check", new_callable=AsyncMock) as mock_health_check, \
         patch.object(CloudClient, "close", new_callable=AsyncMock) as mock_close, \
         patch("cognee.api.v1.serve.device_auth.refresh_access_token", new_callable=AsyncMock, return_value=refreshed_token) as mock_refresh:
        
        mock_health_check.side_effect = health_check_returns

        client = await _serve_cloud()

        assert client is not None
        mock_refresh.assert_called_once_with("some_refresh_token", domain=None, client_id=None)
        mock_save.assert_called_once()
        mock_set_client.assert_called_once()
        assert mock_close.call_count == 1


@pytest.mark.asyncio
async def test_serve_cloud_complete_fallback_to_device_flow():
    """If fast check fails, and refresh/token fallback fails, it falls back to full device login flow."""
    creds = CloudCredentials(
        access_token="expired_access_token",
        refresh_token="some_refresh_token",
        expires_at=time.time() - 3600, # expired
        service_url="https://cloud.cognee.test",
        api_key="test_api_key",
        email="test@cognee.test",
    )

    from cognee.api.v1.serve.device_auth import TokenResponse
    device_token = TokenResponse(
        access_token="device_access_token",
        id_token="device_id_token",
        refresh_token="device_refresh_token",
        expires_in=3600,
        token_type="Bearer",
    )

    class MockTenant:
        id = "tenant-123"
        name = "test-tenant"

    with patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds), \
         patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_client, \
         patch("cognee.api.v1.serve.credentials.save_credentials") as mock_save, \
         patch.object(CloudClient, "_health_check", new_callable=AsyncMock) as mock_health_check, \
         patch.object(CloudClient, "close", new_callable=AsyncMock) as mock_close, \
         patch("cognee.api.v1.serve.device_auth.refresh_access_token", side_effect=Exception("refresh error")), \
         patch("cognee.api.v1.serve.device_auth.device_code_login", new_callable=AsyncMock, return_value=device_token) as mock_device_login, \
         patch("cognee.api.v1.serve.management_api.get_current_tenant", new_callable=AsyncMock, return_value=MockTenant()) as mock_tenant, \
         patch("cognee.api.v1.serve.management_api.get_service_url", new_callable=AsyncMock, return_value="https://new.cognee.test") as mock_url, \
         patch("cognee.api.v1.serve.management_api.get_or_create_api_key", new_callable=AsyncMock, return_value="new_api_key") as mock_api_key:
        
        mock_health_check.side_effect = [False, False, False, True]

        client = await _serve_cloud()

        assert client is not None
        assert client.service_url == "https://new.cognee.test"
        assert client.api_key == "new_api_key"
        
        mock_device_login.assert_called_once()
        mock_tenant.assert_called_once()
        mock_url.assert_called_once()
        mock_api_key.assert_called_once()
        mock_save.assert_called_once()
        mock_set_client.assert_called_once_with(client)
        assert mock_close.call_count == 1
