import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from cognee.api.v1.serve.credentials import CloudCredentials
from cognee.api.v1.serve.serve import _serve_cloud


class TokenResponseMock:
    def __init__(
        self,
        access_token="new-access-token",
        refresh_token="new-refresh-token",
        expires_in=3600,
        id_token="id-token",
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_in = expires_in
        self.id_token = id_token


class TenantMock:
    def __init__(self, id="tenant-id", name="tenant-name"):
        self.id = id
        self.name = name


@pytest.mark.asyncio
async def test_serve_cloud_uses_valid_saved_api_key_without_refresh():
    # Saved credentials have expired management token (expires_at is in the past)
    creds = CloudCredentials(
        access_token="old-access-token",
        refresh_token="some-refresh-token",
        expires_at=time.time() - 1000,  # expired
        service_url="https://saved.cognee.ai",
        api_key="valid-cognee-api-key",
        email="test@example.com",
    )

    mock_client = MagicMock()
    mock_client._health_check = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with (
        patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds),
        patch("cognee.api.v1.serve.credentials.save_credentials") as mock_save,
        patch("cognee.api.v1.serve.cloud_client.CloudClient", return_value=mock_client),
        patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_remote,
        patch("cognee.api.v1.serve.device_auth.refresh_access_token", AsyncMock()) as mock_refresh,
    ):
        result = await _serve_cloud()

        assert result == mock_client
        mock_client._health_check.assert_awaited_once()
        mock_set_remote.assert_called_once_with(mock_client)
        # Should not have refreshed the token since API key health check succeeded
        mock_refresh.assert_not_awaited()
        mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_serve_cloud_refreshes_and_fetches_new_keys_when_health_check_fails():
    # Stored credentials fail health check, but we have a refresh token
    creds = CloudCredentials(
        access_token="old-access-token",
        refresh_token="valid-refresh-token",
        expires_at=time.time() - 1000,  # expired
        service_url="https://old.cognee.ai",
        api_key="expired-cognee-api-key",
        email="test@example.com",
    )

    mock_client_old = MagicMock()
    mock_client_old._health_check = AsyncMock(return_value=False)
    mock_client_old.close = AsyncMock()

    mock_client_new = MagicMock()
    mock_client_new._health_check = AsyncMock(return_value=True)
    mock_client_new.close = AsyncMock()

    # We want CloudClient to return mock_client_old on first call, mock_client_new on second
    def cloud_client_side_effect(service_url, api_key):
        if service_url == "https://old.cognee.ai":
            return mock_client_old
        return mock_client_new

    refreshed_token = TokenResponseMock()
    tenant = TenantMock()

    with (
        patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds),
        patch("cognee.api.v1.serve.credentials.save_credentials") as mock_save,
        patch("cognee.api.v1.serve.cloud_client.CloudClient", side_effect=cloud_client_side_effect),
        patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_remote,
        patch(
            "cognee.api.v1.serve.device_auth.refresh_access_token",
            AsyncMock(return_value=refreshed_token),
        ) as mock_refresh,
        patch(
            "cognee.api.v1.serve.management_api.get_current_tenant", AsyncMock(return_value=tenant)
        ),
        patch(
            "cognee.api.v1.serve.management_api.get_service_url",
            AsyncMock(return_value="https://new.cognee.ai"),
        ),
        patch(
            "cognee.api.v1.serve.management_api.get_or_create_api_key",
            AsyncMock(return_value="new-api-key"),
        ),
    ):
        result = await _serve_cloud()

        assert result == mock_client_new
        mock_client_old._health_check.assert_awaited_once()
        mock_client_new._health_check.assert_awaited_once()
        mock_refresh.assert_awaited_once()

        # Verify credentials were updated and saved
        assert creds.access_token == "new-access-token"
        assert creds.service_url == "https://new.cognee.ai"
        assert creds.api_key == "new-api-key"
        mock_save.assert_called_once_with(creds)
        mock_set_remote.assert_called_once_with(mock_client_new)


@pytest.mark.asyncio
async def test_serve_cloud_falls_back_to_device_login_when_refresh_fails():
    creds = CloudCredentials(
        access_token="old-access-token",
        refresh_token="invalid-refresh-token",
        expires_at=time.time() - 1000,  # expired
        service_url="https://old.cognee.ai",
        api_key="expired-cognee-api-key",
        email="test@example.com",
    )

    mock_client_old = MagicMock()
    mock_client_old._health_check = AsyncMock(return_value=False)
    mock_client_old.close = AsyncMock()

    mock_client_final = MagicMock()
    mock_client_final._health_check = AsyncMock(return_value=True)
    mock_client_final.close = AsyncMock()

    refreshed_token = TokenResponseMock()
    tenant = TenantMock()

    with (
        patch("cognee.api.v1.serve.credentials.load_credentials", return_value=creds),
        patch("cognee.api.v1.serve.credentials.save_credentials"),
        patch(
            "cognee.api.v1.serve.cloud_client.CloudClient",
            side_effect=[mock_client_old, mock_client_final],
        ),
        patch("cognee.api.v1.serve.state.set_remote_client") as mock_set_remote,
        patch(
            "cognee.api.v1.serve.device_auth.refresh_access_token",
            AsyncMock(side_effect=Exception("Auth0 error")),
        ),
        patch(
            "cognee.api.v1.serve.device_auth.device_code_login",
            AsyncMock(return_value=refreshed_token),
        ),
        patch(
            "cognee.api.v1.serve.device_auth.extract_email_from_id_token",
            return_value="test@example.com",
        ),
        patch(
            "cognee.api.v1.serve.management_api.get_current_tenant", AsyncMock(return_value=tenant)
        ),
        patch(
            "cognee.api.v1.serve.management_api.get_service_url",
            AsyncMock(return_value="https://final.cognee.ai"),
        ),
        patch(
            "cognee.api.v1.serve.management_api.get_or_create_api_key",
            AsyncMock(return_value="final-api-key"),
        ),
    ):
        result = await _serve_cloud()

        assert result == mock_client_final
        mock_set_remote.assert_called_once_with(mock_client_final)
