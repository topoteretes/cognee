"""serve() must fail loudly at connect time instead of lying.

Covers the two failure modes: nothing to connect with (no url, no env
config, no usable saved credentials, no device client ID), and a reachable
instance that rejects the API key — which previously printed "Connected"
and failed on the first operation."""

import json
from unittest.mock import MagicMock

import pytest

from cognee.api.v1.serve import credentials as creds_mod
from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.api.v1.serve.serve import serve
from cognee.exceptions import CogneeConfigurationError


@pytest.fixture(autouse=True)
def no_serve_env(monkeypatch):
    for var in ("COGNEE_AUTH0_DEVICE_CLIENT_ID", "COGNEE_SERVICE_URL", "COGNEE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.asyncio
async def test_serve_no_config_fails_with_guidance(monkeypatch, tmp_path):
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", tmp_path / "cloud_credentials.json")

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    message = exc_info.value.message
    assert "COGNEE_SERVICE_URL" in message
    assert "cognee.serve(url=" in message


@pytest.mark.asyncio
async def test_serve_stale_credentials_fail_with_guidance(monkeypatch, tmp_path):
    creds_file = tmp_path / "cloud_credentials.json"
    creds_file.write_text(
        json.dumps(
            {
                "access_token": "stale",
                "service_url": "http://stale.invalid",
                "api_key": "dead-key",
                "email": "user@example.com",
                "expires_at": 1.0,
            }
        )
    )
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)

    async def failing_health_check(self):
        return False

    monkeypatch.setattr(CloudClient, "_health_check", failing_health_check)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    message = exc_info.value.message
    assert "user@example.com" in message
    assert "http://stale.invalid" in message
    assert "COGNEE_SERVICE_URL" in message


def _patch_probes(monkeypatch, health: bool, auth_status):
    async def health_check(self):
        return health

    async def auth_check(self):
        return auth_status

    monkeypatch.setattr(CloudClient, "_health_check", health_check)
    monkeypatch.setattr(CloudClient, "_auth_check", auth_check)


@pytest.mark.asyncio
async def test_serve_direct_rejected_key_fails_and_saves_nothing(monkeypatch, tmp_path):
    creds_file = tmp_path / "cloud_credentials.json"
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    _patch_probes(monkeypatch, health=True, auth_status=401)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve(url="http://some-instance:8000", api_key="bad-key")

    assert "API key was rejected" in exc_info.value.message
    # A rejected connect must not poison the credentials cache.
    assert not creds_file.exists()


@pytest.mark.asyncio
async def test_serve_direct_missing_key_explains_how_to_get_one(monkeypatch, tmp_path):
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", tmp_path / "cloud_credentials.json")
    _patch_probes(monkeypatch, health=True, auth_status=401)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve(url="http://some-instance:8000")

    message = exc_info.value.message
    assert "requires authentication" in message
    assert "POST /api/v1/auth/api-keys" in message


@pytest.mark.asyncio
async def test_serve_direct_accepted_key_connects_and_saves(monkeypatch, tmp_path):
    import cognee

    creds_file = tmp_path / "cloud_credentials.json"
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    _patch_probes(monkeypatch, health=True, auth_status=200)

    client = await serve(url="http://some-instance:8000", api_key="good-key")
    try:
        assert client.service_url == "http://some-instance:8000"
        assert creds_file.exists()
    finally:
        await cognee.disconnect()


@pytest.mark.asyncio
async def test_serve_cloud_rejected_provisioned_key_saves_nothing(monkeypatch, tmp_path):
    """The cloud flow must validate the provisioned key before persisting
    credentials — a tenant rejection may not poison the cache (same
    validate-then-save order as the direct flow)."""
    from cognee.api.v1.serve import device_auth, management_api

    creds_file = tmp_path / "cloud_credentials.json"
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    monkeypatch.setenv("COGNEE_AUTH0_DEVICE_CLIENT_ID", "test-client-id")

    async def fake_login(**kwargs):
        return device_auth.TokenResponse(access_token="at", refresh_token="rt")

    async def fake_tenant(mgmt_url, token):
        return management_api.Tenant(id="t1", name="tenant-1")

    async def fake_service_url(mgmt_url, token):
        return "http://tenant-instance.invalid"

    async def fake_api_key(mgmt_url, token):
        return "provisioned-but-rejected"

    monkeypatch.setattr(device_auth, "device_code_login", fake_login)
    monkeypatch.setattr(management_api, "get_current_tenant", fake_tenant)
    monkeypatch.setattr(management_api, "get_service_url", fake_service_url)
    monkeypatch.setattr(management_api, "get_or_create_api_key", fake_api_key)
    _patch_probes(monkeypatch, health=True, auth_status=401)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    assert "rejected the provisioned API key" in exc_info.value.message
    assert not creds_file.exists()


@pytest.mark.asyncio
async def test_serve_cached_cloud_creds_with_rejected_key_fall_through(monkeypatch, tmp_path):
    """A reachable instance whose saved key no longer works must not 'connect';
    without a device client ID it lands in the stale-credentials guidance."""
    creds_file = tmp_path / "cloud_credentials.json"
    creds_file.write_text(
        json.dumps(
            {
                "access_token": "stale",
                "service_url": "http://alive-but-rejects.invalid",
                "api_key": "revoked-key",
                "email": "user@example.com",
                "expires_at": 1.0,
            }
        )
    )
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    _patch_probes(monkeypatch, health=True, auth_status=401)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    assert "user@example.com" in exc_info.value.message


def _write_saved_creds(path):
    path.write_text(
        json.dumps(
            {
                "access_token": "at",
                "service_url": "http://saved-instance.invalid",
                "api_key": "saved-key",
                "email": "user@example.com",
            }
        )
    )


def test_local_execution_with_saved_creds_warns_once(monkeypatch, tmp_path):
    """Saved credentials + no serve() in this process means operations run
    locally — a one-time warning must say so."""
    creds_file = tmp_path / "cloud_credentials.json"
    _write_saved_creds(creds_file)
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    monkeypatch.setattr(state_mod, "_local_execution_noted", False)
    mock_logger = MagicMock()
    monkeypatch.setattr(state_mod, "logger", mock_logger)

    assert state_mod.get_remote_client() is None
    assert state_mod.get_remote_client() is None

    assert mock_logger.warning.call_count == 1
    assert "cognee.serve()" in mock_logger.warning.call_args.args[0]


def test_local_execution_without_saved_creds_is_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", tmp_path / "cloud_credentials.json")
    monkeypatch.setattr(state_mod, "_local_execution_noted", False)
    mock_logger = MagicMock()
    monkeypatch.setattr(state_mod, "logger", mock_logger)

    assert state_mod.get_remote_client() is None

    mock_logger.warning.assert_not_called()


def test_remote_mode_with_saved_creds_is_silent(monkeypatch, tmp_path):
    creds_file = tmp_path / "cloud_credentials.json"
    _write_saved_creds(creds_file)
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    monkeypatch.setattr(state_mod, "_local_execution_noted", False)
    mock_logger = MagicMock()
    monkeypatch.setattr(state_mod, "logger", mock_logger)

    sentinel_client = object()
    monkeypatch.setattr(state_mod, "_remote_client", sentinel_client)
    try:
        assert state_mod.get_remote_client() is sentinel_client
    finally:
        monkeypatch.setattr(state_mod, "_remote_client", None)

    mock_logger.warning.assert_not_called()
