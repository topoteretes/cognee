"""serve() must fail loudly at connect time instead of lying.

Covers the two failure modes: nothing to connect with (no url, no env
config, no usable saved connection), and a reachable instance that rejects
the API key — which previously printed "Connected" and failed on the first
operation."""

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
    for var in ("COGNEE_SERVICE_URL", "COGNEE_API_KEY"):
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
        json.dumps({"service_url": "http://stale.invalid", "api_key": "dead-key"})
    )
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)

    async def failing_health_check(self):
        return False

    monkeypatch.setattr(CloudClient, "_health_check", failing_health_check)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    message = exc_info.value.message
    assert str(creds_file) in message
    assert "http://stale.invalid" in message
    assert "COGNEE_SERVICE_URL" in message


def _patch_probes(monkeypatch, health: bool, auth_status):
    async def health_check(self):
        return health

    async def auth_check(self):
        return auth_status

    async def get_tenant_id(self):
        return ""

    monkeypatch.setattr(CloudClient, "_health_check", health_check)
    monkeypatch.setattr(CloudClient, "_auth_check", auth_check)
    monkeypatch.setattr(CloudClient, "_get_tenant_id", get_tenant_id)


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
        assert json.loads(creds_file.read_text()) == {
            "service_url": "http://some-instance:8000",
            "api_key": "good-key",
            "tenant_id": "",
        }
    finally:
        await cognee.disconnect()


@pytest.mark.asyncio
async def test_serve_reconnects_to_saved_instance(monkeypatch, tmp_path):
    """A bare serve() after a successful connection reuses the saved instance."""
    import cognee

    creds_file = tmp_path / "cloud_credentials.json"
    _write_saved_creds(creds_file)
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    _patch_probes(monkeypatch, health=True, auth_status=200)

    client = await serve()
    try:
        assert client.service_url == "http://saved-instance.invalid"
    finally:
        await cognee.disconnect()


@pytest.mark.asyncio
async def test_serve_does_not_reuse_a_saved_keyless_connection(monkeypatch, tmp_path):
    """Without a key nothing proves the saved URL still serves the same
    instance, so a bare serve() asks for the URL instead of reconnecting."""
    creds_file = tmp_path / "cloud_credentials.json"
    creds_file.write_text(json.dumps({"service_url": "http://localhost:8000", "api_key": ""}))
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    _patch_probes(monkeypatch, health=True, auth_status=200)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    assert "cognee.serve(url=" in exc_info.value.message


@pytest.mark.asyncio
async def test_serve_saved_creds_with_rejected_key_fail_with_guidance(monkeypatch, tmp_path):
    """A reachable instance whose saved key no longer works must not 'connect'."""
    creds_file = tmp_path / "cloud_credentials.json"
    creds_file.write_text(
        json.dumps({"service_url": "http://alive-but-rejects.invalid", "api_key": "revoked"})
    )
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)
    _patch_probes(monkeypatch, health=True, auth_status=401)

    with pytest.raises(CogneeConfigurationError) as exc_info:
        await serve()

    assert "http://alive-but-rejects.invalid" in exc_info.value.message


def test_saved_creds_from_an_older_release_still_load(monkeypatch, tmp_path):
    """Files written by the removed Auth0 login carry extra fields; they must
    load as the connection they also hold, not be rejected."""
    creds_file = tmp_path / "cloud_credentials.json"
    creds_file.write_text(
        json.dumps(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_at": 1.0,
                "service_url": "http://saved-instance.invalid",
                "api_key": "saved-key",
                "management_url": "https://api.example",
                "tenant_id": "t1",
                "tenant_name": "tenant",
                "email": "user@example.com",
            }
        )
    )
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", creds_file)

    creds = creds_mod.load_credentials()

    assert creds == creds_mod.CloudCredentials(
        service_url="http://saved-instance.invalid", api_key="saved-key", tenant_id="t1"
    )


def _write_saved_creds(path):
    path.write_text(
        json.dumps({"service_url": "http://saved-instance.invalid", "api_key": "saved-key"})
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
