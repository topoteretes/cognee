"""Unit tests for cognee.modules.integrations.github.adapter.

Network calls (code exchange, installation lookup) are mocked at the
adapter/app_auth seam — what's under test is the install-flow logic: the
installation_id must be verified against the completing user before it is
trusted, and the stored credential must carry no token material.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.github import app_auth
from cognee.modules.integrations.github.adapter import GithubIntegration
from cognee.modules.integrations.github.verify_github_signature import GithubWebhookVerifier

_INSTALLATION = {
    "id": 12345,
    "app_id": 7,
    "repository_selection": "selected",
    "account": {"login": "acme-org", "id": 99, "type": "Organization"},
}


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    settings = "cognee.modules.integrations.github.github_settings.github_settings"
    monkeypatch.setattr(f"{settings}.app_slug", "cognee-app")
    monkeypatch.setattr(f"{settings}.client_id", "Iv1.abc")
    monkeypatch.setattr(f"{settings}.client_secret", "shhh")
    monkeypatch.setattr(f"{settings}.webhook_secret", "hook-secret")
    monkeypatch.setattr(f"{settings}.frontend_base_url", "https://app.example.com")


def test_authorize_url_is_the_app_installation_flow():
    url = GithubIntegration().authorize_url("the-state")
    assert url == "https://github.com/apps/cognee-app/installations/new?state=the-state"


def test_state_signing_secret_is_the_webhook_secret():
    assert GithubIntegration().state_signing_secret() == "hook-secret"


def test_webhook_verifier_is_registered():
    assert isinstance(GithubIntegration().webhook_verifier(), GithubWebhookVerifier)


@pytest.mark.asyncio
async def test_exchange_callback_rejects_a_missing_installation_id():
    with pytest.raises(ValueError, match="installation_id"):
        await GithubIntegration().exchange_callback("code", {})
    with pytest.raises(ValueError, match="installation_id"):
        await GithubIntegration().exchange_callback("code", {"installation_id": "not-a-number"})


@pytest.mark.asyncio
async def test_exchange_callback_requires_the_user_to_administer_the_installation():
    integration = GithubIntegration()
    with (
        patch.object(
            integration, "exchange_code", new=AsyncMock(return_value={"access_token": "user-tok"})
        ),
        patch.object(
            app_auth, "get_installation", new=AsyncMock(return_value=_INSTALLATION)
        ) as lookup,
        patch.object(
            app_auth, "user_administers_installation", new=AsyncMock(return_value=True)
        ) as admin_check,
    ):
        result = await integration.exchange_callback("code", {"installation_id": "12345"})

    assert result == _INSTALLATION
    # The credential comes from the app-authenticated lookup, never from the
    # unauthenticated callback query string, and the admin check runs
    # against that record's account.
    lookup.assert_awaited_once_with(12345)
    admin_check.assert_awaited_once_with("user-tok", _INSTALLATION)


@pytest.mark.asyncio
async def test_exchange_callback_refuses_a_member_who_does_not_administer_the_installation():
    """SDK-832: an org member who can read one repository must not bind the org.

    Once bound, every sync indexes every repository the installation covers
    into the binder's dataset, so "can see the installation" (GitHub's
    user-installations listing, true for any member with one readable repo)
    is not the bar; administering the account is.
    """
    integration = GithubIntegration()
    with (
        patch.object(
            integration, "exchange_code", new=AsyncMock(return_value={"access_token": "user-tok"})
        ),
        patch.object(app_auth, "get_installation", new=AsyncMock(return_value=_INSTALLATION)),
        patch.object(app_auth, "user_administers_installation", new=AsyncMock(return_value=False)),
        pytest.raises(PermissionError, match="administer"),
    ):
        await integration.exchange_callback("code", {"installation_id": "12345"})


def test_parse_installation_stores_no_token_material():
    installation = GithubIntegration().parse_installation(_INSTALLATION)

    assert installation.provider_account_id == "12345"
    # No durable per-installation secret exists — tokens are minted on
    # demand from the app private key, so the encrypted payload stays empty.
    assert installation.token_payload == {}
    assert installation.auth_type == "github_app"
    assert installation.account_label == "acme-org"
    assert installation.provider_metadata == {
        "account_login": "acme-org",
        "account_id": 99,
        "account_type": "Organization",
        "repository_selection": "selected",
        "app_id": 7,
    }


def test_parse_installation_without_id_raises():
    with pytest.raises(ValueError, match="no id"):
        GithubIntegration().parse_installation({"account": {"login": "acme-org"}})


@pytest.mark.asyncio
async def test_a_full_sync_while_one_runs_is_dropped_and_marked_running(monkeypatch):
    """A schedule shorter than a sync would otherwise stack one more clone of every
    repository per slot; the manual route reads the same mark to say it did not start."""
    import asyncio
    from types import SimpleNamespace

    from cognee.modules.integrations import ingestion
    from cognee.modules.integrations.github import adapter as adapter_module

    entered, release, calls = asyncio.Event(), asyncio.Event(), []

    async def sync_repositories(credential):
        calls.append(credential)
        entered.set()
        await release.wait()

    monkeypatch.setattr(adapter_module, "sync_repositories", sync_repositories)
    monkeypatch.setattr(adapter_module, "sync_is_paused", AsyncMock(return_value=False))
    integration, credential = GithubIntegration(), SimpleNamespace(provider_account_id="42")

    first = asyncio.create_task(integration.sync_now(credential))
    await entered.wait()
    assert ingestion.sync_is_running("github", "42")
    # Dropped calls return at once; one that slipped past the guard would wait on `release`.
    await asyncio.wait_for(integration.sync_now(credential), timeout=1)
    await asyncio.wait_for(integration.on_installed(credential), timeout=1)
    release.set()
    await first

    assert calls == [credential]
    assert not ingestion.sync_is_running("github", "42")


@pytest.mark.asyncio
async def test_the_install_sync_of_a_paused_connection_does_not_start(monkeypatch):
    """An owner's paused default holds a connection from its first sync (SDK-1434)."""
    from types import SimpleNamespace

    from cognee.modules.integrations.github import adapter as adapter_module

    sync = AsyncMock()
    monkeypatch.setattr(adapter_module, "sync_repositories", sync)
    monkeypatch.setattr(adapter_module, "sync_is_paused", AsyncMock(return_value=True))

    await GithubIntegration().on_installed(SimpleNamespace(provider_account_id="42"))

    sync.assert_not_awaited()
