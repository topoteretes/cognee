"""Unit tests for cognee.modules.integrations.registry.

The registry is a plain dict behind two functions — the invariants that
matter: registering makes an integration look-up-able by its own `provider`
name, and an unregistered provider raises rather than silently returning
None (so the router can turn it into a clean 404).
"""

import pytest

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.registry import (
    get_integration,
    supported_integrations,
    use_integration,
)


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def authorize_url(self, state):
        return f"https://fake.example/authorize?state={state}"

    async def exchange_code(self, code):
        return {"code": code}

    def parse_installation(self, token_response):
        return OAuthInstallation(provider_account_id="fake-account", token_payload={})

    def state_signing_secret(self):
        return "fake-secret"

    def frontend_base_url(self):
        return "https://fake.example"


@pytest.fixture(autouse=True)
def _clean_registry():
    # Don't let one test's registration leak into another.
    before = dict(supported_integrations)
    supported_integrations.clear()
    yield
    supported_integrations.clear()
    supported_integrations.update(before)


def test_use_integration_registers_under_its_own_provider_name():
    integration = _FakeIntegration()
    use_integration(integration)
    assert get_integration("fake") is integration


def test_unregistered_provider_raises_key_error():
    with pytest.raises(KeyError):
        get_integration("does-not-exist")


def test_registering_same_provider_twice_replaces_the_first():
    use_integration(_FakeIntegration())
    second = _FakeIntegration()
    use_integration(second)
    assert get_integration("fake") is second
