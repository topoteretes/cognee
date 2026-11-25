"""Unit tests for cognee_connectors_core.registry.

The registration checks are ported from the cloud registry (cognee-saas-backend,
2026-10-26); the catalog shape and the entry-point loaders are this package's.
Entry points are faked with real ``importlib.metadata.EntryPoint`` objects
pointing at modules planted in ``sys.modules``, so ``load()`` runs the real
import path without a package being installed.
"""

import importlib.metadata
import sys
import types
from collections.abc import Mapping
from typing import Any

import pytest

from cognee_connectors_core import registry
from cognee_connectors_core.base import Connector, ConnectorInstallation, SyncScope
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.signatures import HmacVerifier, SignatureScheme
from cognee_connectors_core.testing import FakeSettings


def _descriptor(**overrides) -> ConnectorDescriptor:
    base = {
        "provider": "testprov",
        "display_name": "Test Provider",
        "authorize_url": "https://testprov.example/oauth/authorize",
        "token_url": "https://testprov.example/oauth/token",
        "account_id_path": "account.id",
        "signature_scheme": SignatureScheme.SHA256_PREFIX,
        "signature_header": "x-signature",
        "webhook_account_id_path": "account.id",
    }
    return ConnectorDescriptor(**{**base, **overrides})


def _connector(**overrides) -> DescriptorConnector:
    return DescriptorConnector(_descriptor(**overrides), FakeSettings(webhook_secret="s"))


class _Tier3(Connector):
    provider = "custom"
    display_name = "Custom"

    def authorize_url(self, state: str) -> str:
        return "https://custom.example/authorize"

    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        return {}

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        return ConnectorInstallation(provider_account_id="a")

    async def sync_scope(self, credential, tokens, *, resources=None) -> SyncScope:
        return SyncScope(dataset_name="custom_a")


# ---- use_connector / lookups --------------------------------------------


def test_use_connector_registers_and_returns_the_connector():
    connector = _connector()

    assert registry.use_connector(connector) is connector
    assert registry.get_connector("testprov") is connector
    assert registry.find_connector("testprov") is connector
    assert registry.registered_providers() == ["testprov"]


def test_a_bare_descriptor_is_wrapped():
    """An API-key or local provider's CONNECTOR may be the descriptor itself."""
    registered = registry.use_connector(
        ConnectorDescriptor(provider="apollo", display_name="Apollo", auth="api_key")
    )

    assert isinstance(registered, DescriptorConnector)
    assert registered.auth == "api_key"
    assert registry.get_connector("apollo") is registered


def test_something_else_is_refused():
    with pytest.raises(TypeError, match="Connector or ConnectorDescriptor"):
        registry.use_connector(object())


def test_get_connector_raises_key_error_for_an_unknown_provider():
    with pytest.raises(KeyError):
        registry.get_connector("nope")


def test_find_connector_returns_none_for_an_unknown_provider():
    assert registry.find_connector("nope") is None


def test_registered_providers_are_sorted():
    registry.use_connector(_connector(provider="zeta"))
    registry.use_connector(_connector(provider="alpha"))

    assert registry.registered_providers() == ["alpha", "zeta"]


def test_an_empty_provider_is_refused():
    class _Nameless(_Tier3):
        provider = ""

    with pytest.raises(ValueError, match="no provider set"):
        registry.use_connector(_Nameless())


def test_an_overlong_provider_is_refused():
    with pytest.raises(ValueError, match="over the 100-char limit"):
        registry.use_connector(_connector(provider="p" * 101))


def test_a_colon_in_the_provider_is_refused():
    """The signed OAuth state is colon-joined; the name would change its field count."""
    with pytest.raises(ValueError, match="cannot contain ':'"):
        registry.use_connector(_connector(provider="my:prov"))


@pytest.mark.parametrize("provider", ["my prov", "my.prov", "my/prov", "café", "ｇithub"])
def test_a_provider_that_cannot_become_a_settings_prefix_is_refused(provider):
    with pytest.raises(ValueError, match="ASCII alphanumeric"):
        registry.use_connector(_connector(provider=provider))


@pytest.mark.parametrize("provider", ["ms-teams", "ms_teams", "GitHub", "github2"])
def test_a_plain_provider_name_is_accepted(provider):
    assert registry.use_connector(_connector(provider=provider)).provider == provider


def test_two_providers_normalizing_to_the_same_prefix_are_refused():
    """``ms-teams`` and ``ms_teams`` would both read MS_TEAMS_* settings."""
    registry.use_connector(_connector(provider="ms-teams"))

    with pytest.raises(ValueError, match="collides with already-registered 'ms-teams'"):
        registry.use_connector(_connector(provider="ms_teams"))


def test_replacing_a_connector_is_allowed_and_logged(caplog):
    """A Tier 3 implementation may supersede a descriptor for the same provider."""
    first = _connector()
    second = _connector()
    registry.use_connector(first)

    with caplog.at_level("INFO"):
        registry.use_connector(second)

    assert registry.get_connector("testprov") is second
    assert "Replacing connector for testprov" in caplog.text


def test_a_verifier_without_a_handler_is_warned_about(caplog):
    class _HalfWebhook(_Tier3):
        def webhook_verifier(self):
            return HmacVerifier(
                provider="custom",
                scheme=SignatureScheme.BARE_HEX,
                header="x-sig",
                secret_provider=lambda: "s",
            )

    with caplog.at_level("WARNING"):
        registry.use_connector(_HalfWebhook())

    assert "has not overridden handle_webhook()" in caplog.text


def test_a_connector_whose_verifier_raises_still_registers(caplog):
    class _Broken(_Tier3):
        def webhook_verifier(self):
            raise RuntimeError("misconfigured")

    with caplog.at_level("WARNING"):
        registered = registry.use_connector(_Broken())

    assert registry.get_connector("custom") is registered
    assert "failed to build its webhook verifier" in caplog.text


def test_reset_registry_empties_everything():
    registry.use_connector(_connector(provider="ms-teams"))

    registry.reset_registry()

    assert registry.registered_providers() == []
    # The normalized-name history is cleared too, or the next registration
    # of the same name would be reported as a collision with itself.
    registry.use_connector(_connector(provider="ms_teams"))


# ---- catalog -----------------------------------------------------------


def test_catalog_describes_a_descriptor_connector():
    registry.use_connector(_connector(content_type="code"))

    assert registry.catalog() == [
        {
            "provider": "testprov",
            "display_name": "Test Provider",
            "auth": "oauth2",
            "tier": 1,
            "has_webhooks": True,
            "supports_resource_selection": False,
            "poll_interval_seconds": None,
            "content_type": "code",
            "degraded": False,
        }
    ]


def test_catalog_guesses_the_tier_from_the_class():
    class _Tier2(DescriptorConnector):
        async def resolve_account(self, token_response):
            return None

    registry.use_connector(_Tier2(_descriptor(provider="two"), FakeSettings()))
    registry.use_connector(_Tier3())

    tiers = {entry["provider"]: entry["tier"] for entry in registry.catalog()}
    assert tiers == {"two": 2, "custom": 3}


def test_catalog_shows_an_api_key_provider():
    registry.use_connector(
        ConnectorDescriptor(
            provider="apollo", display_name="Apollo", auth="api_key", poll_interval_seconds=3600
        )
    )

    [entry] = registry.catalog()
    assert entry["auth"] == "api_key"
    assert entry["has_webhooks"] is False
    assert entry["poll_interval_seconds"] == 3600


def test_catalog_reports_a_connector_that_cannot_describe_itself_as_degraded(caplog):
    """One broken connector must not blank the status page for the others."""

    class _Broken(_Tier3):
        provider = "broken"

        @property
        def poll_interval_seconds(self):
            raise RuntimeError("boom")

    registry.use_connector(_connector())
    registry.use_connector(_Broken())

    with caplog.at_level("ERROR"):
        entries = {entry["provider"]: entry for entry in registry.catalog()}

    assert entries["testprov"]["degraded"] is False
    assert entries["broken"]["degraded"] is True
    assert entries["broken"]["display_name"] == "broken"
    assert entries["broken"]["tier"] == 3
    assert "Connector broken failed to describe itself" in caplog.text


# ---- entry points ------------------------------------------------------


def _plant_module(monkeypatch, name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _entry_point(name: str, value: str, group: str) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(name=name, value=value, group=group)


def _fake_entry_points(monkeypatch, entry_points):
    """Make the registry see exactly ``entry_points`` for their group."""
    by_group: dict[str, list] = {}
    for entry_point in entry_points:
        by_group.setdefault(entry_point.group, []).append(entry_point)
    monkeypatch.setattr(registry, "_entry_points", lambda group: list(by_group.get(group, [])))


def test_load_connectors_registers_every_advertised_connector(monkeypatch):
    _plant_module(monkeypatch, "fake_connector_one", CONNECTOR=_connector(provider="one"))
    _plant_module(
        monkeypatch,
        "fake_connector_two",
        CONNECTOR=ConnectorDescriptor(provider="two", display_name="Two", auth="local"),
    )
    _fake_entry_points(
        monkeypatch,
        [
            _entry_point("one", "fake_connector_one:CONNECTOR", registry.CONNECTORS_GROUP),
            _entry_point("two", "fake_connector_two:CONNECTOR", registry.CONNECTORS_GROUP),
        ],
    )

    loaded = registry.load_connectors()

    assert set(loaded) == {"one", "two"}
    assert registry.registered_providers() == ["one", "two"]
    assert loaded["one"] is registry.get_connector("one")
    assert isinstance(loaded["two"], DescriptorConnector)


def test_load_connectors_skips_a_package_that_fails_to_import(monkeypatch, caplog):
    _plant_module(monkeypatch, "fake_connector_ok", CONNECTOR=_connector(provider="ok"))
    _fake_entry_points(
        monkeypatch,
        [
            _entry_point("broken", "no_such_package_xyz:CONNECTOR", registry.CONNECTORS_GROUP),
            _entry_point("ok", "fake_connector_ok:CONNECTOR", registry.CONNECTORS_GROUP),
        ],
    )

    with caplog.at_level("ERROR"):
        loaded = registry.load_connectors()

    assert set(loaded) == {"ok"}
    assert "Skipping cognee.connectors entry point 'broken'" in caplog.text


def test_load_connectors_skips_an_object_that_is_not_a_connector(monkeypatch, caplog):
    _plant_module(monkeypatch, "fake_connector_bad", CONNECTOR=object())
    _fake_entry_points(
        monkeypatch,
        [_entry_point("bad", "fake_connector_bad:CONNECTOR", registry.CONNECTORS_GROUP)],
    )

    with caplog.at_level("ERROR"):
        assert registry.load_connectors() == {}

    assert "failed to load or register" in caplog.text


def test_load_connectors_warns_when_the_entry_point_name_differs(monkeypatch, caplog):
    _plant_module(monkeypatch, "fake_connector_named", CONNECTOR=_connector(provider="real"))
    _fake_entry_points(
        monkeypatch,
        [_entry_point("alias", "fake_connector_named:CONNECTOR", registry.CONNECTORS_GROUP)],
    )

    with caplog.at_level("WARNING"):
        loaded = registry.load_connectors()

    assert set(loaded) == {"real"}
    assert "entry point 'alias' registers provider 'real'" in caplog.text


def test_load_sources_returns_the_factories_without_calling_them(monkeypatch):
    calls = []

    def notion_source(**kwargs):
        calls.append(kwargs)
        return "source"

    _plant_module(monkeypatch, "fake_source_pkg", notion_source=notion_source, NOT_CALLABLE=3)
    _fake_entry_points(
        monkeypatch,
        [
            _entry_point("notion", "fake_source_pkg:notion_source", registry.SOURCES_GROUP),
            _entry_point("odd", "fake_source_pkg:NOT_CALLABLE", registry.SOURCES_GROUP),
            _entry_point("gone", "no_such_package_xyz:source", registry.SOURCES_GROUP),
        ],
    )

    loaded = registry.load_sources()

    assert loaded == {"notion": notion_source}
    assert calls == []


def test_load_host_registrations_calls_register_and_reports_who_ran(monkeypatch, caplog):
    ran = []

    def register_one():
        ran.append("one")

    def register_failing():
        raise ImportError("No module named 'cognee'")

    _plant_module(monkeypatch, "fake_host_one", register=register_one)
    _plant_module(monkeypatch, "fake_host_two", register=register_failing)
    _fake_entry_points(
        monkeypatch,
        [
            _entry_point("one", "fake_host_one:register", registry.INTEGRATIONS_GROUP),
            _entry_point("two", "fake_host_two:register", registry.INTEGRATIONS_GROUP),
            _entry_point("three", "no_such_package_xyz:register", registry.INTEGRATIONS_GROUP),
        ],
    )

    with caplog.at_level("WARNING"):
        registered = registry.load_host_registrations()

    assert registered == ["one"]
    assert ran == ["one"]
    assert "Skipping cognee.integrations registration 'two'" in caplog.text
    assert "No module named 'cognee'" in caplog.text


def test_loaders_accept_a_custom_group(monkeypatch):
    _plant_module(monkeypatch, "fake_connector_grp", CONNECTOR=_connector(provider="grp"))
    _fake_entry_points(
        monkeypatch, [_entry_point("grp", "fake_connector_grp:CONNECTOR", "my.group")]
    )

    assert set(registry.load_connectors("my.group")) == {"grp"}
    assert registry.load_connectors() == {}


def test_entry_points_use_the_selectable_api(monkeypatch):
    seen = []
    entry_point = _entry_point("x", "mod:attr", "cognee.connectors")

    def fake_entry_points(**params):
        seen.append(params)
        return [entry_point]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert registry._entry_points("cognee.connectors") == [entry_point]
    assert seen == [{"group": "cognee.connectors"}]


def test_entry_points_fall_back_to_the_mapping_shape(monkeypatch):
    """An ``entry_points()`` that takes no selection parameters returns a
    mapping of group name to entry points; both shapes must work."""
    entry_point = _entry_point("x", "mod:attr", "cognee.connectors")

    def fake_entry_points(**params):
        if params:
            raise TypeError("entry_points() takes no keyword arguments")
        return {"cognee.connectors": [entry_point]}

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert registry._entry_points("cognee.connectors") == [entry_point]
    assert registry._entry_points("cognee.sources") == []


def test_the_real_entry_point_lookup_works():
    """Whatever is installed, the lookup itself must not raise."""
    assert isinstance(registry._entry_points(registry.CONNECTORS_GROUP), list)
