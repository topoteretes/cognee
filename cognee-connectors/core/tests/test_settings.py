"""Unit tests for cognee_connectors_core.settings.

``ConnectorSettings`` is the base every provider's settings class subclasses
with its own ``env_prefix``; the environment variable names a provider had in
cognee never change, so the prefix plus the upper-cased field name is the
whole contract.
"""

import pytest
from pydantic_settings import SettingsConfigDict

from cognee_connectors_core.settings import ConnectorSettings, env_var_name, require
from cognee_connectors_core.testing import FakeSettings


class NotionSettings(ConnectorSettings):
    model_config = SettingsConfigDict(env_prefix="NOTION_")

    state_secret: str = ""


def test_fields_default_to_empty_so_an_unconfigured_deployment_still_boots(monkeypatch):
    for name in ("NOTION_CLIENT_ID", "NOTION_CLIENT_SECRET", "NOTION_REDIRECT_URI"):
        monkeypatch.delenv(name, raising=False)

    settings = NotionSettings()

    assert settings.client_id == ""
    assert settings.client_secret == ""
    assert settings.redirect_uri == ""
    assert settings.frontend_base_url == ""


def test_a_subclass_reads_its_own_prefix(monkeypatch):
    monkeypatch.setenv("NOTION_CLIENT_ID", "id-1")
    monkeypatch.setenv("NOTION_CLIENT_SECRET", "secret-1")
    monkeypatch.setenv("NOTION_REDIRECT_URI", "https://host.example/cb")
    monkeypatch.setenv("NOTION_FRONTEND_BASE_URL", "https://app.example")
    monkeypatch.setenv("NOTION_STATE_SECRET", "state-1")

    settings = NotionSettings()

    assert settings.client_id == "id-1"
    assert settings.client_secret == "secret-1"
    assert settings.redirect_uri == "https://host.example/cb"
    assert settings.frontend_base_url == "https://app.example"
    assert settings.state_secret == "state-1"


def test_unrelated_variables_under_the_prefix_are_ignored(monkeypatch):
    """The SDK's own NOTION_API_KEY shares the prefix and is not a setting here.

    pydantic-settings defaults to ``extra="forbid"``, which would refuse to
    load at all; the base sets ``extra="ignore"`` so a subclass inherits it.
    """
    monkeypatch.setenv("NOTION_API_KEY", "sdk-only")

    assert NotionSettings().client_id == ""


def test_require_returns_a_configured_value():
    assert require(FakeSettings(client_id="abc"), "client_id") == "abc"


def test_require_names_the_environment_variable_when_unset(monkeypatch):
    monkeypatch.delenv("NOTION_CLIENT_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="NOTION_CLIENT_SECRET is not configured"):
        require(NotionSettings(), "client_secret")


def test_require_uses_the_stand_in_prefix_from_model_config():
    with pytest.raises(RuntimeError, match="TESTPROV_CLIENT_ID is not configured"):
        require(FakeSettings(client_id=""), "client_id")


def test_env_var_name_without_a_model_config_is_the_bare_field():
    class Bare:
        client_id = ""

    assert env_var_name(Bare(), "client_id") == "CLIENT_ID"


def test_require_raises_for_an_unknown_field():
    with pytest.raises(AttributeError):
        require(FakeSettings(), "no_such_field")
