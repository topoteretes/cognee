"""Unit tests for cognee_connectors_core.base.

The dataset-naming tests are ported from the slug half of cloud-backend
tests/modules/integrations/test_framework_oauth2.py (cognee-saas-backend,
2026-10-26); the ``Connector`` default-behavior tests are new, since the
contract's optional hooks are what a Tier 3 connector inherits and a host
relies on.
"""

import unicodedata
from collections.abc import Mapping
from typing import Any

import pytest

from cognee_connectors_core import base
from cognee_connectors_core.base import (
    Connector,
    ConnectorInstallation,
    ConnectorResource,
    RefreshedTokens,
    SyncScope,
    WebhookVerifier,
    account_slug,
    dataset_name_for,
)
from cognee_connectors_core.testing import FakeCredential

# ---- A minimal Tier 3 connector -----------------------------------------


class _Tier3(Connector):
    provider = "custom"
    display_name = "Custom"

    def authorize_url(self, state: str) -> str:
        return f"https://custom.example/authorize?state={state}"

    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        return {"code": code}

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        return ConnectorInstallation(provider_account_id="a")

    async def sync_scope(self, credential, tokens, *, resources=None) -> SyncScope:
        return SyncScope(dataset_name="custom_a", items=list(resources or []))


class _WithPicker(_Tier3):
    async def list_resources(self, credential, tokens) -> list[ConnectorResource]:
        return [ConnectorResource(id="r1", name="Repo 1")]


def test_auth_defaults_to_oauth2():
    assert _Tier3.auth == "oauth2"
    assert _Tier3().auth == "oauth2"


def test_optional_hooks_default_to_nothing():
    connector = _Tier3()
    credential = FakeCredential()

    assert connector.webhook_verifier() is None
    assert connector.dataset_name(credential) is None
    assert connector.webhook_delivery_key(b"{}", {}) is None
    assert connector.poll_interval_seconds is None
    assert connector.poll_targets(credential) is None
    assert connector.content_type is None


@pytest.mark.asyncio
async def test_async_hooks_default_to_nothing():
    connector = _Tier3()
    credential = FakeCredential()

    assert await connector.revoke_remote(credential, {}) is None
    assert await connector.refresh(credential, {}) is None
    assert await connector.list_resources(credential, {}) == []


@pytest.mark.asyncio
async def test_handle_webhook_is_not_implemented_by_default():
    with pytest.raises(NotImplementedError, match="custom declares no webhook handling"):
        await _Tier3().handle_webhook(b"{}", {})


def test_supports_resource_selection_is_derived_from_the_override():
    """A flag could disagree with the implementation in both directions."""
    assert _Tier3.supports_resource_selection() is False
    assert _WithPicker.supports_resource_selection() is True
    assert _WithPicker().supports_resource_selection() is True


def test_the_contract_cannot_be_instantiated_half_implemented():
    class _Partial(Connector):
        provider = "partial"
        display_name = "Partial"

    with pytest.raises(TypeError):
        _Partial()


def test_repr_names_the_provider():
    assert repr(_Tier3()) == "<_Tier3 provider='custom'>"


def test_webhook_verifier_is_abstract():
    with pytest.raises(TypeError):
        WebhookVerifier()


def test_refreshed_tokens_defaults():
    refreshed = RefreshedTokens(token_payload={"access_token": "new"})

    assert refreshed.token_expires_at is None
    assert refreshed.scopes is None


def test_installation_defaults():
    installation = ConnectorInstallation(provider_account_id="a")

    assert installation.token_payload == {}
    assert installation.provider_metadata == {}
    assert installation.auth_type == "oauth2"
    assert installation.token_expires_at is None


# ---- account_slug / dataset_name_for ----------------------------------


def test_account_slug_collapses_punctuation_and_case():
    assert account_slug("My-Org") == "my_org"
    assert account_slug("my.org") == "my_org"
    assert account_slug("ACME") == "acme"
    assert account_slug("a--b__c") == "a_b_c"


def test_account_slug_falls_back_for_an_all_punctuation_id():
    assert account_slug("---") == "account"
    assert account_slug("", fallback="nobody") == "nobody"


def test_account_slug_warns_when_two_different_ids_collapse_onto_one_slug(caplog):
    """The event actually worth a WARNING: an observed collision.

    Two differently-punctuated names reaching one slug means their datasets
    are now one dataset, which is exactly what support needs told.
    """
    with caplog.at_level("WARNING"):
        assert dataset_name_for("p", "My-Org") == "p_my_org"
        assert dataset_name_for("p", "my.org") == "p_my_org"

    assert "collapsed two different account ids onto one slug" in caplog.text
    assert "'My-Org'" in caplog.text and "'my.org'" in caplog.text


def test_account_slug_does_not_warn_on_normalization_alone(caplog):
    """The log spam this replaced.

    ``dataset_name_for`` runs once per sync and once per webhook delivery, and a
    Slack team id normalizes on every single call (case fold) while two team
    ids differing only in case do not exist. Warning on "the slug differs
    from the input" therefore emitted an unactionable WARNING per delivery
    forever — the same 100%-noise problem the UUID exemption exists for.
    """
    with caplog.at_level("WARNING"):
        for _ in range(5):
            assert dataset_name_for("p", "T01ABCD") == "p_t01abcd"

    assert caplog.text == ""


def test_account_slug_does_not_warn_for_a_name_it_did_not_change(caplog):
    """Nothing was collapsed, so there is no collision to warn about."""
    with caplog.at_level("WARNING"):
        # Through dataset_name_for, because account_slug itself no longer
        # logs — asserting "no warning" against the pure function would pass
        # whatever the collision logic did.
        assert dataset_name_for("p", "acmecorp") == "p_acmecorp"

    assert caplog.text == ""


def test_account_slug_warns_on_a_case_only_collision(caplog):
    """The collision the original punctuation-only condition missed entirely.

    Slack team ids and GitHub GraphQL node ids are case-sensitive, so
    ``T01ABCD`` and ``t01abcd`` are two different accounts at the provider
    and one dataset here.
    """
    with caplog.at_level("WARNING"):
        assert dataset_name_for("p", "T01ABCD") == "p_t01abcd"
        assert dataset_name_for("p", "t01abcd") == "p_t01abcd"

    assert "collapsed two different account ids onto one slug" in caplog.text


def test_account_slug_does_not_warn_for_a_canonical_uuid(caplog):
    """A UUID's dashes sit at fixed positions and two different UUIDs never
    collapse to the same slug, so there is nothing to warn about — and UUIDs
    are the id shape Linear (and others) actually send, so warning here would
    make this fire on every sync for those providers.
    """
    with caplog.at_level("WARNING"):
        assert dataset_name_for("p", "f47ac10b-58cc-4372-a567-0e02b2c3d479") == (
            "p_f47ac10b_58cc_4372_a567_0e02b2c3d479"
        )

    assert caplog.text == ""


def test_account_slug_normalizes_nfc_and_nfd_forms_to_the_same_slug():
    """``café-org`` typed on one platform (NFC) and another (NFD) are two
    different byte sequences for visually identical text — both alnum
    per-character either way, so without normalizing first they slugify
    differently and split one logical account across two datasets.
    """
    nfc = unicodedata.normalize("NFC", "café-org")
    nfd = unicodedata.normalize("NFD", "café-org")
    assert nfc != nfd  # sanity: these really are different byte sequences

    assert account_slug(nfc) == account_slug(nfd)


def test_account_slug_does_not_treat_a_non_canonical_dashed_string_as_a_uuid(caplog):
    """``UUID(raw)`` strips dashes from anywhere and accepts urn:/braced
    forms, so a string merely shaped like dashes-and-hex-groups (but not at
    canonical UUID positions) would otherwise still parse and suppress the
    collision warning — even though it is exactly the punctuated-string shape
    the warning exists to catch.
    """
    with caplog.at_level("WARNING"):
        assert dataset_name_for("p", "1234-5678-1234-5678-1234-5678-1234-5678") == (
            "p_1234_5678_1234_5678_1234_5678_1234_5678"
        )
        # A real UUID would be exempt from the collision bookkeeping entirely,
        # so this second, colliding spelling could never produce a warning.
        assert dataset_name_for("p", "1234.5678.1234.5678.1234.5678.1234.5678") == (
            "p_1234_5678_1234_5678_1234_5678_1234_5678"
        )

    assert "collapsed two different account ids onto one slug" in caplog.text


def test_two_providers_normalizing_alike_is_not_reported_as_a_collision(caplog):
    """They are different datasets, so it is not a collision."""
    with caplog.at_level("WARNING"):
        assert dataset_name_for("slack", "ACME") == "slack_acme"
        assert dataset_name_for("github", "acme") == "github_acme"

    assert caplog.text == ""


def test_a_recurring_collision_is_warned_about_once(caplog):
    """Not once per delivery.

    ``dataset_name_for`` runs on every sync and every webhook delivery, and an
    earlier version warned and returned without recording the pair, so a
    real collision produced an unactionable WARNING forever.
    """
    with caplog.at_level("WARNING"):
        for _ in range(6):
            dataset_name_for("p", "My-Org")
            dataset_name_for("p", "my.org")

    assert caplog.text.count("collapsed two different account ids") == 1


def test_the_origin_history_only_tracks_ids_that_normalized(monkeypatch):
    """Otherwise ordinary traffic burns the bound and the clear() drops the
    history a colliding pair has to straddle.

    Probed with a tiny limit: ids that slug to themselves can never produce a
    warning, so storing them made detection degrade toward zero exactly as
    the account count grew.
    """
    monkeypatch.setattr(base, "_SLUG_ORIGIN_LIMIT", 3)
    for index in range(20):
        base.dataset_name_for("p", f"plainid{index}")

    assert base._slug_origins == {}


def test_the_origin_history_is_bounded(monkeypatch):
    """The bound itself — deleting the eviction would leave the dict growing
    unbounded on provider-controlled ids.
    """
    monkeypatch.setattr(base, "_SLUG_ORIGIN_LIMIT", 3)
    for index in range(10):
        base.dataset_name_for("p", f"Needs-Slug-{index}")

    assert len(base._slug_origins) <= 3


def test_slug_origins_leak_across_tests_without_the_reset_part_one():
    """Half of the pair that makes ``reset_slug_origins`` in the autouse
    fixture falsifiable: these two claim the same (provider, slug) from
    DIFFERENT raw ids, so whichever runs second warns if the history was not
    cleared.
    """
    assert dataset_name_for("shared", "Leak-Probe") == "shared_leak_probe"
    assert ("shared", "leak_probe") not in base._slug_collisions


def test_slug_origins_leak_across_tests_without_the_reset_part_two():
    """See part one — same slug, deliberately a different raw id."""
    assert dataset_name_for("shared", "leak.probe") == "shared_leak_probe"
    assert ("shared", "leak_probe") not in base._slug_collisions
