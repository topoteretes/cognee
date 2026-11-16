"""The dlt resource a host hands to ``remember``: markers, schema and argument checks.

A bad argument fails when ``mixpanel_source()`` is built, not halfway through a run.
"""

import pytest
from fake_mixpanel import FakeMixpanel

from cognee_connector_mixpanel import mixpanel_source
from cognee_connector_mixpanel.source import (
    MixpanelAccessError,
    MixpanelAuthError,
    MixpanelSourceError,
)
from cognee_connectors_core import dlt_conventions
from cognee_connectors_core.dlt_conventions import (
    DOCUMENT_SOURCE_ATTR,
    PIPELINE_SCOPE_ATTR,
)

TABLE = "mixpanel_test"
CREDENTIALS = {"service_account_username": "svc-user", "service_account_secret": "s3cret-s3cret"}

VERSION_GUARD = (
    "Mixpanel sync requires a Cognee build that reads per-row node sets "
    "(DOCUMENT_SYNC_VERSION >= 2). Upgrade Cognee before syncing."
)


def test_the_resource_is_a_merge_resource_tagged_for_document_ingestion():
    resource = mixpanel_source(service=FakeMixpanel(), resource_name=TABLE, **CREDENTIALS)

    assert resource.name == TABLE
    assert resource.write_disposition == "merge"
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "mixpanel"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == TABLE
    hints = resource.compute_table_schema()
    assert hints["columns"]["id"]["primary_key"] is True
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True
    assert resource.cognee_sync_stats == {}


def test_an_old_host_is_refused_before_any_request(monkeypatch):
    monkeypatch.setattr(dlt_conventions, "host_document_sync_version", lambda: 1)

    with pytest.raises(RuntimeError, match="DOCUMENT_SYNC_VERSION"):
        mixpanel_source(service=FakeMixpanel(), resource_name=TABLE, **CREDENTIALS)


@pytest.mark.parametrize("resource_name", ["", "   "])
def test_a_missing_resource_name_fails_at_construction(resource_name):
    with pytest.raises(ValueError, match="resource_name"):
        mixpanel_source(service=FakeMixpanel(), resource_name=resource_name, **CREDENTIALS)


def test_a_credential_is_required():
    with pytest.raises(ValueError, match="credential"):
        mixpanel_source(service=None, resource_name=TABLE)


def test_the_two_credential_shapes_are_exclusive():
    with pytest.raises(ValueError, match="either"):
        mixpanel_source(resource_name=TABLE, api_secret="x", **CREDENTIALS)


def test_a_service_account_needs_both_parts():
    with pytest.raises(ValueError, match="credential"):
        mixpanel_source(resource_name=TABLE, service_account_username="svc-user")


def test_a_partial_service_account_secret_is_rejected():
    with pytest.raises(ValueError, match="credential"):
        mixpanel_source(resource_name=TABLE, service_account_secret="s3cret")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include": ("churn",)},
        {"include": ()},
    ],
)
def test_include_must_name_known_kinds(kwargs):
    with pytest.raises(ValueError, match="include"):
        mixpanel_source(service=FakeMixpanel(), resource_name=TABLE, **CREDENTIALS, **kwargs)


def test_events_from_must_be_an_iso_date():
    with pytest.raises(ValueError, match="events_from"):
        mixpanel_source(
            service=FakeMixpanel(),
            resource_name=TABLE,
            **CREDENTIALS,
            include=("events",),
            events_from="11 November 2026",
        )


def test_today_must_be_a_date_or_a_callable():
    with pytest.raises(ValueError, match="today"):
        mixpanel_source(
            service=FakeMixpanel(), resource_name=TABLE, **CREDENTIALS, today="2026-11-16"
        )


def test_a_malformed_credential_fails_at_construction():
    with pytest.raises(ValueError, match="invalid format"):
        mixpanel_source(
            service=None,
            resource_name=TABLE,
            service_account_username="svc-user",
            service_account_secret="not\nprintable",
        )


def test_the_errors_a_host_catches_are_exported():
    assert issubclass(MixpanelAuthError, MixpanelSourceError)
    assert issubclass(MixpanelAccessError, MixpanelSourceError)
