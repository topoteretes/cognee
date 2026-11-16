"""Contract and factory tests for the BambooHR connector.

The contract is an API-key descriptor: no OAuth flow, nothing to register for a
cognee server. The factory is where the explicit field allowlist, the subdomain
and the resource name are validated, before any provider call could be made.
"""

import pytest

from cognee_connector_bamboohr import CONNECTOR, bamboohr_source
from cognee_connector_bamboohr.connector import DESCRIPTOR
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeSettings

FIELDS = ["firstName", "lastName", "jobTitle", "department"]


def test_descriptor_identity():
    assert DESCRIPTOR.provider == "bamboohr"
    assert DESCRIPTOR.display_name == "BambooHR"
    assert DESCRIPTOR.auth == "api_key"


def test_connector_serves_the_descriptor():
    assert CONNECTOR.provider == DESCRIPTOR.provider
    assert CONNECTOR.display_name == DESCRIPTOR.display_name
    assert CONNECTOR.auth == DESCRIPTOR.auth


def test_the_oauth_members_refuse_for_an_api_key_provider():
    with pytest.raises(NotImplementedError):
        CONNECTOR.authorize_url("state-1")
    with pytest.raises(NotImplementedError):
        CONNECTOR.parse_installation({"access_token": "x"})


@pytest.mark.skipif(DESCRIPTOR.auth != "oauth2", reason="no OAuth flow for this provider")
def test_authorize_url_carries_client_and_state():
    connector = DescriptorConnector(DESCRIPTOR, settings=FakeSettings(client_id="cid"))

    url = connector.authorize_url("state-1")

    assert url.startswith(DESCRIPTOR.authorize_url + "?")
    assert "client_id=cid" in url
    assert "state=state-1" in url


# -- the factory --------------------------------------------------------------
def test_a_source_carries_the_document_markers():
    resource = bamboohr_source(
        resource_name="bamboohr_acme",
        subdomain="acme",
        fields=FIELDS,
        service=object(),
    )

    assert resource.name == "bamboohr_acme"
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "bamboohr"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == "bamboohr_acme"
    # The counts fill in as the resource runs; before a run the dict is empty.
    assert resource.cognee_sync_stats == {}


def test_the_allowlist_is_required_and_never_defaulted():
    with pytest.raises(ValueError, match="explicit field allowlist"):
        bamboohr_source(
            resource_name="bamboohr_acme", subdomain="acme", service=object(), fields=None
        )
    with pytest.raises(ValueError, match="explicit field allowlist"):
        bamboohr_source(
            resource_name="bamboohr_acme", subdomain="acme", service=object(), fields=[]
        )


@pytest.mark.parametrize(
    "fields",
    ["firstName", ("firstName", 3), ["ok", "not ok!"], ["ok", "a/b"]],
)
def test_a_malformed_allowlist_is_refused(fields):
    with pytest.raises(ValueError, match="fields"):
        bamboohr_source(
            resource_name="bamboohr_acme", subdomain="acme", service=object(), fields=fields
        )


@pytest.mark.parametrize(
    "subdomain",
    ["", "Acme", "ac me", "acme.example.com", "../etc"],
)
def test_an_invalid_subdomain_is_refused(subdomain):
    with pytest.raises(ValueError, match="subdomain"):
        bamboohr_source(
            resource_name="bamboohr_acme", subdomain=subdomain, service=object(), fields=FIELDS
        )


@pytest.mark.parametrize("resource_name", ["", "   "])
def test_a_blank_resource_name_is_refused(resource_name):
    with pytest.raises(ValueError, match="resource_name"):
        bamboohr_source(
            resource_name=resource_name, subdomain="acme", service=object(), fields=FIELDS
        )


def test_a_key_or_a_service_is_required():
    with pytest.raises(ValueError, match="service or an api_key"):
        bamboohr_source(
            resource_name="bamboohr_acme", subdomain="acme", fields=FIELDS, api_key=None
        )


@pytest.mark.parametrize("max_requests", [0, -1, "10", 1.5])
def test_an_invalid_request_budget_is_refused(max_requests):
    with pytest.raises(ValueError, match="max_requests"):
        bamboohr_source(
            resource_name="bamboohr_acme",
            subdomain="acme",
            service=object(),
            fields=FIELDS,
            max_requests=max_requests,
        )
