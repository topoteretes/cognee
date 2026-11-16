"""Contract tests for the Mixpanel connector.

Grow these with the provider's real token-response shape: build a
``DescriptorConnector(DESCRIPTOR, settings=FakeSettings(), http=FakeHttpClient(...))``
and drive ``exchange_callback`` / ``parse_installation`` through it
(docs/PORTING.md, "The contract object").
"""

import pytest

from cognee_connector_mixpanel import CONNECTOR
from cognee_connector_mixpanel.connector import DESCRIPTOR
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeSettings


def test_descriptor_identity():
    assert DESCRIPTOR.provider == "mixpanel"
    assert DESCRIPTOR.display_name == "Mixpanel"
    assert DESCRIPTOR.auth == "api_key"


def test_connector_serves_the_descriptor():
    assert CONNECTOR.provider == DESCRIPTOR.provider
    assert CONNECTOR.display_name == DESCRIPTOR.display_name
    assert CONNECTOR.auth == DESCRIPTOR.auth


@pytest.mark.skipif(DESCRIPTOR.auth != "oauth2", reason="no OAuth flow for this provider")
def test_authorize_url_carries_client_and_state():
    connector = DescriptorConnector(DESCRIPTOR, settings=FakeSettings(client_id="cid"))

    url = connector.authorize_url("state-1")

    assert url.startswith(DESCRIPTOR.authorize_url + "?")
    assert "client_id=cid" in url
    assert "state=state-1" in url
