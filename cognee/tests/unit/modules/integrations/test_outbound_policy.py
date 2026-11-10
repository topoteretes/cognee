"""Connectors follow a set ALLOW_HTTP_REQUESTS only; the URL-ingestion API default does not apply.

URL ingestion refuses API requests while the variable is unset, because there the
caller chooses the destination. A connector only reaches its provider's fixed
hosts, so it works unset or blank on every surface and stops only when a set value
disables outbound HTTP.
"""

import pytest

from cognee.modules.integrations.outbound import (
    OutboundHttpDisabledError,
    require_outbound_http,
)
from cognee.modules.operations.origin import operation_origin_scope


@pytest.mark.parametrize("origin", ["api", "sdk", "background"])
@pytest.mark.parametrize("setting", [None, "", "   ", "true", "1", "yes", "on"])
def test_connectors_work_unless_a_set_value_disables_them(monkeypatch, origin, setting):
    if setting is None:
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)

    with operation_origin_scope(origin):
        require_outbound_http("slack")


@pytest.mark.parametrize("origin", ["api", "sdk"])
@pytest.mark.parametrize("setting", ["false", "0", "off", "flase"])
def test_a_set_value_that_is_not_true_stops_connectors(monkeypatch, origin, setting):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)

    with operation_origin_scope(origin), pytest.raises(OutboundHttpDisabledError) as error:
        require_outbound_http("slack")

    assert error.value.status_code == 503
    assert f"ALLOW_HTTP_REQUESTS={setting} disables" in error.value.message
    # The router sends only the message as the 503 detail, so it carries the fix itself.
    assert "Set ALLOW_HTTP_REQUESTS=true in the server's environment" in error.value.message
    assert "(the default)" not in error.value.remediation
