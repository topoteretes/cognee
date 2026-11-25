"""``dlt_utils.bearer_token`` and ``BearerAuth``, shared by the Bearer-token connectors."""

import pytest
import requests

from cognee.tasks.ingestion import dlt_utils


def test_a_padded_token_is_stripped_before_it_is_sent():
    token = dlt_utils.bearer_token("  tok_123\n", "Example token")
    request = requests.Request("GET", "https://example.invalid/", auth=dlt_utils.BearerAuth(token))
    assert token == "tok_123"
    assert request.prepare().headers["Authorization"] == "Bearer tok_123"


@pytest.mark.parametrize("value", ["", "   ", "tok 123", "toké", "tok\x00", None, 5])
def test_a_token_requests_would_reject_is_refused_without_being_echoed(value):
    with pytest.raises(ValueError, match="^The Example token is empty or has an invalid format$"):
        dlt_utils.bearer_token(value, "Example token")


def test_the_repr_hides_the_token():
    assert "tok_123" not in repr(dlt_utils.BearerAuth("tok_123"))
