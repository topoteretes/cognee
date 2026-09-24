"""CloudClient forwards only_context and session_id to /api/v1/search and sends no
``context_format``; the remote's string result comes back untouched."""

import asyncio
from unittest.mock import patch

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.modules.search.types import SearchType

PROMPT = "=== SYSTEM PROMPT ===\nAnswer briefly.\n\n=== USER PROMPT ===\nThe question is: `why?`"


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return [{"search_result": PROMPT, "dataset_id": None, "dataset_name": None}]

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self):
        self.last_url = None
        self.last_json = None

    def post(self, url, json=None, **kwargs):
        self.last_url = url
        self.last_json = json
        return _FakeResponse()


def test_search_forwards_only_context_and_session_id_without_context_format():
    client = CloudClient(service_url="https://example.test", api_key="k")
    fake_session = _FakeSession()

    async def run():
        with patch.object(client, "_get_session", return_value=fake_session):
            return await client.search(
                "why?",
                search_type=SearchType.GRAPH_COMPLETION,
                only_context=True,
                session_id="s1",
                # Retired knob: a caller still passing it must not put it on the wire.
                context_format="prompt",
            )

    result = asyncio.run(run())

    assert fake_session.last_url.endswith("/api/v1/search")
    payload = fake_session.last_json
    assert payload["onlyContext"] is True
    assert payload["sessionId"] == "s1"
    assert "contextFormat" not in payload and "context_format" not in payload
    assert result[0]["search_result"] == PROMPT
