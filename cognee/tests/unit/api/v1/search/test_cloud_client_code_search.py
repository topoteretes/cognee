from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO
from cognee.modules.search.types import SearchType


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return [{"found": True}]


class _Session:
    def __init__(self):
        self.payload = None

    def post(self, _url, *, json):
        self.payload = json
        return _Response()


def test_search_payload_accepts_camel_case_code_query():
    payload = SearchPayloadDTO.model_validate(
        {
            "searchType": "CODE",
            "query": "CheckoutService",
            "codeQuery": {"operation": "traverse", "direction": "reverse"},
        }
    )

    assert payload.search_type is SearchType.CODE
    assert payload.code_query == {"operation": "traverse", "direction": "reverse"}


@pytest.mark.asyncio
async def test_cloud_client_forwards_structured_code_query():
    client = CloudClient("https://example.test", "key")
    session = _Session()
    client._get_session = AsyncMock(return_value=session)

    result = await client.search(
        "CheckoutService",
        search_type=SearchType.CODE,
        datasets=["product"],
        code_query={"operation": "find_path", "target": "PaymentStore"},
    )

    assert result == [{"found": True}]
    assert session.payload["searchType"] == "CODE"
    assert session.payload["codeQuery"] == {
        "operation": "find_path",
        "target": "PaymentStore",
    }
