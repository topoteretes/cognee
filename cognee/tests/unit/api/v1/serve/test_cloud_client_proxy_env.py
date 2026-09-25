"""CloudClient must honour HTTP(S)_PROXY / NO_PROXY like cognee's other HTTP clients.

aiohttp ignores the proxy environment variables unless the session is created
with ``trust_env=True``. Without it, a proxied deployment (corporate egress,
Docker Sandboxes' credential proxy) sends ``cognee push`` / ``serve()`` traffic
around the proxy: a proxy-managed API key is never substituted and the server
sees the placeholder, so every request fails with 401.
"""

from unittest.mock import MagicMock, patch

import pytest

# Patch through the module object, not a dotted string: cognee.api.v1 re-exports
# the ``serve`` function, which shadows the ``serve`` package, so Python 3.10's
# mock.patch resolves "cognee.api.v1.serve..." to the function and fails.
from cognee.api.v1.serve import cloud_client as cloud_client_module
from cognee.api.v1.serve.cloud_client import CloudClient


@pytest.mark.asyncio
async def test_session_is_created_with_trust_env():
    client = CloudClient(service_url="https://api.example.test", api_key="k")

    with patch.object(cloud_client_module.aiohttp, "ClientSession") as session_cls:
        session_cls.return_value = MagicMock(closed=False)
        await client._get_session()

    session_cls.assert_called_once()
    assert session_cls.call_args.kwargs["trust_env"] is True
    assert session_cls.call_args.kwargs["headers"] == {"X-Api-Key": "k"}
