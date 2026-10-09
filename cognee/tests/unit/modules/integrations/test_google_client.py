"""Shared OAuth response parsing keeps errors useful without exposing secrets."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from cognee.modules.integrations.google import client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret response body"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "secret response body"),
        aiohttp.ContentTypeError(
            SimpleNamespace(real_url="https://example.com/secret"),
            (),
            message="secret response body",
        ),
        aiohttp.ClientPayloadError("secret response body"),
        asyncio.TimeoutError("secret response body"),
    ],
)
async def test_invalid_token_response_raises_a_sanitized_error(error, caplog):
    response = AsyncMock()
    response.status = 502
    response.json.side_effect = error
    response.__aenter__.return_value = response
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.post = Mock(return_value=response)

    with (
        patch.object(client.aiohttp, "ClientSession", return_value=session),
        pytest.raises(client.GoogleAuthError) as caught,
    ):
        await client.exchange_code(
            "secret-code", client_id="client", client_secret="secret", redirect_uri="callback"
        )

    assert caught.value.code == "http_502"
    assert str(caught.value) == "Google code exchange failed: http_502"
    assert caught.value.__suppress_context__ is True
    assert "secret" not in str(caught.value)
    assert "secret" not in caplog.text
