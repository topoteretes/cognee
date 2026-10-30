"""Tests for the cognee host binding (``cognee_connector_github.cognee_host``).

``register()`` replaces the import-time registration cognee's own package had,
and the FastAPI shim adapts a request to the core verifier. Both need cognee.
"""

import hashlib
import hmac

import pytest
from cognee.modules.integrations.registry import supported_integrations
from fastapi import HTTPException
from starlette.requests import Request

from cognee_connector_github.cognee_host import register, routers
from cognee_connector_github.cognee_host.adapter import GithubIntegration
from cognee_connector_github.cognee_host.verify_github_signature import GithubWebhookVerifier
from cognee_connectors_core.signatures import MAX_BODY_BYTES

_SECRET = "It's a Secret to Everybody"


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    monkeypatch.setattr("cognee_connector_github.settings.github_settings.webhook_secret", _SECRET)


def _sign(body: bytes, secret: str = _SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _request(body: bytes, headers: dict[str, str]) -> Request:
    """A real Starlette request carrying ``body`` and ``headers``."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/integrations/github/events",
        "query_string": b"",
        "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def test_register_puts_the_adapter_in_cognees_registry(monkeypatch):
    monkeypatch.delitem(supported_integrations, "github", raising=False)

    register()

    assert isinstance(supported_integrations["github"], GithubIntegration)
    assert supported_integrations["github"].provider == "github"


def test_github_mounts_no_routers():
    assert routers() == []


@pytest.mark.asyncio
async def test_shim_returns_the_verified_raw_body():
    body = b'{"action":"created","installation":{"id":42}}'
    request = _request(body, {"X-Hub-Signature-256": _sign(body)})

    assert await GithubWebhookVerifier().verify(request) == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Hub-Signature-256": "sha256=" + "0" * 64},
        {"X-Hub-Signature-256": _sign(b'{"action":"created"}', "other-secret")},
    ],
)
async def test_shim_maps_a_refused_delivery_to_401(headers):
    request = _request(b'{"action":"created"}', headers)

    with pytest.raises(HTTPException) as info:
        await GithubWebhookVerifier().verify(request)

    assert info.value.status_code == 401
    assert info.value.detail == "Invalid signature"


@pytest.mark.asyncio
async def test_shim_maps_an_oversized_body_to_413():
    body = b"x"
    request = _request(
        body,
        {"Content-Length": str(MAX_BODY_BYTES + 1), "X-Hub-Signature-256": _sign(body)},
    )

    with pytest.raises(HTTPException) as info:
        await GithubWebhookVerifier().verify(request)

    assert info.value.status_code == 413


@pytest.mark.asyncio
async def test_shim_refuses_everything_without_a_secret(monkeypatch):
    monkeypatch.setattr("cognee_connector_github.settings.github_settings.webhook_secret", "")
    body = b"payload"
    request = _request(body, {"X-Hub-Signature-256": _sign(body, "")})

    with pytest.raises(HTTPException) as info:
        await GithubWebhookVerifier().verify(request)

    assert info.value.status_code == 401
