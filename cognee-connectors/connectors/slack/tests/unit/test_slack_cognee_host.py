"""The cognee-host entry points: ``register()``, ``routers()`` and the FastAPI signature shim."""

import time

import pytest

pytest.importorskip("cognee", reason="the cognee_host adapter needs cognee")

from cognee.modules.integrations import registry as cognee_registry
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from cognee_connector_slack.cognee_host import register, routers
from cognee_connector_slack.cognee_host.adapter import SlackIntegration
from cognee_connector_slack.cognee_host.verify_slack_signature import (
    verify_slack_request,
)
from cognee_connector_slack.settings import SlackSettings
from cognee_connectors_core.signatures import SignatureScheme
from cognee_connectors_core.testing import sign_body

_SECRET = "s3cret"


@pytest.fixture
def cognee_integrations():
    previous = cognee_registry.supported_integrations.get("slack")
    cognee_registry.supported_integrations.pop("slack", None)
    yield cognee_registry.supported_integrations
    if previous is None:
        cognee_registry.supported_integrations.pop("slack", None)
    else:
        cognee_registry.supported_integrations["slack"] = previous


def test_register_puts_the_slack_integration_in_cognees_registry(cognee_integrations):
    register()

    integration = cognee_registry.get_integration("slack")
    assert isinstance(integration, SlackIntegration)
    assert integration.provider == "slack"
    assert integration.settings_cls is SlackSettings


def test_register_twice_replaces_rather_than_duplicates(cognee_integrations):
    register()
    first = cognee_registry.get_integration("slack")
    register()

    assert cognee_registry.get_integration("slack") is not first
    assert [name for name in cognee_integrations if name == "slack"] == ["slack"]


def test_routers_is_the_function_even_after_the_routers_module_was_imported():
    # Importing the submodule rebinds ``cognee_host.routers`` to the module; the
    # package re-exports the function afterwards, so both spellings reach it.
    import importlib

    from cognee_connector_slack import cognee_host

    module = importlib.import_module("cognee_connector_slack.cognee_host.routers")

    assert callable(cognee_host.routers)
    assert cognee_host.routers is module.routers is routers


def _paths(router) -> set[str]:
    """Every route path on ``router``, descending into the routers it included.

    FastAPI keeps a router included with a lifespan nested in ``routes`` (the history
    router inside the channels router), so a flat scan would miss its paths.
    """
    paths: set[str] = set()
    for route in router.routes:
        nested = getattr(route, "original_router", None)
        if nested is not None:
            paths |= _paths(nested)
        else:
            paths.add(route.path)
    return paths


def test_routers_lists_the_two_slack_routers_with_cognees_prefix_and_tags():
    mounted = routers()

    assert [(prefix, tags) for _, prefix, tags in mounted] == [("/api/v1/slack", ["slack"])] * 2
    paths = set().union(*(_paths(router) for router, _, _ in mounted))
    assert {
        "/commands",
        "/events",
        "/interactive",
        "/channels",
        "/link",
        "/history/{team_id}/import",
        "/history/{team_id}/sync",
        "/history/{team_id}",
    } <= paths


def _signed(body: bytes, secret: str = _SECRET, timestamp: int | None = None) -> dict[str, str]:
    stamp = str(timestamp if timestamp is not None else int(time.time()))
    return {
        "X-Slack-Signature": sign_body(SignatureScheme.SLACK_V0, secret, body, timestamp=stamp),
        "X-Slack-Request-Timestamp": stamp,
        "Content-Type": "application/x-www-form-urlencoded",
    }


@pytest.fixture
def hook_client(monkeypatch):
    monkeypatch.setattr("cognee_connector_slack.settings.slack_settings.signing_secret", _SECRET)
    app = FastAPI()

    @app.post("/hook")
    async def hook(raw_body: bytes = Depends(verify_slack_request)):
        return {"body": raw_body.decode()}

    return TestClient(app)


def test_verify_slack_request_hands_the_raw_body_to_the_route(hook_client):
    body = b"command=%2Fcognee-ask&team_id=T123"

    response = hook_client.post("/hook", content=body, headers=_signed(body))

    assert response.status_code == 200
    assert response.json() == {"body": body.decode()}


def test_verify_slack_request_answers_401_invalid_slack_signature(hook_client):
    body = b"command=%2Fcognee-ask&team_id=T123"

    tampered = hook_client.post("/hook", content=body + b"&x=1", headers=_signed(body))
    unsigned = hook_client.post("/hook", content=body)
    stale = hook_client.post(
        "/hook", content=body, headers=_signed(body, timestamp=int(time.time()) - 600)
    )

    for response in (tampered, unsigned, stale):
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid Slack signature"
