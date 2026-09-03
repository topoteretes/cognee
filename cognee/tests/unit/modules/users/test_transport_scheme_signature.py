"""The auth schemes' ``__call__`` must stay annotated with a connection type.

Both transports subclass a FastAPI security scheme to add the WebSocket
``?token=`` fallback. FastAPI decides what to pass a dependency parameter by
its annotation: only a handful of types (Request, WebSocket, HTTPConnection,
Response, ...) are injected as the connection, and anything else is read as an
ordinary request field. So an override written as ``async def __call__(self,
request)`` -- no annotation -- silently becomes a *required query parameter
named "request"*, and every unauthenticated HTTP call answers 422 (missing
query param) instead of 401.

Nothing about that is visible in the transport's own tests: the scheme still
works when called directly with a connection object. It only shows up through
a route, which is why it survived once already and had to be fixed twice.

HTTPConnection rather than Request because the same scheme serves HTTP routes
and WebSocket handshakes, and it is the common base FastAPI injects for both.
"""

import inspect

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import HTTPConnection

from cognee.modules.users.authentication.api_bearer.api_bearer_transport import (
    api_bearer_transport,
)
from cognee.modules.users.authentication.api_key.get_api_key_transport import (
    get_api_key_transport,
)

# The types FastAPI injects as the connection itself rather than parsing out of
# the request. Kept as an explicit tuple: the point is to assert against a
# written-down expectation, not against whatever FastAPI happens to accept.
CONNECTION_TYPES = (HTTPConnection,)


def _schemes():
    yield "bearer", api_bearer_transport.scheme
    yield "api_key", get_api_key_transport().scheme


@pytest.mark.parametrize("name,scheme", list(_schemes()))
def test_scheme_call_annotates_its_connection_parameter(name, scheme):
    signature = inspect.signature(scheme.__call__)
    parameters = [p for p in signature.parameters.values() if p.name != "self"]
    assert len(parameters) == 1, f"{name}: expected one parameter, got {parameters}"

    annotation = parameters[0].annotation
    assert annotation is not inspect.Parameter.empty, (
        f"{name}: unannotated -- FastAPI will treat this as a query parameter"
    )
    assert annotation in CONNECTION_TYPES, (
        f"{name}: annotated {annotation!r}, expected one of {CONNECTION_TYPES}"
    )


@pytest.mark.parametrize("name,scheme", list(_schemes()))
def test_route_using_the_scheme_rejects_rather_than_failing_validation(name, scheme):
    """The behavioural half: a missing credential must not read as a bad request."""
    app = FastAPI()

    @app.get("/probe")
    async def probe(token=Depends(scheme)):
        # auto_error=False on both schemes, so a missing credential arrives as
        # None and the route -- not the scheme -- decides the status.
        return {"token": token}

    response = TestClient(app).get("/probe")

    assert response.status_code == 200, (
        f"{name}: {response.status_code} -- a 422 here means the connection "
        f"parameter was parsed as a request field: {response.text}"
    )
    assert response.json() == {"token": None}
