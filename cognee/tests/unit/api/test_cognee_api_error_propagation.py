"""A CogneeApiError must keep its declared status code on the way out.

Routers wrap their work in ``except Exception`` and answer with a generic error
response. Without a guard, a ``CogneeApiError`` raised inside that block is
caught by the same catch-all, so a 403 permission denial was answered as a 500
or a 409 and the global handler in ``cognee/api/client.py`` never ran. Callers
could not tell "you are not allowed to do this" from "we crashed".

Each such endpoint now re-raises ``CogneeApiError`` before its catch-all, and
the structural test below keeps new routers honest.
"""

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import cognee
import cognee.api.v1.add as add_package
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.exceptions import CogneeApiError
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user

ROUTERS = pathlib.Path(cognee.__file__).parent / "api" / "v1"
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "websocket"}

# Endpoints that must answer rather than propagate, each for a documented
# reason. Keyed by endpoint function name so the list survives edits above it.
EXEMPT = {
    # A liveness/readiness probe always reports a status; it never 500s.
    ("health/routers/get_health_router.py", "health_check"),
    ("health/routers/get_health_router.py", "detailed_health_check"),
    # The OAuth callback must redirect the browser, never render an error page.
    ("integrations/routers/get_integrations_router.py", "callback"),
}


def _handler_names(handler: ast.ExceptHandler) -> list[str]:
    node = handler.type
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Tuple):
        return [e.id if isinstance(e, ast.Name) else getattr(e, "attr", "") for e in node.elts]
    return []


def _is_endpoint(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr in HTTP_METHODS:
            return True
    return False


def _unguarded_endpoints() -> list[str]:
    offenders = []
    for path in sorted(ROUTERS.rglob("*.py")):
        tree = ast.parse(path.read_text())
        relative = path.relative_to(ROUTERS).as_posix()
        for function in ast.walk(tree):
            if not _is_endpoint(function):
                continue
            for block in ast.walk(function):
                if not isinstance(block, ast.Try):
                    continue
                catch_alls = [
                    h
                    for h in block.handlers
                    if h.type is None
                    or (
                        isinstance(h.type, ast.Name) and h.type.id in ("Exception", "BaseException")
                    )
                ]
                if not catch_alls:
                    continue
                catch_all = catch_alls[0]
                # Only blocks that answer the caller can hide the error. A
                # catch-all that re-raises is not a trap.
                if not any(isinstance(n, ast.Return) and n.value for n in ast.walk(catch_all)):
                    continue
                guarded = any(
                    "CogneeApiError" in _handler_names(h)
                    for h in block.handlers
                    if h is not catch_all
                )
                if not guarded and (relative, function.name) not in EXEMPT:
                    offenders.append(f"{relative}:{catch_all.lineno} {function.name}")
    return offenders


def test_every_catch_all_lets_a_cognee_api_error_through():
    offenders = _unguarded_endpoints()
    assert not offenders, (
        "These endpoints swallow a CogneeApiError into their generic error response, "
        "so its status code and message never reach the caller. Add "
        "`except CogneeApiError:\n    raise` before the catch-all, or add the endpoint "
        "to EXEMPT with a reason:\n  " + "\n  ".join(offenders)
    )


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/add")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4()
    )

    # The real handler lives in cognee/api/client.py; this mirrors the part
    # under test, that the exception's own status code is what goes out.
    @app.exception_handler(CogneeApiError)
    async def _handler(_, exc: CogneeApiError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    return TestClient(app)


def _post_add(client):
    return client.post(
        "/add",
        files={"data": ("note.txt", b"hello", "text/plain")},
        data={"datasetName": "demo"},
    )


def test_permission_denied_keeps_its_403(client, monkeypatch):
    monkeypatch.setattr(
        add_package, "add", AsyncMock(side_effect=PermissionDeniedError("no access"))
    )
    assert _post_add(client).status_code == 403


def test_an_unexpected_crash_is_still_a_500(client, monkeypatch):
    """The guard must not widen into swallowing real bugs differently."""
    monkeypatch.setattr(add_package, "add", AsyncMock(side_effect=RuntimeError("real bug")))
    assert _post_add(client).status_code == 500
