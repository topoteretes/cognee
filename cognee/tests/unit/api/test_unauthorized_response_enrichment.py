"""Unauthenticated 401s on /api/v1/* explain the auth posture (SDK-814).

fastapi-users rejects an unauthenticated request with a bare
``HTTPException(401)`` whose rendered detail is the stock "Unauthorized".
``cognee/api/client.py`` registers ``explain_generic_401`` to enrich exactly
that body with the resolved auth posture and the env vars that change it,
while leaving every other HTTPException (specific 401 details, other status
codes, non-API paths) byte-for-byte on FastAPI's stock behavior.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from cognee.api.client import GENERIC_401_HELP, explain_generic_401


def _app() -> FastAPI:
    """A miniature app wired exactly like cognee/api/client.py."""
    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, explain_generic_401)

    @app.get("/api/v1/datasets")
    async def guarded():
        # What fastapi-users' current_user dependency raises for a missing or
        # invalid token: a bare 401, no detail.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    @app.get("/api/v1/with-headers")
    async def guarded_with_headers():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.post("/api/v1/auth/token-exchange")
    async def specific_detail():
        # A 401 that already explains itself must not be rewritten.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="TOKEN_EXPIRED")

    @app.get("/metrics")
    async def outside_api():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    @app.get("/api/v1/missing")
    async def other_status():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return app


class TestGeneric401Enrichment:
    def test_bare_401_on_api_path_explains_the_posture(self):
        response = TestClient(_app()).get("/api/v1/datasets")

        assert response.status_code == 401
        detail = response.json()["detail"]
        assert "ENABLE_BACKEND_ACCESS_CONTROL" in detail
        assert "DEFAULT_USER_PASSWORD" in detail
        assert "auth posture:" in detail

    def test_help_text_names_both_remediations(self):
        assert "ENABLE_BACKEND_ACCESS_CONTROL=false" in GENERIC_401_HELP
        assert "DEFAULT_USER_PASSWORD" in GENERIC_401_HELP

    def test_401_headers_survive_enrichment(self):
        response = TestClient(_app()).get("/api/v1/with-headers")

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert "ENABLE_BACKEND_ACCESS_CONTROL" in response.json()["detail"]

    def test_401_with_specific_detail_is_untouched(self):
        response = TestClient(_app()).post("/api/v1/auth/token-exchange")

        assert response.status_code == 401
        assert response.json() == {"detail": "TOKEN_EXPIRED"}

    def test_bare_401_outside_api_prefix_is_untouched(self):
        response = TestClient(_app()).get("/metrics")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    def test_other_status_codes_keep_stock_behavior(self):
        response = TestClient(_app()).get("/api/v1/missing")

        assert response.status_code == 404
        assert response.json() == {"detail": "Dataset not found"}


class TestRealApplicationWiring:
    """A replica can pass while client.py is wired wrongly; check the real app."""

    def test_handler_is_registered_on_the_real_app(self):
        from cognee.api.client import app

        assert app.exception_handlers.get(StarletteHTTPException) is explain_generic_401
