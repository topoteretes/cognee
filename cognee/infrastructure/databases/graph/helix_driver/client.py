"""Async HTTP client for the HelixDB v2 dynamic-query gateway.

A thin wrapper around ``POST /v1/query``. Queries are hand-built JSON ASTs
(see ``ast.py``); this client only handles transport, auth, and error mapping.
There is no first-party Python SDK for HelixDB v2 — the dynamic JSON route is
the supported language-agnostic surface.
"""

from typing import Any, Dict, List, Optional

import httpx

from cognee.shared.logging_utils import get_logger

logger = get_logger("HelixClient")

# Optional request headers exposed by the Helix gateway.
_HEADER_AWAIT_DURABLE = "X-Helix-Await-Durable"
_HEADER_REQUIRE_WRITER = "X-Helix-Require-Writer"
_HEADER_WARM = "X-Helix-Warm"


class HelixQueryError(Exception):
    """Raised when the Helix gateway returns a non-success response."""

    def __init__(self, status_code: int, body: str, request_type: str) -> None:
        self.status_code = status_code
        self.body = body
        self.request_type = request_type
        super().__init__(f"Helix {request_type} query failed with status {status_code}: {body}")


class HelixClient:
    """Minimal async client for the HelixDB dynamic-query endpoint.

    The ``httpx.AsyncClient`` is created lazily on first use so the adapter can
    be constructed in a synchronous factory without an event loop.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise EnvironmentError(
                "HelixDB requires a base URL (e.g. http://localhost:6969). "
                "Set GRAPH_DATABASE_URL / VECTOR_DB_URL."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def query(
        self,
        request_type: str,
        queries: List[Dict[str, Any]],
        returns: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        parameter_types: Optional[Dict[str, Any]] = None,
        await_durable: bool = True,
        require_writer: bool = False,
    ) -> Dict[str, Any]:
        """Execute a dynamic query batch and return the parsed JSON response.

        Parameters
        ----------
        request_type: ``"read"`` or ``"write"`` (lowercase, per the gateway enum).
        queries: list of ``BatchEntry`` dicts (``{"Query": {...}}`` / ``{"ForEach": {...}}``).
        returns: names of variables to include in the response (``[]`` = all).
        await_durable / require_writer: optional write-consistency headers; only
            attached to write requests so cognify→search reads are consistent.
        """
        if request_type not in ("read", "write"):
            raise ValueError(f"request_type must be 'read' or 'write', got {request_type!r}")

        body: Dict[str, Any] = {
            "request_type": request_type,
            "query": {"queries": queries, "returns": returns},
        }
        if parameters is not None:
            body["parameters"] = parameters
        if parameter_types is not None:
            body["parameter_types"] = parameter_types

        headers: Dict[str, str] = {}
        if request_type == "write":
            if await_durable:
                headers[_HEADER_AWAIT_DURABLE] = "true"
            if require_writer:
                headers[_HEADER_REQUIRE_WRITER] = "true"

        client = self._ensure_client()
        response = await client.post("/v1/query", json=body, headers=headers or None)

        if response.status_code >= 400:
            raise HelixQueryError(response.status_code, response.text, request_type)

        if not response.content:
            return {}
        return response.json()

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
