"""HTTP transport: the worker's ``POST /v1/extract`` and ``GET /readyz``.

Uses httpx (a core dependency), so this transport needs no extra.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from .errors import (
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from .protocol import (
    WorkerLimits,
    WorkerReadiness,
    WorkerReply,
    parse_json_reply,
    redact_url,
)

RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
REJECTED_STATUSES = frozenset({400, 413, 422})
UNAUTHORIZED_STATUSES = frozenset({401, 403})
# The readiness probe is a cheap GET; it should not wait as long as inference.
READINESS_TIMEOUT_SECS = 30.0
# How much of an error body is quoted back in the error message.
ERROR_DETAIL_CHARS = 1024


def _retry_after(response: httpx.Response) -> float | None:
    """Delta-seconds ``Retry-After``; an HTTP date is ignored."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _detail(body: bytes) -> str:
    text = body[:ERROR_DETAIL_CHARS].decode("utf-8", errors="replace").strip()
    return f": {text}" if text else ""


class HttpWorkerTransport:
    name = "http"

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None,
        connect_timeout: float,
        request_timeout: float,
        max_response_bytes: int,
        client: httpx.AsyncClient | None = None,
    ):
        split = urlsplit(endpoint)
        if split.scheme not in ("http", "https") or not split.netloc:
            raise GlinerRemoteConfigError(
                f"COGNEE_GLINER_ENDPOINT must be an http(s) URL for the HTTP transport, "
                f"got {redact_url(endpoint)!r}."
            )
        self.endpoint_label = redact_url(endpoint)
        self._max_response_bytes = max_response_bytes
        self._headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._readiness_timeout = httpx.Timeout(
            min(request_timeout, READINESS_TIMEOUT_SECS), connect=connect_timeout
        )
        self._client = client or httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            # One pooled connection per in-flight request; the adapter's budget
            # bounds how many there are.
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
        )

    async def extract(self, payload: Mapping[str, Any], *, request_id: str) -> WorkerReply:
        headers = {**self._headers, "x-request-id": request_id}
        status, body, retry_after = await self._send(
            "POST", "/v1/extract", json=payload, headers=headers
        )
        if status == 200:
            return parse_json_reply(body)
        raise self._error(status, body, retry_after)

    async def readiness(self) -> WorkerReadiness:
        status, body, retry_after = await self._send(
            "GET", "/readyz", headers=self._headers, timeout=self._readiness_timeout
        )
        if status != 200:
            raise self._error(status, body, retry_after)
        return self._parse_readiness(body)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(self, method: str, path: str, **kwargs) -> tuple[int, bytes, float | None]:
        try:
            async with self._client.stream(method, path, **kwargs) as response:
                body = await self._read_capped(response)
                return response.status_code, body, _retry_after(response)
        except httpx.TransportError as error:
            # Unreachable, refused, reset, timed out, or a broken stream: all
            # transient from here, and inference is safe to resend.
            raise GlinerWorkerUnavailableError(
                f"Cannot reach the GLiNER worker at {self.endpoint_label} "
                f"({type(error).__name__}{': ' + str(error) if str(error) else ''})."
            ) from error

    async def _read_capped(self, response: httpx.Response) -> bytes:
        cap = self._max_response_bytes
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > cap:
            raise GlinerWorkerRuntimeError(
                f"The GLiNER worker's reply ({declared} bytes) exceeds the {cap}-byte cap "
                "(COGNEE_GLINER_MAX_RESPONSE_BYTES)."
            )
        received = bytearray()
        async for part in response.aiter_bytes():
            received += part
            if len(received) > cap:
                raise GlinerWorkerRuntimeError(
                    f"The GLiNER worker's reply exceeds the {cap}-byte cap "
                    "(COGNEE_GLINER_MAX_RESPONSE_BYTES)."
                )
        return bytes(received)

    def _error(self, status: int, body: bytes, retry_after: float | None) -> Exception:
        where = f"The GLiNER worker at {self.endpoint_label} answered {status}{_detail(body)}"
        if status in RETRYABLE_STATUSES:
            return GlinerWorkerUnavailableError(where, retry_after=retry_after)
        if status in UNAUTHORIZED_STATUSES:
            return GlinerWorkerUnauthorizedError(where)
        if status in REJECTED_STATUSES:
            return GlinerWorkerRejectedError(where)
        # 500 is documented as deterministic (inference failed); anything else
        # unexpected is treated the same way rather than retried blindly.
        return GlinerWorkerRuntimeError(where)

    def _parse_readiness(self, body: bytes) -> WorkerReadiness:
        """Strict: a 200 from something that is not a worker must not count as ready."""
        import json

        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if (
            not isinstance(data, dict)
            or data.get("status") != "ready"
            or not isinstance(data.get("model"), str)
        ):
            raise GlinerWorkerRuntimeError(
                f"{self.endpoint_label}/readyz did not answer like a GLiNER worker "
                "(expected status 'ready' and a model)."
            )
        limits = data.get("limits") if isinstance(data.get("limits"), dict) else {}

        def limit(name: str) -> int | None:
            value = limits.get(name)
            return value if isinstance(value, int) and value > 0 else None

        features = data.get("features")
        return WorkerReadiness(
            model=data["model"],
            limits=WorkerLimits(
                max_inputs=limit("max_inputs"),
                max_text_chars=limit("max_text_chars"),
                max_batch_size=limit("max_batch_size"),
            ),
            # HTTP's probe is authoritative: no list means a worker without
            # optional features.
            features=frozenset(f for f in features if isinstance(f, str))
            if isinstance(features, list)
            else frozenset(),
        )
