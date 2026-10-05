"""Thin async client for Linear's GraphQL API.

Linear exposes a single GraphQL endpoint rather than REST resources, so this
module is one generic :func:`graphql` call plus a named wrapper for the one
mutation the agent loop depends on (:func:`create_agent_activity`). Every
call opens its own short-lived session — the same per-call
``aiohttp.ClientSession`` idiom as the GitHub adapter's ``app_auth`` — since
these fire from detached webhook handlers with no shared lifecycle to hook a
pooled session onto.

Error messages carry the operation name and HTTP status only — never the
access token, the variables, or the response body, any of which could
contain secret or user content that must not reach logs.
"""

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.linear.app/graphql"

_TIMEOUT = aiohttp.ClientTimeout(total=30)


class LinearUnauthorizedError(RuntimeError):
    """Linear answered 401: the token was rejected, whatever its stored expiry says."""


class LinearRateLimitedError(RuntimeError):
    """Linear answered HTTP 400 with the ``RATELIMITED`` code."""


async def _is_rate_limited(response: Any) -> bool:
    """Whether a 400 carries the RATELIMITED code. Only the code is read, never echoed."""
    try:
        body = await response.json()
    except Exception:  # noqa: BLE001 - an unreadable body is just a plain 400
        return False
    errors = body.get("errors") if isinstance(body, dict) else None
    return isinstance(errors, list) and any(
        isinstance(error, dict)
        and isinstance(error.get("extensions"), dict)
        and error["extensions"].get("code") == "RATELIMITED"
        for error in errors
    )


_AGENT_ACTIVITY_CREATE_MUTATION = """
mutation AgentActivityCreate($input: AgentActivityCreateInput!) {
  agentActivityCreate(input: $input) {
    success
  }
}
"""


def _operation_label(query: str) -> str:
    """A safe, short label for error messages — the operation header only.

    Deliberately not the full query (which could inline user content) and
    never the variables (which routinely do).
    """
    head = query.strip().split("(", 1)[0].split("{", 1)[0].strip()
    return head or "anonymous operation"


async def graphql(
    access_token: str, query: str, variables: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one GraphQL operation as the app user and return its ``data`` dict.

    Raises ``RuntimeError`` naming the operation on a non-200 response or a
    GraphQL-level ``errors`` array (Linear, like most GraphQL servers,
    returns those as HTTP 200).
    """
    operation = _operation_label(query)
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    async with (
        aiohttp.ClientSession(timeout=_TIMEOUT) as session,
        session.post(
            GRAPHQL_URL,
            json=payload,
            headers={"Authorization": f"Bearer {access_token}"},
        ) as response,
    ):
        if response.status == 401:
            raise LinearUnauthorizedError(f"Linear {operation} failed: HTTP 401")
        if response.status == 400 and await _is_rate_limited(response):
            # Linear answers a rate limit with HTTP 400 and a RATELIMITED code.
            raise LinearRateLimitedError(f"Linear {operation} failed: HTTP 400 RATELIMITED")
        if response.status != 200:
            raise RuntimeError(f"Linear {operation} failed: HTTP {response.status}")
        body: dict[str, Any] = await response.json()

    errors = body.get("errors")
    if errors:
        # Only stable machine codes, never ``errors[].message`` — GraphQL
        # validation messages echo the offending variable value ("Variable
        # '$input' got invalid value {...}"), which would put user/memory
        # content into the exception and thence into server logs.
        codes = sorted(
            {
                str(code)
                for error in errors
                if isinstance(error, dict)
                for code in ((error.get("extensions") or {}).get("code"), error.get("code"))
                if code
            }
        )
        detail = f"codes: {', '.join(codes)}" if codes else f"{len(errors)} GraphQL error(s)"
        raise RuntimeError(f"Linear {operation} failed: {detail}")

    return body.get("data") or {}


async def create_agent_activity(
    access_token: str, agent_session_id: str, content: dict[str, Any]
) -> None:
    """Emit one agent activity into a session.

    ``content`` is Linear's discriminated-union shape, e.g.
    ``{"type": "thought", "body": ...}`` / ``{"type": "response", "body": ...}``
    / ``{"type": "error", "body": ...}``. A "response" completes the turn;
    Linear tracks session lifecycle from the last emitted activity, which is
    why callers must always end a turn with a response or an error.
    """
    data = await graphql(
        access_token,
        _AGENT_ACTIVITY_CREATE_MUTATION,
        {"input": {"agentSessionId": agent_session_id, "content": content}},
    )
    if not (data.get("agentActivityCreate") or {}).get("success"):
        raise RuntimeError("Linear agentActivityCreate reported failure")
