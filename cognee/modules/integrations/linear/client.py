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
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.linear.app/graphql"

_TIMEOUT = aiohttp.ClientTimeout(total=30)

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
    access_token: str, query: str, variables: Optional[dict[str, Any]] = None
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

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.post(
            GRAPHQL_URL,
            json=payload,
            headers={"Authorization": f"Bearer {access_token}"},
        ) as response:
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
