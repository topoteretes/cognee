"""T3 (#3361): MCP Docker image e2e over streamable HTTP (direct mode).

Drives the built ``cognee/cognee-mcp`` image through a real MCP session: boots it
with ``TRANSPORT_MODE=http``, connects an MCP client at ``/mcp``, asserts the tool
surface, exercises a ``remember`` -> ``recall`` round-trip, and validates
``MCP_ALLOWED_HOSTS`` / DNS-rebinding behaviour on the ``0.0.0.0`` bind.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from mcp_harness import mcp_client_session, run_mcp_http_container

pytestmark = pytest.mark.deployment

# Mirrors EXPECTED_TOOLS in cognee-mcp/tests/test_mcp_server_hardening.py and
# cognee-mcp/src/test_client.py.
EXPECTED_TOOLS = {
    "remember",
    "recall",
    "forget",
    "visualize_graph_ui",
    "upload_file_ui",
    "open_cognee_workspace",
    "cognify_file",
    "list_datasets_json",
    "list_dataset_data_json",
    "create_dataset_json",
    "get_client_info_json",
}

# Host-validation rejection statuses, tolerant across MCP SDK versions.
HOST_REJECTION_CODES = {400, 403, 421}

_MCP_POST_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _initialize_body(client_name: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "0.0.0"},
        },
    }


def _text_of(result) -> str:
    """Flatten an MCP tool-call result into plain text."""
    blocks = getattr(result, "content", None) or []
    return "\n".join(getattr(block, "text", str(block)) for block in blocks)


def _post_initialize_status(client: httpx.Client, url: str, host: str):
    """POST an initialize to ``/mcp`` with a given Host header.

    Returns the HTTP status code, or ``None`` when the request times out while
    reading the body. A timeout means the request passed Host validation and the
    server began streaming a response (a rejected Host is answered immediately
    with a small error body), so ``None`` is treated as "accepted".
    """
    try:
        response = client.post(
            url,
            headers={**_MCP_POST_HEADERS, "Host": host},
            json=_initialize_body("t3-host-probe"),
        )
    except httpx.TimeoutException:
        return None
    return response.status_code


def test_health_endpoint(mcp_http_container):
    """The container boots in HTTP mode and serves ``/health``."""
    response = httpx.get(mcp_http_container.health_url, timeout=10)
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


@pytest.mark.asyncio
async def test_list_tools_over_http(mcp_http_container):
    """A real MCP client at ``/mcp`` lists exactly the expected tool surface."""
    async with mcp_client_session(mcp_http_container.mcp_url) as session:
        tools = await session.list_tools()

    names = {tool.name for tool in tools.tools}
    assert names == EXPECTED_TOOLS, (
        f"Tool surface mismatch. "
        f"Missing: {sorted(EXPECTED_TOOLS - names)}. "
        f"Unexpected: {sorted(names - EXPECTED_TOOLS)}."
    )


@pytest.mark.asyncio
async def test_remember_recall_roundtrip(mcp_http_container):
    """``remember`` then ``recall`` returns the stored content.

    Uses session mode (keyword session cache) so the assertion targets stable,
    stored text rather than model-generated output.
    """
    session_id = f"t3-{uuid.uuid4().hex}"
    unique_token = f"T3_TOKEN_{uuid.uuid4().hex}"
    memory_text = f"Cognee T3 end-to-end memory {unique_token}"

    async with mcp_client_session(mcp_http_container.mcp_url) as session:
        remember_result = await session.call_tool(
            "remember",
            arguments={"data": memory_text, "session_id": session_id},
        )
        recall_result = await session.call_tool(
            "recall",
            arguments={"query": unique_token, "session_id": session_id, "top_k": 5},
        )

    remember_text = _text_of(remember_result)
    recall_text = _text_of(recall_result)

    assert "Stored in session cache" in remember_text, remember_text
    assert unique_token in recall_text, (
        f"Stored token {unique_token!r} not retrievable via recall. "
        f"recall returned: {recall_text!r}"
    )


def test_dns_rebinding_rejects_spoofed_host(mcp_http_container):
    """With the server bound to ``0.0.0.0`` a spoofed Host header is rejected.

    Only the Host header differs between the two requests, so loopback-accepted
    vs spoofed-rejected isolates the DNS-rebinding guard as the cause.
    """
    mcp_url = mcp_http_container.mcp_url

    with httpx.Client(timeout=10) as client:
        spoofed_status = _post_initialize_status(client, mcp_url, "evil.example.com")
        control_status = _post_initialize_status(
            client, mcp_url, f"127.0.0.1:{mcp_http_container.host_port}"
        )

    assert spoofed_status in HOST_REJECTION_CODES, (
        f"Spoofed Host was not rejected (status {spoofed_status})"
    )
    assert control_status not in HOST_REJECTION_CODES, (
        f"Loopback Host was rejected like a spoofed one (status {control_status})"
    )


def test_allowed_hosts_env_permits_configured_host(mcp_image):
    """``MCP_ALLOWED_HOSTS`` lets an otherwise-untrusted Host through.

    Boots a dedicated container with ``MCP_ALLOWED_HOSTS`` set and asserts the
    configured host is accepted while a still-unlisted host is rejected.
    """
    allowed_host = "cognee-mcp.test"

    with run_mcp_http_container(
        mcp_image,
        extra_env={"MCP_ALLOWED_HOSTS": f"{allowed_host}:*"},
        name_suffix="-allowedhosts",
    ) as container:
        with httpx.Client(timeout=10) as client:
            allowed_status = _post_initialize_status(
                client, container.mcp_url, f"{allowed_host}:{container.host_port}"
            )
            rejected_status = _post_initialize_status(
                client, container.mcp_url, "still-not-allowed.example.com"
            )

    assert rejected_status in HOST_REJECTION_CODES, (
        f"Unlisted Host was not rejected (status {rejected_status})"
    )
    assert allowed_status not in HOST_REJECTION_CODES, (
        f"MCP_ALLOWED_HOSTS host was rejected (status {allowed_status})"
    )
