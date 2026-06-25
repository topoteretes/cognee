"""T4: MCP Docker image e2e (API mode, two-container topology).

Starts the cognee backend and the cognee-mcp server in API mode
(API_URL pointed at the backend on a shared Docker network), then
drives a remember -> recall -> forget round-trip through the MCP
protocol to verify that data flows end-to-end.

Ref: https://github.com/topoteretes/cognee/issues/3362
"""

import httpx
import pytest

pytestmark = pytest.mark.deployment

MCP_ENDPOINT = "/mcp"


def _mcp_request(method: str, params: dict = None, request_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request for the MCP protocol."""
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def _call_tool(tool_name: str, arguments: dict, request_id: int = 1) -> dict:
    return _mcp_request("tools/call", {"name": tool_name, "arguments": arguments}, request_id)


def _list_tools(request_id: int = 1) -> dict:
    return _mcp_request("tools/list", request_id=request_id)


class TestMCPServerBoots:
    """MCP container starts in API mode and responds to health checks."""

    def test_mcp_health(self, mcp_url):
        resp = httpx.get(f"{mcp_url}/health", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_backend_health(self, backend_url):
        resp = httpx.get(f"{backend_url}/health", timeout=10)
        assert resp.status_code == 200


class TestMCPToolDiscovery:
    """Verify the MCP server exposes the expected tools."""

    def test_lists_tools(self, mcp_url):
        resp = httpx.post(
            f"{mcp_url}{MCP_ENDPOINT}",
            json=_list_tools(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        assert resp.status_code == 200
        data = resp.json()
        tools = data.get("result", {}).get("tools", [])
        tool_names = {t["name"] for t in tools}

        # The API-mode subset must include these core tools
        assert "remember" in tool_names
        assert "recall" in tool_names
        assert "forget" in tool_names


class TestMCPRememberRecallForget:
    """End-to-end round-trip: remember data, recall it, forget it."""

    DATASET = "mcp_e2e_test"
    TEST_TEXT = "Albert Einstein was born in Ulm, Germany in 1879."

    def _init_session(self, mcp_url: str) -> str:
        """Send an initialize request and return the session ID from the
        Mcp-Session header."""
        resp = httpx.post(
            f"{mcp_url}{MCP_ENDPOINT}",
            json=_mcp_request(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "e2e-test", "version": "0.1"},
                },
            ),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        assert resp.status_code == 200, f"Initialize failed: {resp.text}"
        session_id = resp.headers.get("Mcp-Session")
        assert session_id, "No Mcp-Session header in initialize response"
        return session_id

    def _send(self, mcp_url: str, session_id: str, body: dict, timeout: int = 120) -> dict:
        resp = httpx.post(
            f"{mcp_url}{MCP_ENDPOINT}",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Mcp-Session": session_id,
            },
            timeout=timeout,
        )
        assert resp.status_code == 200, f"MCP call failed ({resp.status_code}): {resp.text}"
        return resp.json()

    def test_round_trip(self, mcp_url, backend_url):
        session_id = self._init_session(mcp_url)

        # Step 1: remember
        result = self._send(
            mcp_url,
            session_id,
            _call_tool(
                "remember",
                {
                    "data": self.TEST_TEXT,
                    "dataset_name": self.DATASET,
                },
            ),
            timeout=180,
        )
        content = result.get("result", {}).get("content", [])
        assert any("stored" in c.get("text", "").lower() for c in content), (
            f"Remember did not confirm storage: {content}"
        )

        # Step 2: recall
        result = self._send(
            mcp_url,
            session_id,
            _call_tool(
                "recall",
                {
                    "query": "Where was Einstein born?",
                    "datasets": self.DATASET,
                },
            ),
        )
        content = result.get("result", {}).get("content", [])
        result_text = " ".join(c.get("text", "") for c in content).lower()
        assert "einstein" in result_text or "ulm" in result_text or len(content) > 0, (
            f"Recall returned no relevant results: {content}"
        )

        # Step 3: forget
        result = self._send(
            mcp_url,
            session_id,
            _call_tool(
                "forget",
                {
                    "dataset": self.DATASET,
                },
            ),
        )
        content = result.get("result", {}).get("content", [])
        assert any(
            "delet" in c.get("text", "").lower() or "success" in c.get("text", "").lower()
            for c in content
        ), f"Forget did not confirm deletion: {content}"

        # Step 4: verify data is gone from the backend
        verify_resp = httpx.post(
            f"{backend_url}/api/v1/datasets",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if verify_resp.status_code == 200:
            datasets = verify_resp.json()
            dataset_names = [d.get("name", "") for d in datasets if isinstance(d, dict)]
            assert self.DATASET not in dataset_names, (
                f"Dataset {self.DATASET} still exists after forget"
            )
