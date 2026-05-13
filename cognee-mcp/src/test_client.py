#!/usr/bin/env python3
"""Smoke-test the public Cognee MCP memory tools."""

import asyncio
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


EXPECTED_TOOLS = {"remember", "recall", "forget"}


class CogneeTestClient:
    """Test client for the public Cognee MCP tool surface."""

    def __init__(self):
        self.test_results = {}

    @asynccontextmanager
    async def mcp_server_session(self):
        """Start the MCP server over stdio and yield an initialized client session."""
        server_script = os.path.join(os.path.dirname(__file__), "server.py")
        server_params = StdioServerParameters(
            command="python",
            args=[server_script, "--transport", "stdio"],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @staticmethod
    def _content_text(result) -> str:
        if not getattr(result, "content", None):
            return ""
        return "\n".join(getattr(item, "text", str(item)) for item in result.content)

    async def test_tool_discovery(self):
        """Verify the MCP server exposes only the supported memory tools."""
        print("\nTesting MCP tool discovery...")

        try:
            async with self.mcp_server_session() as session:
                tools_result = await session.list_tools()
                available_tools = {tool.name for tool in tools_result.tools}

            if available_tools != EXPECTED_TOOLS:
                missing = EXPECTED_TOOLS - available_tools
                unexpected = available_tools - EXPECTED_TOOLS
                raise AssertionError(
                    f"Tool surface mismatch. Missing: {sorted(missing)}. "
                    f"Unexpected: {sorted(unexpected)}."
                )

            self.test_results["tool_discovery"] = {
                "status": "PASS",
                "message": f"Found tools: {', '.join(sorted(available_tools))}",
            }
            print(f"PASS tool discovery: {', '.join(sorted(available_tools))}")
        except Exception as e:
            self.test_results["tool_discovery"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "Tool discovery failed",
            }
            print(f"FAIL tool discovery: {e}")

    async def test_memory_tools(self):
        """Exercise the three public tools without deleting existing memory."""
        print("\nTesting memory tool calls...")

        session_id = f"mcp-test-{uuid4().hex}"
        memory_text = f"Cognee MCP smoke memory {session_id}"

        try:
            async with self.mcp_server_session() as session:
                remember_result = await session.call_tool(
                    "remember",
                    arguments={"data": memory_text, "session_id": session_id},
                )
                recall_result = await session.call_tool(
                    "recall",
                    arguments={"query": session_id, "session_id": session_id, "top_k": 3},
                )
                forget_validation_result = await session.call_tool("forget", arguments={})

            remember_text = self._content_text(remember_result)
            recall_text = self._content_text(recall_result)
            forget_validation_text = self._content_text(forget_validation_result)

            if "Stored in session cache" not in remember_text:
                raise AssertionError(f"Unexpected remember response: {remember_text}")
            if session_id not in recall_text:
                raise AssertionError(f"Unexpected recall response: {recall_text}")
            if "Specify 'dataset' name or set 'everything' to true" not in forget_validation_text:
                raise AssertionError(
                    f"Unexpected forget validation response: {forget_validation_text}"
                )

            self.test_results["memory_tools"] = {
                "status": "PASS",
                "message": "remember, recall, and forget validation responded successfully",
            }
            print("PASS memory tools")
        except Exception as e:
            self.test_results["memory_tools"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "Memory tool test failed",
            }
            print(f"FAIL memory tools: {e}")

    async def run_all_tests(self):
        """Run the MCP smoke test suite."""
        print("Starting Cognee MCP memory tool smoke tests")
        print("=" * 50)

        await self.test_tool_discovery()
        await self.test_memory_tools()
        self.print_test_summary()

    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            if result["status"] == "PASS":
                passed += 1
            else:
                failed += 1
            print(f"{test_name}: {result['status']}")
            if result["status"] == "FAIL" and "error" in result:
                print(f"  Error: {result['error']}")

        print("\n" + "-" * 50)
        total_tests = passed + failed
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed / total_tests * 100):.1f}%")

        assert failed == 0, f"{failed} test(s) failed - review results above"


async def main():
    client = CogneeTestClient()
    await client.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
