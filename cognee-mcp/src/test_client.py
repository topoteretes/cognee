#!/usr/bin/env python3
"""
Test client for Cognee MCP Server functionality.

This script tests all the tools and functions available in the Cognee MCP server,
including cognify, codify, search, prune, status checks, and utility functions.
"""

import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from cognee.shared.logging_utils import setup_logging


from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from src.server import (
    cognify,
    codify,
    search,
    prune,
    cognify_status,
    codify_status,
    cognee_add_developer_rules,
    node_to_string,
    retrieved_edges_to_string,
    load_class,
)

# Import MCP client functionality for server testing

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Set timeout for cognify/codify to complete in
TIMEOUT = 5 * 60  # 5 min  in seconds


class CogneeTestClient:
    """Test client for Cognee MCP Server functionality."""

    def __init__(self):
        self.test_results = {}
        self.temp_files = []

    async def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")

        # Create temporary test files
        self.test_data_dir = tempfile.mkdtemp(prefix="cognee_test_")

        # Create a test text file
        self.test_text_file = os.path.join(self.test_data_dir, "test.txt")
        with open(self.test_text_file, "w") as f:
            f.write(
                "This is a test document for Cognee testing. It contains information about AI and knowledge graphs."
            )

        # Create a test code repository structure
        self.test_repo_dir = os.path.join(self.test_data_dir, "test_repo")
        os.makedirs(self.test_repo_dir)

        # Create test Python files
        test_py_file = os.path.join(self.test_repo_dir, "main.py")
        with open(test_py_file, "w") as f:
            f.write("""
def hello_world():
    '''A simple hello world function.'''
    return "Hello, World!"

class TestClass:
    '''A test class for demonstration.'''
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"
""")

        # Create a test configuration file
        config_file = os.path.join(self.test_repo_dir, "config.py")
        with open(config_file, "w") as f:
            f.write("""
# Configuration settings
DATABASE_URL = "sqlite:///test.db"
DEBUG = True
""")

        # Create test developer rules files
        cursorrules_file = os.path.join(self.test_data_dir, ".cursorrules")
        with open(cursorrules_file, "w") as f:
            f.write("# Test cursor rules\nUse Python best practices.")

        self.temp_files.extend([self.test_text_file, test_py_file, config_file, cursorrules_file])
        print(f"✅ Test environment created at: {self.test_data_dir}")

    async def cleanup(self):
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")
        import shutil

        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
        print("✅ Cleanup completed")

    @asynccontextmanager
    async def mcp_server_session(self):
        """Context manager to start and manage MCP server session."""
        # Get the path to the server script
        server_script = os.path.join(os.path.dirname(__file__), "server.py")

        # Start the server process
        server_params = StdioServerParameters(
            command="python",
            args=[server_script, "--transport", "stdio"],
            env=None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                yield session

    async def test_mcp_server_startup_and_tools(self):
        """Test that the MCP server starts properly and returns tool results."""
        print("\n🧪 Testing MCP server startup and tool execution...")

        try:
            async with self.mcp_server_session() as session:
                # Test 1: List available tools
                print("  🔍 Testing tool discovery...")
                tools_result = await session.list_tools()

                expected_tools = {
                    "cognify",
                    "codify",
                    "search",
                    "prune",
                    "cognify_status",
                    "codify_status",
                    "cognee_add_developer_rules",
                }
                available_tools = {tool.name for tool in tools_result.tools}

                if not expected_tools.issubset(available_tools):
                    missing_tools = expected_tools - available_tools
                    raise AssertionError(f"Missing expected tools: {missing_tools}")

                print(
                    f"    ✅ Found {len(available_tools)} tools: {', '.join(sorted(available_tools))}"
                )

        except Exception as e:
            self.test_results["mcp_server_integration"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "MCP server integration test failed",
            }
            print(f"❌ MCP server integration test failed: {e}")

    async def test_prune(self):
        """Test the prune functionality."""
        print("\n🧪 Testing prune functionality...")
        try:
            result = await prune()
            self.test_results["prune"] = {
                "status": "PASS",
                "result": result,
                "message": "Prune executed successfully",
            }
            print("✅ Prune test passed")
        except Exception as e:
            self.test_results["prune"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "Prune test failed",
            }
            print(f"❌ Prune test failed: {e}")
            raise e

    async def test_cognify(self):
        """Test the cognify functionality."""
        print("\n🧪 Testing cognify functionality...")
        try:
            # Test with simple text
            test_text = "Artificial Intelligence is transforming the world through machine learning and deep learning technologies."
            cognify_result = await cognify(test_text)

            start = time.time()  # mark the start
            while True:
                try:
                    # Wait a moment
                    await asyncio.sleep(5)

                    # Check if cognify processing is finished
                    status_result = await cognify_status()
                    if str(PipelineRunStatus.DATASET_PROCESSING_COMPLETED) in status_result[0].text:
                        break
                    elif time.time() - start > TIMEOUT:
                        raise TimeoutError("Cognify did not complete in 5min")
                except DatabaseNotCreatedError:
                    if time.time() - start > TIMEOUT:
                        raise TimeoutError("Database was not created in 5min")

            self.test_results["cognify"] = {
                "status": "PASS",
                "result": cognify_result,
                "message": "Cognify executed successfully",
            }
            print("✅ Cognify test passed")

        except Exception as e:
            self.test_results["cognify"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "Cognify test failed",
            }
            print(f"❌ Cognify test failed: {e}")

    async def test_codify(self):
        """Test the codify functionality."""
        print("\n🧪 Testing codify functionality...")
        try:
            codify_result = await codify(self.test_repo_dir)

            start = time.time()  # mark the start
            while True:
                try:
                    # Wait a moment
                    await asyncio.sleep(5)

                    # Check if codify processing is finished
                    status_result = await codify_status()
                    if str(PipelineRunStatus.DATASET_PROCESSING_COMPLETED) in status_result[0].text:
                        break
                    elif time.time() - start > TIMEOUT:
                        raise TimeoutError("Codify did not complete in 5min")
                except DatabaseNotCreatedError:
                    if time.time() - start > TIMEOUT:
                        raise TimeoutError("Database was not created in 5min")

            self.test_results["codify"] = {
                "status": "PASS",
                "result": codify_result,
                "message": "Codify executed successfully",
            }
            print("✅ Codify test passed")

        except Exception as e:
            self.test_results["codify"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "Codify test failed",
            }
            print(f"❌ Codify test failed: {e}")

    async def test_cognee_add_developer_rules(self):
        """Test the cognee_add_developer_rules functionality."""
        print("\n🧪 Testing cognee_add_developer_rules functionality...")
        try:
            result = await cognee_add_developer_rules(base_path=self.test_data_dir)

            start = time.time()  # mark the start
            while True:
                try:
                    # Wait a moment
                    await asyncio.sleep(5)

                    # Check if developer rule cognify processing is finished
                    status_result = await cognify_status()
                    if str(PipelineRunStatus.DATASET_PROCESSING_COMPLETED) in status_result[0].text:
                        break
                    elif time.time() - start > TIMEOUT:
                        raise TimeoutError("Cognify of developer rules did not complete in 5min")
                except DatabaseNotCreatedError:
                    if time.time() - start > TIMEOUT:
                        raise TimeoutError("Database was not created in 5min")

            self.test_results["cognee_add_developer_rules"] = {
                "status": "PASS",
                "result": result,
                "message": "Developer rules addition executed successfully",
            }
            print("✅ Developer rules test passed")

        except Exception as e:
            self.test_results["cognee_add_developer_rules"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "Developer rules test failed",
            }
            print(f"❌ Developer rules test failed: {e}")

    async def test_search_functionality(self):
        """Test the search functionality with different search types."""
        print("\n🧪 Testing search functionality...")

        search_query = "What is artificial intelligence?"
        search_types = ["GRAPH_COMPLETION", "RAG_COMPLETION", "CODE", "CHUNKS", "INSIGHTS"]

        # Test if all search types will execute
        for search_type in search_types:
            try:
                result = await search(search_query, search_type)
                self.test_results[f"search_{search_type.lower()}"] = {
                    "status": "PASS",
                    "result": result,
                    "message": f"Search with {search_type} successful",
                }
                print(f"✅ Search {search_type} test passed")
            except Exception as e:
                self.test_results[f"search_{search_type.lower()}"] = {
                    "status": "FAIL",
                    "error": str(e),
                    "message": f"Search with {search_type} failed",
                }
                print(f"❌ Search {search_type} test failed: {e}")

    def test_utility_functions(self):
        """Test utility functions."""
        print("\n🧪 Testing utility functions...")

        # Test node_to_string
        try:
            test_node = {"id": "test_id", "name": "test_name", "type": "test_type"}
            result = node_to_string(test_node)
            expected = 'Node(id: "test_id", name: "test_name")'

            if result == expected:
                self.test_results["node_to_string"] = {
                    "status": "PASS",
                    "result": result,
                    "message": "node_to_string function works correctly",
                }
                print("✅ node_to_string test passed")
            else:
                self.test_results["node_to_string"] = {
                    "status": "FAIL",
                    "result": result,
                    "expected": expected,
                    "message": "node_to_string function output mismatch",
                }
                print(f"❌ node_to_string test failed: expected {expected}, got {result}")

        except Exception as e:
            self.test_results["node_to_string"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "node_to_string test failed",
            }
            print(f"❌ node_to_string test failed: {e}")

        # Test retrieved_edges_to_string
        try:
            test_triplet = [
                (
                    {"id": "node1", "name": "Node1"},
                    {"relationship_name": "CONNECTS_TO"},
                    {"id": "node2", "name": "Node2"},
                )
            ]
            result = retrieved_edges_to_string(test_triplet)
            expected = (
                'Node(id: "node1", name: "Node1") CONNECTS_TO Node(id: "node2", name: "Node2")'
            )
            if result == expected:
                self.test_results["retrieved_edges_to_string"] = {
                    "status": "PASS",
                    "result": result,
                    "message": "retrieved_edges_to_string function works correctly",
                }
                print("✅ retrieved_edges_to_string test passed")
            else:
                self.test_results["retrieved_edges_to_string"] = {
                    "status": "FAIL",
                    "result": result,
                    "expected": expected,
                    "message": "retrieved_edges_to_string function output mismatch",
                }
                print(
                    f"❌ retrieved_edges_to_string test failed: expected {expected}, got {result}"
                )

        except Exception as e:
            self.test_results["retrieved_edges_to_string"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "retrieved_edges_to_string test failed",
            }
            print(f"❌ retrieved_edges_to_string test failed: {e}")

    def test_load_class_function(self):
        """Test load_class function."""
        print("\n🧪 Testing load_class function...")

        try:
            # Create a temporary Python file with a test class
            test_module_file = os.path.join(self.test_data_dir, "test_model.py")
            with open(test_module_file, "w") as f:
                f.write("""
class TestModel:
    def __init__(self):
        self.name = "TestModel"

    def get_name(self):
        return self.name
""")

            # Test loading the class
            loaded_class = load_class(test_module_file, "TestModel")
            instance = loaded_class()

            if hasattr(instance, "get_name") and instance.get_name() == "TestModel":
                self.test_results["load_class"] = {
                    "status": "PASS",
                    "message": "load_class function works correctly",
                }
                print("✅ load_class test passed")
            else:
                self.test_results["load_class"] = {
                    "status": "FAIL",
                    "message": "load_class function did not load class correctly",
                }
                print("❌ load_class test failed: class not loaded correctly")

        except Exception as e:
            self.test_results["load_class"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "load_class test failed",
            }
            print(f"❌ load_class test failed: {e}")

    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Cognee MCP Server Test Suite")
        print("=" * 50)

        await self.setup()

        # Test MCP server integration first
        await self.test_mcp_server_startup_and_tools()

        # Run tests in logical order
        await self.test_prune()  # Start with clean slate
        await self.test_cognify()
        await self.test_codify()
        await self.test_cognee_add_developer_rules()

        await self.test_search_functionality()

        # Test utility functions (synchronous)
        self.test_utility_functions()
        self.test_load_class_function()

        await self.cleanup()

        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            if result["status"] == "PASS":
                status_emoji = "✅"
                passed += 1
            else:
                status_emoji = "❌"
                failed += 1

            print(f"{status_emoji} {test_name}: {result['status']}")

            if result["status"] == "FAIL" and "error" in result:
                print(f"   Error: {result['error']}")

        print("\n" + "-" * 50)
        total_tests = passed + failed
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed / total_tests * 100):.1f}%")

        assert failed == 0, "\n ⚠️ Number of tests didn't pass!"


async def main():
    """Main function to run the test suite."""
    client = CogneeTestClient()
    await client.run_all_tests()


if __name__ == "__main__":
    from logging import ERROR

    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
