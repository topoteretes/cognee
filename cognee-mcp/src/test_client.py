#!/usr/bin/env python3
"""
Test client for Cognee MCP Server functionality.

This script tests all the tools and functions available in the Cognee MCP server,
including cognify, search, prune, status checks, and utility functions.

Usage:
    # Set your OpenAI API key first
    export OPENAI_API_KEY="your-api-key-here"

    # Run the test client
    python src/test_client.py

    # Or use LLM_API_KEY instead of OPENAI_API_KEY
    export LLM_API_KEY="your-api-key-here"
    python src/test_client.py
"""

import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from cognee.shared.logging_utils import setup_logging
from logging import ERROR, INFO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from src.server import (
    node_to_string,
    retrieved_edges_to_string,
    load_class,
)

# Set timeout for cognify to complete in
TIMEOUT = 5 * 60  # 5 min  in seconds


class CogneeTestClient:
    """Test client for Cognee MCP Server functionality."""

    def __init__(self):
        self.test_results = {}
        self.temp_files = []

    async def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")

        # Check for required API keys
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            print("⚠️  Warning: No OPENAI_API_KEY or LLM_API_KEY found in environment.")
            print("   Some tests may fail without proper LLM API configuration.")
            print("   Set OPENAI_API_KEY environment variable for full functionality.")
        else:
            print("✅ API key configured.")

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

        # Pass current environment variables to the server process
        # This ensures OpenAI API key and other config is available
        server_env = os.environ.copy()

        # Start the server process
        server_params = StdioServerParameters(
            command="python",
            args=[server_script, "--transport", "stdio"],
            env=server_env,
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
                    "search",
                    "prune",
                    "cognify_status",
                    "list_data",
                    "delete",
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
        """Test the prune functionality using MCP client."""
        print("\n🧪 Testing prune functionality...")
        try:
            async with self.mcp_server_session() as session:
                result = await session.call_tool("prune", arguments={})
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

    async def test_cognify(self, test_text, test_name):
        """Test the cognify functionality using MCP client."""
        print("\n🧪 Testing cognify functionality...")
        try:
            # Test with simple text using MCP client
            async with self.mcp_server_session() as session:
                cognify_result = await session.call_tool("cognify", arguments={"data": test_text})

                start = time.time()  # mark the start
                while True:
                    try:
                        # Wait a moment
                        await asyncio.sleep(5)

                        # Check if cognify processing is finished
                        status_result = await session.call_tool("cognify_status", arguments={})
                        if hasattr(status_result, "content") and status_result.content:
                            status_text = (
                                status_result.content[0].text
                                if status_result.content
                                else str(status_result)
                            )
                        else:
                            status_text = str(status_result)

                        if str(PipelineRunStatus.DATASET_PROCESSING_COMPLETED) in status_text:
                            break
                        elif time.time() - start > TIMEOUT:
                            raise TimeoutError("Cognify did not complete in 5min")
                    except DatabaseNotCreatedError:
                        if time.time() - start > TIMEOUT:
                            raise TimeoutError("Database was not created in 5min")

                self.test_results[test_name] = {
                    "status": "PASS",
                    "result": cognify_result,
                    "message": f"{test_name} executed successfully",
                }
                print(f"✅ {test_name} test passed")

        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAIL",
                "error": str(e),
                "message": f"{test_name} test failed",
            }
            print(f"❌ {test_name} test failed: {e}")

    async def test_search_functionality(self):
        """Test the search functionality with different search types using MCP client."""
        print("\n🧪 Testing search functionality...")

        search_query = "What is artificial intelligence?"

        # Test if all search types will execute
        from cognee import SearchType

        # Go through all Cognee search types
        for search_type in SearchType:
            # Don't test these search types
            if search_type in [
                SearchType.NATURAL_LANGUAGE,
                SearchType.CYPHER,
                SearchType.TRIPLET_COMPLETION,
            ]:
                break
            try:
                async with self.mcp_server_session() as session:
                    result = await session.call_tool(
                        "search",
                        arguments={"search_query": search_query, "search_type": search_type.value},
                    )
                    self.test_results[f"search_{search_type}"] = {
                        "status": "PASS",
                        "result": result,
                        "message": f"Search with {search_type} successful",
                    }
                    print(f"✅ Search {search_type} test passed")
            except Exception as e:
                self.test_results[f"search_{search_type}"] = {
                    "status": "FAIL",
                    "error": str(e),
                    "message": f"Search with {search_type} failed",
                }
                print(f"❌ Search {search_type} test failed: {e}")

    async def test_list_data(self):
        """Test the list_data functionality."""
        print("\n🧪 Testing list_data functionality...")

        try:
            async with self.mcp_server_session() as session:
                # Test listing all datasets
                result = await session.call_tool("list_data", arguments={})

                if result.content and len(result.content) > 0:
                    content = result.content[0].text

                    # Check if the output contains expected elements
                    if "Available Datasets:" in content or "No datasets found" in content:
                        self.test_results["list_data_all"] = {
                            "status": "PASS",
                            "result": content[:200] + "..." if len(content) > 200 else content,
                            "message": "list_data (all datasets) successful",
                        }
                        print("✅ list_data (all datasets) test passed")

                        # If there are datasets, try to list data for the first one
                        if "Dataset ID:" in content:
                            # Extract the first dataset ID from the output
                            lines = content.split("\n")
                            dataset_id = None
                            for line in lines:
                                if "Dataset ID:" in line:
                                    dataset_id = line.split("Dataset ID:")[1].strip()
                                    break

                            if dataset_id:
                                # Test listing data for specific dataset
                                specific_result = await session.call_tool(
                                    "list_data", arguments={"dataset_id": dataset_id}
                                )

                                if specific_result.content and len(specific_result.content) > 0:
                                    specific_content = specific_result.content[0].text
                                    if "Dataset:" in specific_content:
                                        self.test_results["list_data_specific"] = {
                                            "status": "PASS",
                                            "result": specific_content[:200] + "..."
                                            if len(specific_content) > 200
                                            else specific_content,
                                            "message": "list_data (specific dataset) successful",
                                        }
                                        print("✅ list_data (specific dataset) test passed")
                                    else:
                                        raise Exception(
                                            "Specific dataset listing returned unexpected format"
                                        )
                                else:
                                    raise Exception("Specific dataset listing returned no content")
                    else:
                        raise Exception("list_data returned unexpected format")
                else:
                    raise Exception("list_data returned no content")

        except Exception as e:
            self.test_results["list_data"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "list_data test failed",
            }
            print(f"❌ list_data test failed: {e}")

    async def test_delete(self):
        """Test the delete functionality."""
        print("\n🧪 Testing delete functionality...")

        try:
            async with self.mcp_server_session() as session:
                # First, let's get available data to delete
                list_result = await session.call_tool("list_data", arguments={})

                if not (list_result.content and len(list_result.content) > 0):
                    raise Exception("No data available for delete test - list_data returned empty")

                content = list_result.content[0].text

                # Look for data IDs and dataset IDs in the content
                lines = content.split("\n")
                dataset_id = None
                data_id = None

                for line in lines:
                    if "Dataset ID:" in line:
                        dataset_id = line.split("Dataset ID:")[1].strip()
                    elif "Data ID:" in line:
                        data_id = line.split("Data ID:")[1].strip()
                        break  # Get the first data item

                if dataset_id and data_id:
                    # Test soft delete (default)
                    delete_result = await session.call_tool(
                        "delete",
                        arguments={"data_id": data_id, "dataset_id": dataset_id, "mode": "soft"},
                    )

                    if delete_result.content and len(delete_result.content) > 0:
                        delete_content = delete_result.content[0].text

                        if "Delete operation completed successfully" in delete_content:
                            self.test_results["delete_soft"] = {
                                "status": "PASS",
                                "result": delete_content[:200] + "..."
                                if len(delete_content) > 200
                                else delete_content,
                                "message": "delete (soft mode) successful",
                            }
                            print("✅ delete (soft mode) test passed")
                        else:
                            # Check if it's an expected error (like document not found)
                            if "not found" in delete_content.lower():
                                self.test_results["delete_soft"] = {
                                    "status": "PASS",
                                    "result": delete_content,
                                    "message": "delete test passed with expected 'not found' error",
                                }
                                print("✅ delete test passed (expected 'not found' error)")
                            else:
                                raise Exception(
                                    f"Delete returned unexpected content: {delete_content}"
                                )
                    else:
                        raise Exception("Delete returned no content")

                else:
                    # Test with invalid UUIDs to check error handling
                    invalid_result = await session.call_tool(
                        "delete",
                        arguments={
                            "data_id": "invalid-uuid",
                            "dataset_id": "another-invalid-uuid",
                            "mode": "soft",
                        },
                    )

                    if invalid_result.content and len(invalid_result.content) > 0:
                        invalid_content = invalid_result.content[0].text

                        if "Invalid UUID format" in invalid_content:
                            self.test_results["delete_error_handling"] = {
                                "status": "PASS",
                                "result": invalid_content,
                                "message": "delete error handling works correctly",
                            }
                            print("✅ delete error handling test passed")
                        else:
                            raise Exception(f"Expected UUID error not found: {invalid_content}")
                    else:
                        raise Exception("Delete error test returned no content")

        except Exception as e:
            self.test_results["delete"] = {
                "status": "FAIL",
                "error": str(e),
                "message": "delete test failed",
            }
            print(f"❌ delete test failed: {e}")

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

        # Test cognify twice to make sure updating a dataset with new docs is working as expected
        await self.test_cognify(
            test_text="Artificial Intelligence is transforming the world through machine learning and deep learning technologies.",
            test_name="Cognify1",
        )
        await self.test_cognify(
            test_text="Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval.",
            test_name="Cognify2",
        )

        # Test list_data and delete functionality
        await self.test_list_data()
        await self.test_delete()

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

        if failed > 0:
            print(f"\n ⚠️ {failed} test(s) failed - review results above for details")


async def main():
    """Main function to run the test suite."""
    client = CogneeTestClient()
    await client.run_all_tests()


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
