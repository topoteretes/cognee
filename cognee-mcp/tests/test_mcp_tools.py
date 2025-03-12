#!/usr/bin/env python3
"""
Test script to verify the MCP server tools functionality.

This script tests:
1. Initialization flow
2. Cognify tool
3. Codify tool
4. Search tool
"""

import os
import json
import time
import subprocess
import asyncio
import tempfile
import unittest
import sys
from typing import Dict, Any, Tuple, Optional

# Constants
IMAGE_NAME = "cognee-mcp:test"
CONTAINER_NAME = "cognee-mcp-test"
TIMEOUT = 10  # Timeout for tool operations


class TestMCPTools(unittest.TestCase):
    """Test the MCP server tools functionality."""

    @classmethod
    def setUpClass(cls):
        """Build the Docker image before running tests."""
        print("\n=== Building Docker image ===")
        result = subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "."],
            cwd=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"Failed to build Docker image: {result.stderr}")
            sys.exit(1)
        
        print("Docker image built successfully")
        
        # Start the container for all tests
        print("\n=== Starting MCP server container ===")
        cls.container_id = cls._start_container()
        
        # Initialize the server
        print("\n=== Initializing MCP server ===")
        cls._initialize_server()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        print("\n=== Cleaning up ===")
        # Stop and remove the container if it exists
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        
        # Remove the image
        subprocess.run(["docker", "rmi", "-f", IMAGE_NAME], capture_output=True)
        
        print("Cleanup completed")

    @classmethod
    def _start_container(cls) -> str:
        """Start the MCP server container and return its ID."""
        result = subprocess.run(
            [
                "docker", "run", 
                "--name", CONTAINER_NAME, 
                "-d",
                "-i",  # Interactive mode
                IMAGE_NAME
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        
        # Wait for the container to start
        time.sleep(2)
        
        # Check if the container is running
        result = subprocess.run(
            ["docker", "ps", "-f", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )
        
        if "Up" not in result.stdout:
            raise RuntimeError("Container is not running")
        
        print("Container started successfully")
        return result.stdout.strip()

    @classmethod
    def _initialize_server(cls):
        """Initialize the MCP server with the initialization request."""
        # The initialization request as per the MCP protocol
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "resources": {},
                    "tools": {},
                    "prompts": {},
                    "roots": {},
                    "sampling": {}
                }
            }
        }
        
        # Send the initialization request
        response = cls._send_request_to_container(init_request)
        
        # Validate the response
        if response.get("jsonrpc") != "2.0" or response.get("id") != 1:
            raise RuntimeError("Invalid initialization response")
        
        # Send the initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "initialized"
        }
        
        cls._send_request_to_container(initialized_notification, expect_response=False)
        print("Server initialized successfully")

    @classmethod
    def _send_request_to_container(cls, request: Dict[str, Any], expect_response: bool = True) -> Optional[Dict[str, Any]]:
        """Send a request to the container and return the response."""
        # Create a temporary file for the request
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as request_file:
            request_file.write(json.dumps(request) + "\n")
            request_file.flush()
            request_path = request_file.name
        
        try:
            # Send the request to the container
            result = subprocess.run(
                [
                    "docker", "exec", 
                    "-i",  # Interactive mode
                    CONTAINER_NAME,
                    "sh", "-c", f"cat {request_path} > /proc/1/fd/0"
                ],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to send request to container: {result.stderr}")
            
            if not expect_response:
                return None
            
            # Wait for the response
            time.sleep(1)
            
            # Read the response from the container
            result = subprocess.run(
                [
                    "docker", "logs",
                    CONTAINER_NAME
                ],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to read response from container: {result.stderr}")
            
            # Parse the response
            response_lines = result.stdout.strip().split("\n")
            if not response_lines:
                raise RuntimeError("No response received from container")
            
            # The response should be the last line
            response_json = response_lines[-1]
            return json.loads(response_json)
        
        finally:
            # Clean up the temporary file
            os.unlink(request_path)

    def test_list_tools(self):
        """Test listing available tools."""
        print("\n=== Testing list_tools ===")
        
        # Create the list_tools request
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "mcp/listTools"
        }
        
        # Send the request
        response = self._send_request_to_container(list_tools_request)
        
        # Validate the response
        self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
        self.assertEqual(response.get("id"), 2, "Invalid response ID")
        
        # Check if the response contains the tools
        result = response.get("result", [])
        self.assertIsInstance(result, list, "Result should be a list")
        
        # Check if the expected tools are in the list
        tool_names = [tool.get("name") for tool in result]
        self.assertIn("cognify", tool_names, "Cognify tool not found")
        self.assertIn("codify", tool_names, "Codify tool not found")
        self.assertIn("search", tool_names, "Search tool not found")
        
        print("List tools test passed")

    def test_cognify_tool(self):
        """Test the cognify tool."""
        print("\n=== Testing cognify tool ===")
        
        # Create the cognify request
        cognify_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "mcp/callTool",
            "params": {
                "name": "cognify",
                "arguments": {
                    "text": "Artificial intelligence is the simulation of human intelligence by machines."
                }
            }
        }
        
        # Send the request
        response = self._send_request_to_container(cognify_request)
        
        # Validate the response
        self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
        self.assertEqual(response.get("id"), 3, "Invalid response ID")
        
        # Check if the response contains the result
        result = response.get("result", [])
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(len(result) > 0, "Result should not be empty")
        
        # Check if the result contains the expected content
        content = result[0]
        self.assertEqual(content.get("type"), "text", "Content type should be text")
        self.assertEqual(content.get("text"), "Ingested", "Content text should be 'Ingested'")
        
        print("Cognify tool test passed")

    def test_codify_tool(self):
        """Test the codify tool."""
        print("\n=== Testing codify tool ===")
        
        # Create a temporary directory with a simple Python file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple Python file
            with open(os.path.join(temp_dir, "test.py"), "w") as f:
                f.write("def hello():\n    print('Hello, world!')\n")
            
            # Create the codify request
            codify_request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "mcp/callTool",
                "params": {
                    "name": "codify",
                    "arguments": {
                        "repo_path": temp_dir
                    }
                }
            }
            
            # Send the request
            response = self._send_request_to_container(codify_request)
            
            # Validate the response
            self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
            self.assertEqual(response.get("id"), 4, "Invalid response ID")
            
            # Check if the response contains the result
            result = response.get("result", [])
            self.assertIsInstance(result, list, "Result should be a list")
            self.assertTrue(len(result) > 0, "Result should not be empty")
            
            # Check if the result contains the expected content
            content = result[0]
            self.assertEqual(content.get("type"), "text", "Content type should be text")
            self.assertEqual(content.get("text"), "Indexed", "Content text should be 'Indexed'")
            
            print("Codify tool test passed")

    def test_search_tool(self):
        """Test the search tool."""
        print("\n=== Testing search tool ===")
        
        # First, add some content to search
        self.test_cognify_tool()
        
        # Create the search request
        search_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "mcp/callTool",
            "params": {
                "name": "search",
                "arguments": {
                    "search_query": "artificial intelligence",
                    "search_type": "INSIGHTS"
                }
            }
        }
        
        # Send the request
        response = self._send_request_to_container(search_request)
        
        # Validate the response
        self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
        self.assertEqual(response.get("id"), 5, "Invalid response ID")
        
        # Check if the response contains the result
        result = response.get("result", [])
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertTrue(len(result) > 0, "Result should not be empty")
        
        # Check if the result contains the expected content
        content = result[0]
        self.assertEqual(content.get("type"), "text", "Content type should be text")
        
        print("Search tool test passed")


def main():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    main() 