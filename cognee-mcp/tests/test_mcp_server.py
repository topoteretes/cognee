#!/usr/bin/env python3
"""
Test script for MCP server functionality.
This script tests the core functionality of the MCP server:
1. cognee.add
2. cognify
3. search
"""

import json
import os
import subprocess
import time
import unittest
import sys
import socket
from dotenv import load_dotenv

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file in the project root
env_file_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_file_path)

class TestMCPServer(unittest.TestCase):
    """Test class for MCP server functionality."""
    
    # Class variables for Docker container and image names
    IMAGE_NAME = "cognee-mcp:test"
    CONTAINER_NAME = "cognee-mcp-test"
    SERVER_PORT = 8080  # Use port 8080
    
    # Test data
    TEST_TEXT = "Artificial intelligence is the simulation of human intelligence by machines. Machine learning is a subset of AI that enables systems to learn from data."
    TEST_SEARCH_QUERY = "machine learning"
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Build the Docker image with .env file
        print("\n=== Building Docker image ===")
        
        # Verify .env file exists
        env_file_path = os.path.join(PROJECT_ROOT, ".env")
        if not os.path.exists(env_file_path):
            raise RuntimeError(f"Error: .env file not found at {env_file_path}")
        
        print(f"Using .env file at: {env_file_path}")
        
        # Clean up any existing test containers or images
        print("Cleaning up any existing test resources...")
        subprocess.run(f"docker stop {cls.CONTAINER_NAME} 2>/dev/null || true", shell=True)
        subprocess.run(f"docker rmi {cls.IMAGE_NAME} 2>/dev/null || true", shell=True)
        
        # Build the Docker image
        result = subprocess.run(
            ["docker", "build", "-t", cls.IMAGE_NAME, "."],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"Failed to build Docker image: {result.stderr}")
            sys.exit(1)
        
        print("Docker image built successfully")
        
        # Start the container
        print("\n=== Starting MCP server container ===")
        
        # Read environment variables from .env file
        env_vars = []
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        env_vars.append(f"-e {key}={value}")
                    except ValueError:
                        # Skip lines that don't have a key=value format
                        pass
        
        # Add additional environment variables
        env_vars.extend(["-e DEBUG=true", "-e PYTHONUNBUFFERED=1", "-e MCP_LOG_LEVEL=DEBUG"])
        
        # Start the container
        cmd = f"docker run -d --rm -p {cls.SERVER_PORT}:8080 {' '.join(env_vars)} --name {cls.CONTAINER_NAME} {cls.IMAGE_NAME}"
        print(f"Running command: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Failed to start container: {result.stderr}")
            sys.exit(1)
        
        # Verify container is running
        result = subprocess.run(
            f"docker ps | grep {cls.CONTAINER_NAME}",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if "Up" not in result.stdout:
            print("Container is not running. Checking logs...")
            logs = subprocess.run(
                f"docker logs {cls.CONTAINER_NAME}",
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"Container logs: {logs.stdout}")
            print(f"Container error logs: {logs.stderr}")
            raise RuntimeError("Container failed to start")
            
        print("Container started successfully")

        # Wait for the server to start
        print("\n=== Waiting for server to start ===")
        start_time = time.time()
        server_ready = False
        
        while time.time() - start_time < 30:  # Wait up to 30 seconds
            # Check if server is ready by getting logs
            result = subprocess.run(
                f"docker logs {cls.CONTAINER_NAME}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            print(f"Container logs: {result.stdout}")
            
            if "Starting Cognee MCP server" in result.stdout:
                server_ready = True
                break
                
            time.sleep(1)
            
        if not server_ready:
            print("Warning: Server startup message not found in logs")
            
        # Give the server additional time to fully initialize
        time.sleep(5)
        print("Server should be started now")
        
        # Test if the server is actually responding
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(("localhost", cls.SERVER_PORT))
            sock.close()
            print("Server is responding to connections")
        except Exception as e:
            print(f"Server is not responding to connections: {e}")
            # Get the latest logs
            result = subprocess.run(
                f"docker logs {cls.CONTAINER_NAME}",
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"Latest container logs: {result.stdout}")
            print(f"Latest container error logs: {result.stderr}")
            raise RuntimeError("Server is not responding to connections")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print("\n=== Cleaning up ===")
        
        # Get final logs from container for debugging
        try:
            result = subprocess.run(
                f"docker logs {cls.CONTAINER_NAME}",
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"Final container logs: {result.stdout}")
        except Exception as e:
            print(f"Error getting final logs: {e}")
        
        # Stop and remove the container
        subprocess.run(
            f"docker stop {cls.CONTAINER_NAME}",
            shell=True,
            check=False
        )
        print("Test container stopped")

    def _send_jsonrpc_request(self, method, params=None):
        """Send a JSON-RPC request to the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method
        }
        
        if params:
            request["params"] = params
            
        # Convert request to JSON
        request_json = json.dumps(request)
        
        # Create a socket connection to the container
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # Set a timeout to avoid hanging
        
        try:
            sock.connect(("localhost", self.SERVER_PORT))
            
            # Send the request
            sock.sendall(request_json.encode() + b'\n')
            
            # Receive the response
            response = b""
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                response += data
                if b'\n' in data:  # JSON-RPC responses end with newline
                    break
                    
            sock.close()
            
            # Parse the response
            if response:
                return json.loads(response.decode())
            return None
        except Exception as e:
            print(f"Error sending request: {e}")
            sock.close()
            raise

    def test_01_list_tools(self):
        """Test listing available tools."""
        print("\n=== Testing list_tools ===")
        
        response = self._send_jsonrpc_request("mcp/listTools")
        
        self.assertIsNotNone(response, "No response received")
        self.assertIn("result", response, "No result in response")
        
        tools = response["result"]
        self.assertIsInstance(tools, list, "Tools should be a list")
        
        # Check if the required tools are available
        tool_names = [tool["name"] for tool in tools]
        self.assertIn("cognify", tool_names, "Cognify tool not found")
        self.assertIn("search", tool_names, "Search tool not found")
        
        print("List tools test passed")

    def test_02_add_via_cognify(self):
        """Test the cognee.add functionality via the cognify tool."""
        print("\n=== Testing cognee.add via cognify tool ===")
        
        # The cognify tool internally calls cognee.add
        response = self._send_jsonrpc_request("mcp/callTool", {
            "name": "cognify",
            "arguments": {
                "text": "This is a test for cognee.add functionality."
            }
        })
        
        self.assertIsNotNone(response, "No response received")
        self.assertIn("result", response, "No result in response")
        
        result = response["result"]
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertGreater(len(result), 0, "Result should not be empty")
        
        # Check if the response contains the expected text content
        self.assertEqual(result[0]["type"], "text", "Response should be of type text")
        self.assertEqual(result[0]["text"], "Ingested", "Response should indicate successful ingestion")
        
        print("cognee.add test passed")
        
        # Wait for processing to complete
        time.sleep(5)

    def test_03_cognify(self):
        """Test the cognify tool."""
        print("\n=== Testing cognify tool ===")
        
        response = self._send_jsonrpc_request("mcp/callTool", {
            "name": "cognify",
            "arguments": {
                "text": self.TEST_TEXT
            }
        })
        
        self.assertIsNotNone(response, "No response received")
        self.assertIn("result", response, "No result in response")
        
        result = response["result"]
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertGreater(len(result), 0, "Result should not be empty")
        
        # Check if the response contains the expected text content
        self.assertEqual(result[0]["type"], "text", "Response should be of type text")
        self.assertEqual(result[0]["text"], "Ingested", "Response should indicate successful ingestion")
        
        print("Cognify test passed")
        
        # Wait for cognify to complete
        time.sleep(5)

    def test_04_search(self):
        """Test the search tool."""
        print("\n=== Testing search tool ===")
        
        response = self._send_jsonrpc_request("mcp/callTool", {
            "name": "search",
            "arguments": {
                "search_query": self.TEST_SEARCH_QUERY,
                "search_type": "INSIGHTS"
            }
        })
        
        self.assertIsNotNone(response, "No response received")
        self.assertIn("result", response, "No result in response")
        
        result = response["result"]
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertGreater(len(result), 0, "Result should not be empty")
        
        # Check if the response contains text content
        self.assertEqual(result[0]["type"], "text", "Response should be of type text")
        self.assertIsInstance(result[0]["text"], str, "Response text should be a string")
        
        # The search results should contain something related to machine learning
        self.assertIn("machine learning", result[0]["text"].lower(), "Search results should contain the search query")
        
        print("Search test passed")

if __name__ == "__main__":
    unittest.main() 