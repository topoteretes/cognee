#!/usr/bin/env python3
"""
Test script for MCP Docker tools.
This script tests the functionality of the MCP server tools using Docker.
"""

import json
import os
import subprocess
import tempfile
import time
import unittest
import shutil
import sys
from dotenv import load_dotenv

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file in the project root
env_file_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_file_path)

# Add parent directory to path to import from mcp_client
sys.path.append(os.path.join(PROJECT_ROOT, "..", "cognee", "tests"))
from mcp_client import MCPClient, MCPError

class TestMCPDockerTools(unittest.TestCase):
    """Test class for MCP Docker tools."""
    
    # Class variables for Docker container and image names
    IMAGE_NAME = "cognee-mcp:test"
    CONTAINER_NAME = "cognee-mcp-test"
    
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
        
        # Build the Docker image
        # The Dockerfile should copy the .env file during build
        result = subprocess.run(
            f"docker build -t {cls.IMAGE_NAME} {PROJECT_ROOT}",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        print("Docker image built successfully")

        # Read environment variables from .env file
        env_vars = {}
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
                    except ValueError:
                        pass
        
        # Add additional environment variables
        env_vars.update({
            "DEBUG": "true",
            "PYTHONUNBUFFERED": "1"
        })
        
        # Create environment string for docker run
        env_string = " ".join([f"-e {key}={value}" for key, value in env_vars.items() if value])
        
        print(f"Using environment variables: {env_string}")

        # Start the container with the MCP server
        print("\n=== Starting MCP server container ===")
        result = subprocess.run(
            f"docker run -d --rm -i {env_string} --name {cls.CONTAINER_NAME} --entrypoint python {cls.IMAGE_NAME} -m src.server",
            shell=True,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Container start failed: {result.stderr}"
        
        # Verify container is running
        result = subprocess.run(
            f"docker ps -f name={cls.CONTAINER_NAME} --format '{{{{.Status}}}}'",
            shell=True,
            capture_output=True,
            text=True
        )
        assert "Up" in result.stdout, "Container is not running"
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
            
            # Print the logs for debugging
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
        
        # Initialize the MCP client
        cls.client = None
        try:
            # Create a client that communicates with the container
            cls.client = cls._create_mcp_client()
            # Test the client with an initialization request
            response = cls._initialize_server()
            if not response or 'result' not in response:
                print(f"Warning: Server initialization response not as expected: {response}")
                # Continue anyway for testing purposes
        except Exception as e:
            print(f"Warning: Failed to initialize MCP client: {e}")
            # Continue anyway for testing purposes

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

    @classmethod
    def _create_mcp_client(cls):
        """Create an MCP client that communicates with the container."""
        # We're not directly using the MCPClient since it expects a local process
        # Instead, we'll use our custom _send_request_to_container method
        # But we return a placeholder to maintain the interface
        return "container_client"

    @classmethod
    def _initialize_server(cls):
        """Initialize the MCP server."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "experimental": {},
                    "tools": {
                        "listChanged": False
                    }
                }
            }
        }
        
        return cls._send_request_to_container(init_request)

    def test_docker_build(self):
        """Test that the Docker image can be built."""
        print("\n=== Testing Docker build ===")
        
        # Check if the image exists
        image_check = subprocess.run(
            f'docker images {self.IMAGE_NAME} --format "{{{{.Repository}}}}"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        self.assertTrue(len(image_check.stdout) > 0, "Docker image should exist")
        print("Docker build test passed")

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
                    "text": self.TEST_TEXT
                }
            }
        }
        
        # Send the request
        response = self._send_request_to_container(cognify_request)
        
        # Validate the response
        self.assertIsNotNone(response, "No response received from cognify tool")
        self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
        self.assertEqual(response.get("id"), 3, "Invalid response ID")
        self.assertIn("result", response, "Response should contain a result field")
        
        # Wait for cognify to finish processing (it might continue in the background)
        print("Waiting for cognify to finish processing...")
        time.sleep(10)  # Give it some time to process
        
        # Check if the container is still running
        container_check = subprocess.run(
            f'docker ps -f name={self.CONTAINER_NAME} --format "{{{{.Status}}}}"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        self.assertTrue(len(container_check.stdout) > 0, "Container should be running after cognify request")
        print("Cognify tool test passed")

    def test_search_after_cognify(self):
        """Test the search tool after cognify has processed text."""
        print("\n=== Testing search tool after cognify ===")
        
        # Create the search request
        search_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "mcp/callTool",
            "params": {
                "name": "search",
                "arguments": {
                    "search_query": self.TEST_SEARCH_QUERY,
                    "search_type": "INSIGHTS"
                }
            }
        }
        
        # Send the request
        response = self._send_request_to_container(search_request)
        
        # Validate the response
        self.assertIsNotNone(response, "No response received from search tool")
        self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
        self.assertEqual(response.get("id"), 5, "Invalid response ID")
        self.assertIn("result", response, "Response should contain a result field")
        
        # Check that the search results are not empty
        result = response.get("result", [])
        self.assertTrue(len(result) > 0, "Search results should not be empty")
        
        # Check if the container is still running
        container_check = subprocess.run(
            f'docker ps -f name={self.CONTAINER_NAME} --format "{{{{.Status}}}}"',
            shell=True,
            capture_output=True,
            text=True
        )
        
        self.assertTrue(len(container_check.stdout) > 0, "Container should be running after search request")
        print("Search tool test passed")

    def test_codify_tool(self):
        """Test the codify tool."""
        print("\n=== Testing codify tool ===")
        
        # Create a temporary directory with a simple Python file
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a simple Python file
            with open(os.path.join(temp_dir, "test.py"), "w") as f:
                f.write("def hello():\n    print('Hello, world!')\n")
            
            # Copy the directory to the container
            container_dir = "/tmp/test_repo"
            subprocess.run(
                f"docker exec {self.CONTAINER_NAME} mkdir -p {container_dir}",
                shell=True,
                check=True
            )
            
            # Use docker cp to copy the directory to the container
            subprocess.run(
                f"docker cp {temp_dir}/. {self.CONTAINER_NAME}:{container_dir}",
                shell=True,
                check=True
            )
            
            # Create the codify request
            codify_request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "mcp/callTool",
                "params": {
                    "name": "codify",
                    "arguments": {
                        "repo_path": container_dir
                    }
                }
            }
            
            # Send the request
            response = self._send_request_to_container(codify_request)
            
            # Validate the response
            self.assertIsNotNone(response, "No response received from codify tool")
            self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
            self.assertEqual(response.get("id"), 4, "Invalid response ID")
            self.assertIn("result", response, "Response should contain a result field")
            
            # Wait for codify to finish processing (it might continue in the background)
            print("Waiting for codify to finish processing...")
            time.sleep(10)  # Give it some time to process
            
            # Check if the container is still running
            container_check = subprocess.run(
                f'docker ps -f name={self.CONTAINER_NAME} --format "{{{{.Status}}}}"',
                shell=True,
                capture_output=True,
                text=True
            )
            
            self.assertTrue(len(container_check.stdout) > 0, "Container should be running after codify request")
            print("Codify tool test passed")
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

    @classmethod
    def _send_request_to_container(cls, request_data):
        """Send a request to the container and get the response."""
        request_json = json.dumps(request_data) + "\n"
        response_id = request_data.get('id')
        
        # Create a temporary file with the request
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(request_json)
            f.flush()
            temp_file = f.name
        
        try:
            # Copy the request file into the container
            result = subprocess.run(
                f"docker cp {temp_file} {cls.CONTAINER_NAME}:/tmp/request.json",
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"Copy result: {result.stdout}")
            
            # Send the request using cat and redirect to stdin
            subprocess.run(
                f"docker exec {cls.CONTAINER_NAME} bash -c 'cat /tmp/request.json > /proc/1/fd/0'",
                shell=True,
                check=True
            )
            
            # Wait for processing
            print(f"Waiting for response to request ID {response_id}...")
            
            # Try multiple times to find the response
            max_attempts = 5
            for attempt in range(max_attempts):
                time.sleep(2)  # Wait between attempts
                
                # Get logs from container
                result = subprocess.run(
                    f"docker logs {cls.CONTAINER_NAME}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                # Print the request for debugging
                print(f"Request sent (attempt {attempt+1}/{max_attempts}): {request_json.strip()}")
                
                # Split logs into lines and find the matching JSON response
                log_lines = result.stdout.strip().split('\n')
                
                # Print log line count for debugging
                print(f"Log lines count: {len(log_lines)}")
                
                # Print last few log lines for debugging
                last_lines = log_lines[-10:] if len(log_lines) > 10 else log_lines
                print(f"Last few log lines: {last_lines}")
                
                # Search for response
                for line in reversed(log_lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line contains a JSON object
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            # Try to parse as JSON
                            response = json.loads(line)
                            # Check if it's a JSON-RPC response
                            if isinstance(response, dict) and 'jsonrpc' in response:
                                # Check if it's a response to our request
                                if 'id' in response and response['id'] == response_id:
                                    print(f"Found matching response: {response}")
                                    return response
                        except json.JSONDecodeError:
                            continue
                
                print(f"No matching response found in attempt {attempt+1}, trying again...")
            
            # If no valid response found after all attempts
            print(f"No matching response found after {max_attempts} attempts")
            return None
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass

def main():
    """Run the tests."""
    # Clean up any existing resources
    subprocess.run(
        f"docker stop {TestMCPDockerTools.CONTAINER_NAME} 2>/dev/null || true",
        shell=True
    )
    subprocess.run(
        f"docker rmi {TestMCPDockerTools.IMAGE_NAME} 2>/dev/null || true",
        shell=True
    )
    
    # Run the tests in the correct order
    suite = unittest.TestSuite()
    suite.addTest(TestMCPDockerTools('test_docker_build'))
    suite.addTest(TestMCPDockerTools('test_cognify_tool'))
    suite.addTest(TestMCPDockerTools('test_search_after_cognify'))
    suite.addTest(TestMCPDockerTools('test_codify_tool'))
    
    # Run the tests
    result = unittest.TextTestRunner().run(suite)
    
    # Return the appropriate exit code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main() 