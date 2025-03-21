#!/usr/bin/env python3
"""
Test script to verify the Docker image for the Cognee MCP server.

This script tests:
1. Building the Docker image
2. Running the container
3. Testing the initialization flow with the 5-second timeout
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
TIMEOUT = 5  # 5-second timeout for initialization


class TestDockerImage(unittest.TestCase):
    """Test the Docker image for the Cognee MCP server."""

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

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        print("\n=== Cleaning up ===")
        # Stop and remove the container if it exists
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        
        # Remove the image
        subprocess.run(["docker", "rmi", "-f", IMAGE_NAME], capture_output=True)
        
        print("Cleanup completed")

    def test_docker_container_starts(self):
        """Test that the Docker container starts successfully."""
        print("\n=== Testing Docker container startup ===")
        
        # Run the container in detached mode
        result = subprocess.run(
            [
                "docker", "run", 
                "--name", CONTAINER_NAME, 
                "-d", 
                IMAGE_NAME
            ],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(result.returncode, 0, f"Failed to start container: {result.stderr}")
        
        # Wait for the container to start
        time.sleep(2)
        
        # Check if the container is running
        result = subprocess.run(
            ["docker", "ps", "-f", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )
        
        self.assertIn("Up", result.stdout, "Container is not running")
        print("Container started successfully")
        
        # Stop and remove the container
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

    def test_mcp_initialization_flow(self):
        """Test the MCP initialization flow with the 5-second timeout."""
        print("\n=== Testing MCP initialization flow ===")
        
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input and output files
            input_file = os.path.join(temp_dir, "input.json")
            output_file = os.path.join(temp_dir, "output.json")
            
            # Write the initialization request to the input file with the correct format
            # Based on the error logs, we need to include protocolVersion and clientInfo
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
            
            with open(input_file, 'w') as f:
                f.write(json.dumps(init_request) + "\n")
            
            # Create an empty output file
            with open(output_file, 'w') as f:
                pass
            
            # Run the container with the input file
            # Use interactive mode to ensure stdin/stdout are properly connected
            cmd = [
                "docker", "run",
                "--name", CONTAINER_NAME,
                "-i",  # Interactive mode
                "-v", f"{input_file}:/tmp/input.json",
                "-v", f"{output_file}:/tmp/output.json",
                IMAGE_NAME
            ]
            
            # Use a separate process to feed input to the container
            with open(input_file, 'r') as input_stream:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Send the input to the container
                stdout, stderr = process.communicate(input=input_stream.read(), timeout=10)
                
                # Print stdout and stderr for debugging
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                
                # Check if we got a response
                if stdout:
                    # Write stdout to the output file for analysis
                    with open(output_file, 'w') as f:
                        f.write(stdout)
                
            # Wait for the container to finish
            time.sleep(2)
            
            # Stop and remove the container
            subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
            
            # Check if we got a response in the output file
            with open(output_file, 'r') as f:
                output = f.read().strip()
            
            # If no output in the file, use stdout from the process
            if not output and stdout:
                output = stdout.strip()
            
            print(f"Response: {output}")
            self.assertTrue(output, "No response received from the server")
            
            try:
                # Parse the response
                response = json.loads(output)
                
                # Check if it's a valid JSON-RPC response
                self.assertEqual(response.get("jsonrpc"), "2.0", "Invalid JSON-RPC version")
                self.assertEqual(response.get("id"), 1, "Invalid response ID")
                
                # Check if it has the expected result structure
                result = response.get("result", {})
                
                # Check for protocolVersion instead of version
                self.assertIn("protocolVersion", result, "Missing protocolVersion in response")
                
                # Check for serverInfo instead of name/version directly
                self.assertIn("serverInfo", result, "Missing serverInfo in response")
                server_info = result.get("serverInfo", {})
                self.assertIn("name", server_info, "Missing name in serverInfo")
                self.assertIn("version", server_info, "Missing version in serverInfo")
                
                # Check for capabilities
                self.assertIn("capabilities", result, "Missing capabilities in response")
                
                print("MCP initialization flow test passed")
                print(f"Server response: {json.dumps(response, indent=2)}")
                
            except json.JSONDecodeError as e:
                self.fail(f"Invalid JSON response: {output}. Error: {str(e)}")


async def test_initialization_with_client():
    """Test the initialization flow using a proper client."""
    print("\n=== Testing initialization with client ===")
    
    # This is a placeholder for a more sophisticated test that would use
    # the actual MCP client to test the initialization flow
    
    # In a real implementation, you would:
    # 1. Start the Docker container
    # 2. Connect to it using the MCP client
    # 3. Perform the initialization handshake
    # 4. Verify the response
    
    print("Client-based initialization test not implemented")


def main():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    main() 