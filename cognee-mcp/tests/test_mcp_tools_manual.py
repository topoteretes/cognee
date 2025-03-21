#!/usr/bin/env python3
"""
Simplified test script for MCP tools.
This script tests the functionality of the MCP server tools assuming a server is already running.
"""

import json
import os
import subprocess
import sys
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test MCP tools with a running server')
parser.add_argument('--host', default='localhost', help='Host where MCP server is running')
parser.add_argument('--port', default='8080', help='Port where MCP server is running')
parser.add_argument('--container', default='cognee-mcp-dev', help='Container name (if using Docker)')
parser.add_argument('--use-container', action='store_true', help='Send requests to Docker container instead of host:port')
parser.add_argument('--tool', choices=['list', 'cognify', 'search', 'codify', 'all'], default='all', help='Which tool to test')
args = parser.parse_args()

# Test data
TEST_TEXT = "Artificial intelligence is the simulation of human intelligence by machines. Machine learning is a subset of AI that enables systems to learn from data."
TEST_SEARCH_QUERY = "machine learning"
TEST_REPO_PATH = "/tmp/test_repo"  # Path inside container if using Docker

def send_request(request_data):
    """Send a request to the MCP server and get the response."""
    request_json = json.dumps(request_data)
    
    if args.use_container:
        # Send request to Docker container
        print(f"Sending request to container {args.container}...")
        
        # Create a temporary file with the request
        with open('/tmp/mcp_request.json', 'w') as f:
            f.write(request_json)
        
        # Send the request to the container
        try:
            # Try using cat to pipe to stdin
            result = subprocess.run(
                f"docker exec -i {args.container} bash -c 'cat > /proc/1/fd/0'",
                shell=True,
                input=request_json.encode('utf-8'),
                capture_output=True,
                text=True
            )
            print(f"Request sent. Output: {result.stdout}")
            print(f"Errors: {result.stderr}")
            
            # Get logs from container to find response
            time.sleep(2)  # Wait for processing
            result = subprocess.run(
                f"docker logs {args.container}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            # Parse logs to find response
            response = parse_response_from_logs(result.stdout, request_data.get('id'))
            return response
            
        except subprocess.CalledProcessError as e:
            print(f"Error sending request: {e}")
            return None
    else:
        # Send request to host:port using netcat
        print(f"Sending request to {args.host}:{args.port}...")
        try:
            result = subprocess.run(
                f"echo '{request_json}' | nc {args.host} {args.port}",
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"Request sent. Output: {result.stdout}")
            
            # Try to parse the response
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"Failed to parse response as JSON: {result.stdout}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error sending request: {e}")
            return None

def parse_response_from_logs(logs, request_id):
    """Parse the response from container logs."""
    log_lines = logs.strip().split('\n')
    
    # Search for response in logs
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
                    if 'id' in response and response['id'] == request_id:
                        return response
            except json.JSONDecodeError:
                continue
    
    return None

def test_list_tools():
    """Test listing available tools."""
    print("\n=== Testing list tools ===")
    
    list_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "mcp/listTools"
    }
    
    response = send_request(list_request)
    
    if response and 'result' in response:
        print("List tools test passed!")
        print(f"Available tools: {json.dumps(response['result'], indent=2)}")
        return True
    else:
        print("List tools test failed!")
        return False

def test_cognify():
    """Test the cognify tool."""
    print("\n=== Testing cognify tool ===")
    
    cognify_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "mcp/callTool",
        "params": {
            "name": "cognify",
            "arguments": {
                "text": TEST_TEXT
            }
        }
    }
    
    response = send_request(cognify_request)
    
    if response and 'result' in response:
        print("Cognify tool test passed!")
        print(f"Response: {json.dumps(response, indent=2)}")
        return True
    else:
        print("Cognify tool test failed!")
        return False

def test_search():
    """Test the search tool."""
    print("\n=== Testing search tool ===")
    
    search_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "mcp/callTool",
        "params": {
            "name": "search",
            "arguments": {
                "search_query": TEST_SEARCH_QUERY,
                "search_type": "INSIGHTS"
            }
        }
    }
    
    response = send_request(search_request)
    
    if response and 'result' in response:
        print("Search tool test passed!")
        print(f"Response: {json.dumps(response, indent=2)}")
        return True
    else:
        print("Search tool test failed!")
        return False

def test_codify():
    """Test the codify tool."""
    print("\n=== Testing codify tool ===")
    
    # If using Docker, create a test repo in the container
    if args.use_container:
        print(f"Creating test repo in container at {TEST_REPO_PATH}...")
        try:
            # Create directory in container
            subprocess.run(
                f"docker exec {args.container} mkdir -p {TEST_REPO_PATH}",
                shell=True,
                check=True
            )
            
            # Create a simple Python file
            python_code = "def hello():\n    print('Hello, world!')\n"
            subprocess.run(
                f"docker exec -i {args.container} bash -c 'cat > {TEST_REPO_PATH}/test.py'",
                shell=True,
                input=python_code.encode(),
                check=True
            )
            
            print(f"Test repo created in container")
        except subprocess.CalledProcessError as e:
            print(f"Error creating test repo: {e}")
            return False
    
    codify_request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "mcp/callTool",
        "params": {
            "name": "codify",
            "arguments": {
                "repo_path": TEST_REPO_PATH
            }
        }
    }
    
    response = send_request(codify_request)
    
    if response and 'result' in response:
        print("Codify tool test passed!")
        print(f"Response: {json.dumps(response, indent=2)}")
        return True
    else:
        print("Codify tool test failed!")
        return False

def main():
    """Run the tests."""
    print(f"Testing MCP tools with server at {args.host}:{args.port}")
    if args.use_container:
        print(f"Using Docker container: {args.container}")
    
    # Run the requested tests
    success = True
    
    if args.tool == 'list' or args.tool == 'all':
        if not test_list_tools():
            success = False
    
    if args.tool == 'cognify' or args.tool == 'all':
        if not test_cognify():
            success = False
    
    if args.tool == 'search' or args.tool == 'all':
        if not test_search():
            success = False
    
    if args.tool == 'codify' or args.tool == 'all':
        if not test_codify():
            success = False
    
    # Print summary
    print("\n=== Test Summary ===")
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed. Check the output above for details.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 