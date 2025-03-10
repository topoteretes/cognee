#!/usr/bin/env python3
"""
Simple script to test the MCP initialization flow directly without Docker.
This can be used to debug initialization issues.
"""

import json
import subprocess
import sys
import time
import os
from typing import Dict, Any, Optional

# The initialization request as per the MCP protocol
INIT_REQUEST = {
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

# The initialized notification
INITIALIZED_NOTIFICATION = {
    "jsonrpc": "2.0",
    "method": "initialized"
}


def main():
    """Run the MCP initialization test."""
    print("=== Testing MCP Initialization Flow ===")
    
    # Find the cognee executable
    # Try different possible locations
    cognee_paths = [
        ".venv/bin/cognee",  # Local virtual environment
        os.path.expanduser("~/.venv/bin/cognee"),  # User's virtual environment
        os.path.expanduser("~/cognee/.venv/bin/cognee"),  # Cognee project virtual environment
        "cognee"  # System-wide installation
    ]
    
    cognee_cmd = None
    for path in cognee_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            cognee_cmd = path
            break
    
    if not cognee_cmd:
        print("ERROR: Could not find the cognee executable. Please make sure it's installed and in your PATH.")
        return 1
    
    print(f"Using cognee executable: {cognee_cmd}")
    
    # Prepare the command to run the MCP server
    cmd = [cognee_cmd]
    
    # Start the MCP server process
    print("Starting MCP server...")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1", MCP_LOG_LEVEL="INFO")
    )
    
    try:
        # Send the initialization request
        print("Sending initialization request...")
        init_request_str = json.dumps(INIT_REQUEST) + "\n"
        process.stdin.write(init_request_str)
        process.stdin.flush()
        
        # Wait for a response with timeout
        print("Waiting for response...")
        response = read_with_timeout(process.stdout, timeout=5)
        
        if not response:
            print("ERROR: No response received within timeout")
            return 1
        
        # Parse and validate the response
        try:
            response_obj = json.loads(response)
            print(f"Received response: {json.dumps(response_obj, indent=2)}")
            
            # Validate the response
            if response_obj.get("jsonrpc") != "2.0":
                print("ERROR: Invalid JSON-RPC version")
                return 1
            
            if response_obj.get("id") != 1:
                print("ERROR: Invalid response ID")
                return 1
            
            result = response_obj.get("result", {})
            
            # Check for protocolVersion instead of version
            if "protocolVersion" not in result:
                print("ERROR: Missing protocolVersion in response")
                return 1
            
            # Check for serverInfo instead of name/version directly
            if "serverInfo" not in result:
                print("ERROR: Missing serverInfo in response")
                return 1
            
            server_info = result.get("serverInfo", {})
            if "name" not in server_info:
                print("ERROR: Missing name in serverInfo")
                return 1
            
            if "version" not in server_info:
                print("ERROR: Missing version in serverInfo")
                return 1
            
            # Check for capabilities
            if "capabilities" not in result:
                print("ERROR: Missing capabilities in response")
                return 1
            
            # Send the initialized notification
            print("Sending initialized notification...")
            init_notification_str = json.dumps(INITIALIZED_NOTIFICATION) + "\n"
            process.stdin.write(init_notification_str)
            process.stdin.flush()
            
            # Wait a bit to ensure the notification is processed
            time.sleep(1)
            
            print("Initialization flow test passed!")
            return 0
            
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON response: {response}")
            print(f"JSON Error: {str(e)}")
            return 1
    
    finally:
        # Clean up
        print("Terminating MCP server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Print any stderr output
        stderr = process.stderr.read()
        if stderr:
            print(f"Server stderr output:\n{stderr}")


def read_with_timeout(stream, timeout: float) -> Optional[str]:
    """Read from a stream with a timeout."""
    import select
    
    # Wait for the stream to be readable
    ready, _, _ = select.select([stream], [], [], timeout)
    if not ready:
        return None
    
    # Read the response
    return stream.readline()


if __name__ == "__main__":
    sys.exit(main()) 