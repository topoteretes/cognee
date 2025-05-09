#!/usr/bin/env python3
"""
Comprehensive test script for Cognee's OpenAI-compatible Responses API
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional, List
import sys

# Configuration
API_BASE_URL = os.getenv("COGNEE_API_URL", "http://localhost:8000")
API_ENDPOINT = "/api/v1/responses/"
AUTH_ENDPOINT = "/api/v1/auth/login"
EMAIL = os.getenv("COGNEE_EMAIL", "default_user@example.com")  # Default test user
PASSWORD = os.getenv("COGNEE_PASSWORD", "default_password")  # Default test password

# Constants
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def log_success(message: str) -> None:
    """Print a success message in green"""
    print(f"{GREEN}✅ {message}{RESET}")


def log_error(message: str) -> None:
    """Print an error message in red"""
    print(f"{RED}❌ {message}{RESET}")


def log_warning(message: str) -> None:
    """Print a warning message in yellow"""
    print(f"{YELLOW}⚠️ {message}{RESET}")


def log_info(message: str) -> None:
    """Print an info message"""
    print(f"ℹ️ {message}")


async def authenticate() -> Optional[str]:
    """Authenticate with the API and return access token"""
    log_info("Authenticating with the API...")
    
    auth_payload = {
        "email": EMAIL,
        "password": PASSWORD
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}{AUTH_ENDPOINT}",
                json=auth_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                auth_data = response.json()
                token = auth_data.get("access_token")
                if token:
                    log_success(f"Authentication successful")
                    return token
                else:
                    log_error("Authentication response did not contain access token")
                    return None
            else:
                log_error(f"Authentication failed with status {response.status_code}: {response.text}")
                return None
    except Exception as e:
        log_error(f"Authentication error: {str(e)}")
        return None


async def make_api_request(
    payload: Dict[str, Any], 
    token: Optional[str] = None, 
    expected_status: int = 200
) -> Dict[str, Any]:
    """Make a request to the API and return the response"""
    headers = {"Content-Type": "application/json"}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}{API_ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=60.0  # Increased timeout for cognify operations
            )
            
            log_info(f"Response status: {response.status_code}")
            
            if response.status_code == expected_status:
                if expected_status == 200:
                    result = response.json()
                    return result
                else:
                    return {"status": response.status_code, "text": response.text}
            else:
                log_error(f"Request failed with status {response.status_code}: {response.text}")
                return {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        log_error(f"Request error: {str(e)}")
        return {"error": str(e)}


def validate_response(response: Dict[str, Any]) -> bool:
    """Validate the response structure"""
    required_fields = ["id", "created", "model", "object", "status", "tool_calls"]
    
    missing_fields = [field for field in required_fields if field not in response]
    
    if missing_fields:
        log_error(f"Response missing required fields: {', '.join(missing_fields)}")
        return False
    
    if response["object"] != "response":
        log_error(f"Expected 'object' to be 'response', got '{response['object']}'")
        return False
    
    if not isinstance(response["tool_calls"], list):
        log_error(f"Expected 'tool_calls' to be a list, got {type(response['tool_calls'])}")
        return False
    
    for i, tool_call in enumerate(response["tool_calls"]):
        if "id" not in tool_call or "function" not in tool_call or "type" not in tool_call:
            log_error(f"Tool call {i} missing required fields")
            return False
        
        if "name" not in tool_call["function"] or "arguments" not in tool_call["function"]:
            log_error(f"Tool call {i} function missing required fields")
            return False
    
    return True


async def test_search_function(token: Optional[str] = None) -> bool:
    """Test the search function via the responses API"""
    log_info("\n--- Testing search function ---")
    
    # Define request payload
    payload = {
        "model": "cognee-v1",
        "input": "What information do we have about Python's asyncio module?",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information within the knowledge graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "The query to search for in the knowledge graph"
                            },
                            "search_type": {
                                "type": "string",
                                "description": "Type of search to perform",
                                "enum": ["INSIGHTS", "CODE", "GRAPH_COMPLETION", "SEMANTIC", "NATURAL_LANGUAGE"]
                            }
                        },
                        "required": ["search_query"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }
    
    result = await make_api_request(payload, token)
    
    if "error" in result:
        return False
    
    if not validate_response(result):
        return False
    
    # Check if we got tool calls
    if not result["tool_calls"]:
        log_warning("No tool calls found in response")
        return False
    
    search_tool_calls = [tc for tc in result["tool_calls"] 
                        if tc["function"]["name"] == "search"]
    
    if not search_tool_calls:
        log_error("No search tool calls found in response")
        return False
    
    log_success("Search function test passed")
    return True


async def test_cognify_function(token: Optional[str] = None) -> bool:
    """Test the cognify_text function via the responses API"""
    log_info("\n--- Testing cognify_text function ---")
    
    # Define request payload
    payload = {
        "model": "cognee-v1",
        "input": "Please add this information to the knowledge graph: Python's asyncio module provides infrastructure for writing single-threaded concurrent code using coroutines.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "cognify_text",
                    "description": "Convert text into a knowledge graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text content to be converted into a knowledge graph"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }
    
    result = await make_api_request(payload, token)
    
    if "error" in result:
        return False
    
    if not validate_response(result):
        return False
    
    # Check if we got tool calls
    if not result["tool_calls"]:
        log_warning("No tool calls found in response")
        return False
    
    cognify_tool_calls = [tc for tc in result["tool_calls"] 
                         if tc["function"]["name"] == "cognify_text"]
    
    if not cognify_tool_calls:
        log_error("No cognify_text tool calls found in response")
        return False
    
    # Check if output is successful
    for tool_call in cognify_tool_calls:
        if "output" in tool_call:
            output = tool_call["output"]
            if output.get("status") != "success":
                log_error(f"Cognify operation failed: {output}")
                return False
    
    log_success("Cognify function test passed")
    return True


async def test_with_default_tools(token: Optional[str] = None) -> bool:
    """Test using the default tools provided by the API"""
    log_info("\n--- Testing with default tools ---")
    
    # Define request payload - omitting tools to use defaults
    payload = {
        "model": "cognee-v1",
        "input": "What can I do with this API?",
        "tool_choice": "auto"
    }
    
    result = await make_api_request(payload, token)
    
    if "error" in result:
        return False
    
    if not validate_response(result):
        return False
    
    log_success("Default tools test passed")
    return True


async def test_invalid_request(token: Optional[str] = None) -> bool:
    """Test handling of invalid requests"""
    log_info("\n--- Testing invalid request handling ---")
    
    # Missing required parameter (model)
    payload = {
        "input": "What can I do with this API?"
    }
    
    result = await make_api_request(payload, token, expected_status=422)
    
    if "status_code" in result and result["status_code"] == 422:
        log_success("Invalid request properly rejected")
        return True
    else:
        log_error("Invalid request not properly rejected")
        return False


async def main():
    """Run all tests"""
    log_info("Starting Cognee Responses API Tests")
    
    # Get authentication token
    token = await authenticate()
    
    # Run tests
    results = {}
    
    # Basic functionality
    results["search_function"] = await test_search_function(token)
    results["cognify_function"] = await test_cognify_function(token)
    results["default_tools"] = await test_with_default_tools(token)
    
    # Error handling
    results["invalid_request"] = await test_invalid_request(token)
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print("-"*50)
    print(f"Tests passed: {passed}/{total} ({100 * passed / total:.1f}%)")
    
    if passed == total:
        log_success("\nAll tests passed! The OpenAI-compatible Responses API is working correctly.")
        return 0
    else:
        log_error("\nSome tests failed. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 