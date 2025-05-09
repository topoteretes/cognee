#!/usr/bin/env python3
"""
Test script for Cognee's OpenAI-compatible Responses API
"""

import os
import json
import asyncio
import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your actual API URL
API_ENDPOINT = "/api/v1/responses/"  # Added trailing slash to match the server's redirection
AUTH_ENDPOINT = "/api/v1/auth/login"
# JWT token generated from get_token.py (valid for 1 hour from generation)
# Replace this with a new token if tests fail due to expiration
JWT_TOKEN = os.getenv(
    "COGNEE_JWT_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjc2MzU1NGMtOTFiZC00MzJjLWFiYTgtZDQyY2Q3MmVkNjU5IiwidGVuYW50X2lkIjoiNDUyMzU0NGQtODJiZC00MzJjLWFjYTctZDQyY2Q3MmVkNjUxIiwicm9sZXMiOlsiYWRtaW4iXSwiZXhwIjoxNzQ2NzM1NTg3fQ.fZtYlhg-7S8ikCNsjmAnYYpv9FQYWaXWgbYnTFkdek0"
)

# Note: Direct function tests using the tools parameter aren't working due to 
# issues with how the OpenAI client is processing the requests. However, we can test
# the API by using prompts that should trigger specific functions.


async def test_with_default_tools(token=None):
    """Test using the default tools provided by the API"""
    print("\n--- Testing the OpenAI-compatible Responses API ---")
    
    # Define payloads for different types of prompts that should trigger different functions
    payloads = [
        {
            "name": "General API capabilities",
            "payload": {
                "model": "cognee-v1",
                "input": "What can I do with this API?",
                "tool_choice": "auto"
            },
            "expected_function": None  # We don't expect any function call for this
        },
        {
            "name": "Search query",
            "payload": {
                "model": "cognee-v1",
                "input": "What information do we have about Python's asyncio module?",
                "tool_choice": "auto"
            },
            "expected_function": "search"  # We expect a search function call
        },
        {
            "name": "Cognify request",
            "payload": {
                "model": "cognee-v1",
                "input": "Please add this information to the knowledge graph: Python's asyncio module provides infrastructure for writing single-threaded concurrent code using coroutines.",
                "tool_choice": "auto"
            },
            "expected_function": "cognify_text"  # We expect a cognify_text function call
        }
    ]
    
    test_results = {}
    
    for test_case in payloads:
        print(f"\nTesting: {test_case['name']}")
        
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout
                response = await client.post(
                    f"{API_BASE_URL}{API_ENDPOINT}",
                    json=test_case["payload"],
                    headers=headers
                )
                
                print(f"Status code: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(json.dumps(result, indent=2))
                    
                    # Check for tool calls - handle both snake_case and camelCase property names
                    tool_calls = result.get("tool_calls", result.get("toolCalls", []))
                    
                    if tool_calls:
                        function_names = [tc["function"]["name"] for tc in tool_calls if "function" in tc and "name" in tc["function"]]
                        
                        expected_fn = test_case["expected_function"]
                        if expected_fn is None:
                            # No function expected
                            if not function_names:
                                test_pass = True
                                print(f"✅ {test_case['name']} test passed: No tool calls as expected")
                            else:
                                test_pass = False
                                print(f"❌ {test_case['name']} test failed: Expected no function calls, but got {function_names}")
                        else:
                            # Specific function expected
                            if expected_fn in function_names:
                                test_pass = True
                                print(f"✅ {test_case['name']} test passed: Expected function '{expected_fn}' was called")
                                
                                # If this is a cognify_text function, check for success status
                                if expected_fn == "cognify_text":
                                    for tc in tool_calls:
                                        if tc.get("function", {}).get("name") == "cognify_text":
                                            output = tc.get("output", {})
                                            if output.get("status") == "success":
                                                print(f"✅ cognify_text operation was successful")
                                            else:
                                                print(f"❌ cognify_text operation failed: {output}")
                                
                                # If this is a search function, check if we got results
                                if expected_fn == "search":
                                    for tc in tool_calls:
                                        if tc.get("function", {}).get("name") == "search":
                                            output = tc.get("output", {})
                                            results = output.get("data", {}).get("result", [])
                                            if results:
                                                print(f"✅ search operation returned {len(results)} results")
                                            else:
                                                print(f"⚠️ search operation did not return any results")
                            else:
                                test_pass = False
                                print(f"❌ {test_case['name']} test failed: Expected function '{expected_fn}' was not called. Got {function_names}")
                    else:
                        # No tool_calls in result
                        if test_case["expected_function"] is None:
                            test_pass = True
                            print(f"✅ {test_case['name']} test passed: No tool calls as expected")
                        else:
                            test_pass = False
                            print(f"❌ {test_case['name']} test failed: Expected function '{test_case['expected_function']}' but no tool calls were made")
                else:
                    test_pass = False
                    print(f"❌ Request failed: {response.text}")
        except Exception as e:
            test_pass = False
            print(f"❌ Exception during test: {str(e)}")
        
        test_results[test_case["name"]] = test_pass
    
    # Print summary
    print("\n=== TEST RESULTS SUMMARY ===")
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nPassed {passed}/{total} tests ({passed/total*100:.0f}%)")
    
    return passed == total


async def main():
    """Run all tests"""
    print("Starting Cognee Responses API Tests")
    
    # Use the JWT token for authentication
    token = JWT_TOKEN
    print(f"Using JWT token: {token[:20]}...")
    
    # Run tests with the token
    success = await test_with_default_tools(token)
    
    print("\nAll tests completed")
    
    # Return proper exit code for CI/CD pipelines
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    # Use exit code to signal test success/failure
    import sys
    sys.exit(exit_code) 