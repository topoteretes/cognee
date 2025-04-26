#!/usr/bin/env python
"""
Script to test if Cognee tools are working correctly.
Run this script to test if the tools are correctly importing and functioning.
"""

import os
import cognee
from src.latest_ai_development.tools import CogneeAdd, CogneeSearch

# Set COGNEE_API_KEY if not already set
if "LLM_API_KEY" not in os.environ:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        os.environ["LLM_API_KEY"] = openai_api_key


def test_tools():
    """Test the CogneeAdd and CogneeSearch tools."""
    print("Testing Cognee tools...")

    print("\nTesting CogneeAdd tool...")
    add_tool = CogneeAdd()
    test_input = (
        "This is a test text to add to Cognee memory. It contains information about AI LLMs."
    )
    node_set = ["AI", "LLMs"]
    try:
        result = add_tool._run(context=test_input, node_set=node_set)
        print(f"CogneeAdd result: {result}")
    except Exception as e:
        print(f"Error testing CogneeAdd: {str(e)}")

    print("\nTesting CogneeSearch tool...")
    search_tool = CogneeSearch()
    search_query = "AI LLMs"
    node_set = ["AI"]
    try:
        result = search_tool._run(query_text=search_query, node_set=node_set)
        print(f"CogneeSearch result: {result}")
    except Exception as e:
        print(f"Error testing CogneeSearch: {str(e)}")


if __name__ == "__main__":
    test_tools()
