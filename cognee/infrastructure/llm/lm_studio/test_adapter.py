"""Test script for the LM Studio adapter.

This script tests the structured output functionality of the LM Studio adapter.
Before running, make sure:
1. LM Studio is installed and running
2. The API server is started in LM Studio
3. A model is loaded in LM Studio
4. Your .env file is configured with LM Studio settings:
   LLM_PROVIDER=lm_studio
   LLM_ENDPOINT=http://localhost:1234/v1
   LLM_API_KEY=lm-studio
   LLM_MODEL=<your-model-name>

Usage:
    python -m cognee.infrastructure.llm.lm_studio.test_adapter
"""

import asyncio
import logging
from pydantic import BaseModel, Field
from typing import List

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm import get_llm_config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimpleResponse(BaseModel):
    """A simple response model for testing."""
    message: str = Field(description="A simple message")


class NestedItem(BaseModel):
    """A nested item for testing complex schemas."""
    name: str = Field(description="The name of the item")
    value: int = Field(description="The value of the item")


class ComplexResponse(BaseModel):
    """A more complex response model for testing."""
    title: str = Field(description="The title of the response")
    items: List[NestedItem] = Field(description="A list of nested items")
    summary: str = Field(description="A summary of the response")


async def test_simple_schema():
    """Test the adapter with a simple schema."""
    # Get the LLM client from the configuration
    llm_config = get_llm_config()

    # Log the configuration being used
    logger.info(f"Using LLM provider: {llm_config.llm_provider}")
    logger.info(f"Using LLM model: {llm_config.llm_model}")
    logger.info(f"Using LLM endpoint: {llm_config.llm_endpoint}")

    # Get the LLM client (should be LMStudioAdapter if configured correctly)
    adapter = get_llm_client()

    # Check connection
    if not adapter.check_connection():
        logger.error("Could not connect to LM Studio API. Make sure it's running.")
        return

    logger.info("Connected to LM Studio API")

    # Test with simple schema
    try:
        system_prompt = """
        You are a helpful assistant. Respond with a simple message.
        """

        user_input = "Hello, how are you today?"

        logger.info("Generating simple response...")
        result = await adapter.acreate_structured_output(
            user_input, system_prompt, SimpleResponse
        )

        logger.info(f"Simple response: {result.message}")
        return True
    except Exception as e:
        logger.error(f"Error generating simple response: {e}")
        return False


async def test_complex_schema():
    """Test the adapter with a more complex schema."""
    # Get the LLM client from the configuration
    llm_config = get_llm_config()

    # Get the LLM client (should be LMStudioAdapter if configured correctly)
    adapter = get_llm_client()

    # Check connection
    if not adapter.check_connection():
        logger.error("Could not connect to LM Studio API. Make sure it's running.")
        return

    logger.info("Connected to LM Studio API")

    # Test with complex schema
    try:
        system_prompt = """
        You are a data analyst. Create a report with a title, list of items with names and values,
        and a summary of the data.
        """

        user_input = """
        Here is some sample data:
        - Apple: 10
        - Banana: 5
        - Orange: 8
        """

        logger.info("Generating complex response...")
        result = await adapter.acreate_structured_output(
            user_input, system_prompt, ComplexResponse
        )

        logger.info(f"Complex response title: {result.title}")
        logger.info(f"Complex response items: {[item.model_dump() for item in result.items]}")
        logger.info(f"Complex response summary: {result.summary}")
        return True
    except Exception as e:
        logger.error(f"Error generating complex response: {e}")
        return False


async def main():
    """Run the test script."""
    # Check if LM Studio is configured as the provider
    llm_config = get_llm_config()
    if llm_config.llm_provider != "lm_studio":
        logger.error(f"LLM provider is set to '{llm_config.llm_provider}', not 'lm_studio'")
        logger.error("Please set LLM_PROVIDER=lm_studio in your .env file")
        return

    logger.info("Starting LM Studio adapter tests...")
    logger.info(f"Using configuration: provider={llm_config.llm_provider}, model={llm_config.llm_model}, endpoint={llm_config.llm_endpoint}")

    # Test with simple schema first
    logger.info("Running simple schema test...")
    simple_success = await test_simple_schema()

    if simple_success:
        # If simple schema works, test with complex schema
        logger.info("Simple schema test succeeded, running complex schema test...")
        complex_success = await test_complex_schema()

        if complex_success:
            logger.info("All tests passed successfully!")
        else:
            logger.warning("Complex schema test failed, but simple schema test passed.")
    else:
        logger.error("Simple schema test failed. Please check the logs for details.")


if __name__ == "__main__":
    asyncio.run(main())
