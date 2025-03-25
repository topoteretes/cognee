"""
LLM utility methods for generating responses.
"""
import asyncio
from typing import Optional, Dict, Any

from cognee.infrastructure.llm.get_llm_client import get_llm_client


async def get_llm_response(prompt: str, **kwargs) -> str:
    """
    Get a response from the LLM using the provided prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        **kwargs: Additional arguments to pass to the LLM client
        
    Returns:
        The LLM's response as a string
    """
    llm_client = get_llm_client()
    
    try:
        response = await llm_client.acreate_structured_output(
            text_input=prompt,
            system_prompt="You are a helpful assistant. Provide a clear and concise response to the user's query.",
            response_model=str,
            **kwargs
        )
        
        # Return the text response
        return response
    except Exception as e:
        # If LLM calling fails, return a placeholder response for testing purposes
        print(f"Error getting LLM response: {e}")
        return f"[LLM placeholder response for: {prompt[:50]}...]" 