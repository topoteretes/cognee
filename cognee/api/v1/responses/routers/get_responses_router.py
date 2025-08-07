"""
Get router for the OpenAI-compatible responses API.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
import openai
from fastapi import APIRouter, Depends
from cognee.api.v1.responses.models import (
    ResponseRequest,
    ResponseBody,
    ResponseToolCall,
    ChatUsage,
    FunctionCall,
    ToolCallOutput,
)
from cognee.api.v1.responses.dispatch_function import dispatch_function
from cognee.api.v1.responses.default_tools import DEFAULT_TOOLS
from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user


def get_responses_router() -> APIRouter:
    """
    Returns the FastAPI router for OpenAI-compatible responses.

    This implementation follows the new OpenAI Responses API format as described in:
    https://platform.openai.com/docs/api-reference/responses/create
    """

    router = APIRouter()
    logger = logging.getLogger(__name__)

    def _get_model_client():
        """
        Get appropriate client based on model name
        """
        llm_config = get_llm_config()
        return openai.AsyncOpenAI(api_key=llm_config.llm_api_key)

    async def call_openai_api_for_model(
        input_text: str,
        model: str,
        tools: Optional[List[Dict[str, Any]]] = DEFAULT_TOOLS,
        tool_choice: Any = "auto",
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Call appropriate model API based on model name
        """

        # TODO: Support other models (e.g. cognee-v1-openai-gpt-3.5-turbo, etc.)
        model = "gpt-4o"

        client = _get_model_client()

        logger.debug(f"Using model: {model}")

        response = await client.responses.create(
            model=model,
            input=input_text,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        logger.info(f"Response: {response}")
        return response.model_dump()

    @router.post("/", response_model=ResponseBody)
    async def create_response(
        request: ResponseRequest,
        user: User = Depends(get_authenticated_user),
    ) -> ResponseBody:
        """
        OpenAI-compatible responses endpoint with function calling support.

        This endpoint provides OpenAI-compatible API responses with integrated
        function calling capabilities for Cognee operations.

        ## Request Parameters
        - **input** (str): The input text to process
        - **model** (str): The model to use for processing
        - **tools** (Optional[List[Dict]]): Available tools for function calling
        - **tool_choice** (Any): Tool selection strategy (default: "auto")
        - **temperature** (float): Response randomness (default: 1.0)

        ## Response
        Returns an OpenAI-compatible response body with function call results.

        ## Error Codes
        - **400 Bad Request**: Invalid request parameters
        - **500 Internal Server Error**: Error processing request

        ## Notes
        - Compatible with OpenAI API format
        - Supports function calling with Cognee tools
        - Uses default tools if none provided
        """
        # Use default tools if none provided
        tools = request.tools or DEFAULT_TOOLS

        # Call the API
        response = await call_openai_api_for_model(
            input_text=request.input,
            model=request.model,
            tools=tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
        )

        # Use the response ID from the API or generate a new one
        response_id = response.get("id", f"resp_{uuid.uuid4().hex}")

        # Check if there are function tool calls in the output
        output = response.get("output", [])

        processed_tool_calls = []

        # Process any function tool calls from the output
        for item in output:
            if isinstance(item, dict) and item.get("type") == "function_call":
                # This is a tool call from the new format
                function_name = item.get("name", "")
                arguments_str = item.get("arguments", "{}")
                call_id = item.get("call_id", f"call_{uuid.uuid4().hex}")

                # Create a format the dispatcher can handle
                tool_call = {
                    "id": call_id,
                    "function": {"name": function_name, "arguments": arguments_str},
                    "type": "function",
                }

                # Dispatch the function
                try:
                    function_result = await dispatch_function(tool_call)
                    output_status = "success"
                except Exception as e:
                    logger.exception(f"Error executing function {function_name}: {e}")
                    function_result = f"Error executing {function_name}: {str(e)}"
                    output_status = "error"

                processed_call = ResponseToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name=function_name, arguments=arguments_str),
                    output=ToolCallOutput(status=output_status, data={"result": function_result}),
                )

                processed_tool_calls.append(processed_call)

        # Get usage data from the response if available
        usage = response.get("usage", {})

        # Create the response object with all processed tool calls
        response_obj = ResponseBody(
            id=response_id,
            model=request.model,
            tool_calls=processed_tool_calls,
            usage=ChatUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

        return response_obj

    return router
