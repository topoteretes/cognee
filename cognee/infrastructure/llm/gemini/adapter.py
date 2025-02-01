from typing import Type, Optional
from pydantic import BaseModel
import logging
import litellm
import asyncio
from litellm import acompletion, JSONSchemaValidationError
from cognee.shared.data_models import MonitoringTool
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.base_config import get_base_config

logger = logging.getLogger(__name__)

monitoring = get_base_config().monitoring_tool
if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


class GeminiAdapter(LLMInterface):
    MAX_RETRIES = 5

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.api_version = api_version
        self.streaming = streaming
        self.max_tokens = max_tokens

    @observe(as_type="generation")
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["name", "type", "description", "id", "label"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_node_id": {"type": "string"},
                                "target_node_id": {"type": "string"},
                                "relationship_name": {"type": "string"},
                            },
                            "required": ["source_node_id", "target_node_id", "relationship_name"],
                        },
                    },
                },
                "required": ["summary", "description", "nodes", "edges"],
            }

            simplified_prompt = f"""
{system_prompt}

IMPORTANT: Your response must be a valid JSON object with these required fields:
1. summary: A brief summary
2. description: A detailed description
3. nodes: Array of nodes with name, type, description, id, and label
4. edges: Array of edges with source_node_id, target_node_id, and relationship_name

Example structure:
{{
  "summary": "Brief summary",
  "description": "Detailed description",
  "nodes": [
    {{
      "name": "Example Node",
      "type": "Concept",
      "description": "Node description",
      "id": "example-id",
      "label": "Concept"
    }}
  ],
  "edges": [
    {{
      "source_node_id": "source-id",
      "target_node_id": "target-id",
      "relationship_name": "relates_to"
    }}
  ]
}}"""

            messages = [
                {"role": "system", "content": simplified_prompt},
                {"role": "user", "content": text_input},
            ]

            try:
                response = await acompletion(
                    model=f"{self.model}",
                    messages=messages,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    response_format={"type": "json_object", "schema": response_schema},
                    timeout=10,
                    num_retries=self.MAX_RETRIES,
                )

                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    return response_model.model_validate_json(content)

            except litellm.exceptions.BadRequestError as e:
                logger.error(f"Bad request error: {str(e)}")
                raise ValueError(f"Invalid request: {str(e)}")

            raise ValueError("Failed to get valid response after retries")

        except JSONSchemaValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            logger.debug(f"Raw response: {e.raw_response}")
            raise ValueError(f"Response failed schema validation: {str(e)}")

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )
        return formatted_prompt
