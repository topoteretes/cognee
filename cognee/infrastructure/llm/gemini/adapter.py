from typing import Type, Optional
from pydantic import BaseModel
import json
import logging
from litellm import acompletion, JSONSchemaValidationError
import litellm
from litellm import acompletion, JSONSchemaValidationError

logger = logging.getLogger(__name__)

class GeminiAdapter:
    MAX_TOKENS = 8192
    
    def __init__(self, 
        api_key: str, 
        model: str,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        streaming: bool = False
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.api_version = api_version
        self.streaming = streaming

    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: Type[BaseModel]
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
                                "label": {"type": "string"}
                            },
                            "required": ["name", "type", "description", "id", "label"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_node_id": {"type": "string"},
                                "target_node_id": {"type": "string"},
                                "relationship_name": {"type": "string"}
                            },
                            "required": ["source_node_id", "target_node_id", "relationship_name"]
                        }
                    }
                },
                "required": ["summary", "description", "nodes", "edges"]
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
                {"role": "user", "content": text_input}
            ]

            for attempt in range(3):
                try:
                    response = await acompletion(
                        model=f"gemini/{self.model}",
                        messages=messages,
                        api_key=self.api_key,
                        max_tokens=self.MAX_TOKENS,
                        temperature=0.1,
                        response_format={
                            "type": "json_object",
                            "schema": response_schema
                        }
                    )
                    
                    if response.choices and response.choices[0].message.content:
                        content = response.choices[0].message.content
                        return response_model.model_validate_json(content)
                    
                except (litellm.exceptions.OpenAIError, litellm.exceptions.BadRequestError) as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    continue

            raise ValueError("Failed to get valid response after retries")

        except JSONSchemaValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            logger.debug(f"Raw response: {e.raw_response}")
            raise ValueError(f"Response failed schema validation: {str(e)}")
        except litellm.exceptions.OpenAIError as e:
            logger.error(f"API error in structured output generation: {str(e)}")
            raise ValueError(f"Failed to generate structured output: {str(e)}")
        except Exception as e:
            logger.error(f"Error in structured output generation: {str(e)}")
            raise ValueError(f"Failed to generate structured output: {str(e)}")