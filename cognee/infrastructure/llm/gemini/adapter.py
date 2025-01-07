from typing import Type
from pydantic import BaseModel
import json
import re
from litellm import acompletion

class GeminiAdapter:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: Type[BaseModel]
    ) -> BaseModel:
        json_system_prompt = f"""{system_prompt}
        IMPORTANT: Respond only with a JSON object in the following format:
        {{
            "summary": "Brief summary of the content",
            "description": "Detailed description of the content",
            "nodes": [
                {{
                    "id": "string",
                    "label": "string"
                }}
            ],
            "edges": [
                {{
                    "source": "string",
                    "target": "string",
                    "relation": "string"
                }}
            ]
        }}"""

        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": text_input}
        ]

        try:
            response = await acompletion(
                model=f"gemini/{self.model}",
                messages=messages,
                api_key=self.api_key,
                max_retries=5
            )

            # Extract the JSON string from the response
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("No content in model response")
                
            json_str = response.choices[0].message.content
            
            # Remove markdown formatting if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Clean and sanitize the JSON string
            # Remove control characters and normalize newlines
            json_str = self._sanitize_json(json_str)

            try:
                # Parse JSON into dict
                data = json.loads(json_str)
                
                # Handle case where response is a list
                if isinstance(data, list):
                    data = data[0]
                
                # Transform the data to match expected format
                transformed_data = {
                    "summary": data.get("summary", "No summary provided"),
                    "description": data.get("description", "No description provided"),
                    "nodes": [
                        {
                            "name": node["id"],
                            "type": node["label"],
                            "description": f"A {node['label'].lower()} in the knowledge graph",
                            "id": node["id"]
                        } for node in data.get("nodes", [])
                    ],
                    "edges": [
                        {
                            "source_node_id": edge["source"],
                            "target_node_id": edge["target"],
                            "relationship_name": edge["relation"]
                        } for edge in data.get("edges", [])
                    ]
                }
                
                # Create instance of response model from transformed data
                return response_model(**transformed_data)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse model response into {response_model.__name__}: {str(e)}\n"
                    f"Response was: {json_str}\n"
                    f"Transformed data was: {transformed_data if 'transformed_data' in locals() else 'Not created'}"
                )
        except Exception as e:
            raise ValueError(f"Failed to extract content from model response: {str(e)}\nFull response was: {response}")

    def _sanitize_json(self, json_str: str) -> str:
        """Sanitize JSON string by removing control characters and normalizing newlines."""
        # Remove control characters except newlines and tabs
        json_str = ''.join(char for char in json_str if char >= ' ' or char in '\n\t')
        
        # Normalize newlines
        json_str = json_str.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove any non-ASCII characters
        json_str = json_str.encode('ascii', 'ignore').decode()
        
        # Remove any Chinese characters (like 对话式)
        json_str = re.sub(r'[\u4e00-\u9fff]+', '', json_str)
        
        # Normalize whitespace
        json_str = re.sub(r'\s+', ' ', json_str)
        
        return json_str