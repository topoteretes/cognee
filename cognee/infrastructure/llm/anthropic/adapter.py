from typing import Type
from pydantic import BaseModel
import instructor
from tenacity import retry, stop_after_attempt
import anthropic
from cognee.infrastructure.llm.llm_interface import LLMInterface


class AnthropicAdapter(LLMInterface):
    """Adapter for Anthropic API"""
    name = "Anthropic"
    model: str

    def __init__(self, model: str = None):
        self.aclient = instructor.patch(
            create = anthropic.Anthropic().messages.create,
            mode = instructor.Mode.ANTHROPIC_TOOLS
        )
        self.model = model

    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        return await self.aclient(
            model = self.model,
            max_tokens = 4096,
            max_retries = 5,
            messages = [{
                "role": "user",
                "content": f"""Use the given format to extract information
                from the following input: {text_input}. {system_prompt}""",
            }],
            response_model = response_model,
        )
