from typing import Type
from pydantic import BaseModel
import instructor
from tenacity import retry, stop_after_attempt
import anthropic
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt


class AnthropicAdapter(LLMInterface):
    """Adapter for Ollama's API"""

    def __init__(self, model: str = None):
        self.aclient = instructor.patch(
            create = anthropic.Anthropic().messages.create,
            mode = instructor.Mode.ANTHROPIC_TOOLS
        )
        self.model = model

    @retry(stop = stop_after_attempt(5))
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
            max_retries = 0,
            messages = [{
                "role": "user",
                "content": f"""Use the given format to extract information
                from the following input: {text_input}. {system_prompt}""",
            }],
            response_model = response_model,
        )

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""

        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise ValueError("No system prompt path provided.")

        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n""" if system_prompt else None

        return formatted_prompt
