"""LLM Interface"""

from typing import Type, Protocol
from abc import abstractmethod
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import read_query_prompt


class LLMInterface(Protocol):
    """LLM Interface"""

    @abstractmethod
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """To get structured output, import/call this function"""
        raise NotImplementedError

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise ValueError("No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""

        return formatted_prompt
