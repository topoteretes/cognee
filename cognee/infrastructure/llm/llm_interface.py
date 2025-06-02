"""LLM Interface"""

from typing import Type, Protocol
from abc import abstractmethod
from pydantic import BaseModel
from cognee.infrastructure.llm.prompts import read_query_prompt


class LLMInterface(Protocol):
    """
    Define an interface for LLM models with methods for structured output and prompt
    display.

    Methods:
    - acreate_structured_output(text_input: str, system_prompt: str, response_model:
    Type[BaseModel])
    - show_prompt(text_input: str, system_prompt: str)
    """

    @abstractmethod
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Obtain structured output from text input using a specified response model.

        This function must be implemented by subclasses to provide the actual functionality for
        generating structured output. Raises NotImplementedError if not implemented.

        Parameters:
        -----------

            - text_input (str): Input text from the user to be processed.
            - system_prompt (str): The system prompt that guides the model's response.
            - response_model (Type[BaseModel]): The model type that will structure the response
              output.
        """
        raise NotImplementedError

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """
        Format and display the prompt for a user query.

        Parameters:
        -----------

            - text_input (str): Input text from the user to be included in the prompt.
            - system_prompt (str): The system prompt that will be shown alongside the user
              input.

        Returns:
        --------

            - str: The formatted prompt string combining system prompt and user input.
        """
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise ValueError("No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""

        return formatted_prompt
