"""LLM Interface"""

from typing import Type, Protocol
from abc import abstractmethod
from pydantic import BaseModel
from cognee.infrastructure.llm.LLMGateway import LLMGateway


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
