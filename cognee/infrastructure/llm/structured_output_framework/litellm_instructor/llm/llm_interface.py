"""LLM Interface"""

from typing import Type, Protocol
from abc import abstractmethod
from pydantic import BaseModel
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)


class LLMInterface(Protocol):
    """
    Define an interface for LLM models with methods for structured output, multimodal processing, and prompt display.

    Methods:
    - acreate_structured_output(text_input: str, system_prompt: str, response_model: Type[BaseModel])
    - create_transcript(input): Transcribe audio files to text
    - transcribe_image(input): Analyze image files and return text description
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

    @abstractmethod
    async def create_transcript(self, input) -> TranscriptionReturnType:
        """
        Transcribe audio content to text.

        This method should be implemented by subclasses that support audio transcription.
        If not implemented, returns None and should be handled gracefully by callers.

        Parameters:
        -----------
            - input: The path to the audio file that needs to be transcribed.

        Returns:
        --------
            - BaseModel: A structured output containing the transcription, or None if not supported.
        """
        raise NotImplementedError

    @abstractmethod
    async def transcribe_image(self, input) -> BaseModel:
        """
        Analyze image content and return text description.

        This method should be implemented by subclasses that support image analysis.
        If not implemented, returns None and should be handled gracefully by callers.

        Parameters:
        -----------
            - input: The path to the image file that needs to be analyzed.

        Returns:
        --------
            - BaseModel: A structured output containing the image description, or None if not supported.
        """
        raise NotImplementedError
