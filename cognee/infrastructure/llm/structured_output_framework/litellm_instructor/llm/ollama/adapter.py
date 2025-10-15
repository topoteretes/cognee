import base64
import litellm
import logging
import instructor
from typing import Type
from openai import OpenAI
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.shared.logging_utils import get_logger
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

logger = get_logger()


class OllamaAPIAdapter(LLMInterface):
    """
    Adapter for a Generic API LLM provider using instructor with an OpenAI backend.

    Public methods:

    - acreate_structured_output
    - create_transcript
    - transcribe_image

    Instance variables:

    - name
    - model
    - api_key
    - endpoint
    - max_completion_tokens
    - aclient
    """

    def __init__(
        self, endpoint: str, api_key: str, model: str, name: str, max_completion_tokens: int
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens

        self.aclient = instructor.from_openai(
            OpenAI(base_url=self.endpoint, api_key=self.api_key), mode=instructor.Mode.JSON
        )

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Generate a structured output from the LLM using the provided text and system prompt.

        This asynchronous method sends a request to the API with the user's input and the system
        prompt, and returns a structured response based on the specified model.

        Parameters:
        -----------

            - text_input (str): The input text provided by the user.
            - system_prompt (str): The system prompt that guides the response generation.
            - response_model (Type[BaseModel]): The model type that the response should conform
              to.

        Returns:
        --------

            - BaseModel: A structured output that conforms to the specified response model.
        """

        response = self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"{text_input}",
                },
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
            max_retries=5,
            response_model=response_model,
        )

        return response

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def create_transcript(self, input_file: str) -> str:
        """
        Generate an audio transcript from a user query.

        This synchronous method takes an input audio file and returns its transcription. Raises
        a FileNotFoundError if the input file does not exist, and raises a ValueError if
        transcription fails or returns no text.

        Parameters:
        -----------

            - input_file (str): The path to the audio file to be transcribed.

        Returns:
        --------

            - str: The transcription of the audio as a string.
        """

        async with open_data_file(input_file, mode="rb") as audio_file:
            transcription = self.aclient.audio.transcriptions.create(
                model="whisper-1",  # Ensure the correct model for transcription
                file=audio_file,
                language="en",
            )

        # Ensure the response contains a valid transcript
        if not hasattr(transcription, "text"):
            raise ValueError("Transcription failed. No text returned.")

        return transcription.text

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def transcribe_image(self, input_file: str) -> str:
        """
        Transcribe content from an image using base64 encoding.

        This synchronous method takes an input image file, encodes it as base64, and returns the
        transcription of its content. Raises a FileNotFoundError if the input file does not
        exist, and raises a ValueError if the transcription fails or no valid response is
        received.

        Parameters:
        -----------

            - input_file (str): The path to the image file to be transcribed.

        Returns:
        --------

            - str: The transcription of the image's content as a string.
        """

        async with open_data_file(input_file, mode="rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=300,
        )

        # Ensure response is valid before accessing .choices[0].message.content
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("Image transcription failed. No response received.")

        return response.choices[0].message.content
