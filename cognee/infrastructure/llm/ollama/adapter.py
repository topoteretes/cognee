from typing import Type
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.rate_limiter import (
    rate_limit_async,
    rate_limit_sync,
    sleep_and_retry_async,
)
from openai import OpenAI
import base64
import os


class OllamaAPIAdapter(LLMInterface):
    """Adapter for a Generic API LLM provider using instructor with an OpenAI backend."""

    def __init__(self, endpoint: str, api_key: str, model: str, name: str, max_tokens: int):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens

        self.aclient = instructor.from_openai(
            OpenAI(base_url=self.endpoint, api_key=self.api_key), mode=instructor.Mode.JSON
        )

    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a structured output from the LLM using the provided text and system prompt."""

        response = self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Use the given format to extract information from the following input: {text_input}",
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

    @rate_limit_sync
    def create_transcript(self, input_file: str) -> str:
        """Generate an audio transcript from a user query."""

        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")

        with open(input_file, "rb") as audio_file:
            transcription = self.aclient.audio.transcriptions.create(
                model="whisper-1",  # Ensure the correct model for transcription
                file=audio_file,
                language="en",
            )

        # Ensure the response contains a valid transcript
        if not hasattr(transcription, "text"):
            raise ValueError("Transcription failed. No text returned.")

        return transcription.text

    @rate_limit_sync
    def transcribe_image(self, input_file: str) -> str:
        """Transcribe content from an image using base64 encoding."""

        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")

        with open(input_file, "rb") as image_file:
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
            max_tokens=300,
        )

        # Ensure response is valid before accessing .choices[0].message.content
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("Image transcription failed. No response received.")

        return response.choices[0].message.content
