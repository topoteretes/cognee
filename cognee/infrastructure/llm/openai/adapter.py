import os
import base64
import litellm
import instructor
from typing import Type
from pydantic import BaseModel

from cognee.modules.data.processing.document_types.open_data_file import open_data_file
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.rate_limiter import (
    rate_limit_async,
    rate_limit_sync,
    sleep_and_retry_async,
    sleep_and_retry_sync,
)
from cognee.modules.observability.get_observe import get_observe

observe = get_observe()


class OpenAIAdapter(LLMInterface):
    name = "OpenAI"
    model: str
    api_key: str
    api_version: str

    MAX_RETRIES = 5

    """Adapter for OpenAI's GPT-3, GPT=4 API"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        model: str,
        transcription_model: str,
        max_tokens: int,
        streaming: bool = False,
    ):
        self.aclient = instructor.from_litellm(litellm.acompletion)
        self.client = instructor.from_litellm(litellm.completion)
        self.transcription_model = transcription_model
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.streaming = streaming

    @observe(as_type="generation")
    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        return await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to
                extract information from the following input: {text_input}. """,
                },
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            response_model=response_model,
            max_retries=self.MAX_RETRIES,
        )

    @observe
    @sleep_and_retry_sync()
    @rate_limit_sync
    def create_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to
                extract information from the following input: {text_input}. """,
                },
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            response_model=response_model,
            max_retries=self.MAX_RETRIES,
        )

    @rate_limit_sync
    def create_transcript(self, input):
        """Generate a audio transcript from a user query."""

        if not input.startswith("s3://") and not os.path.isfile(input):
            raise FileNotFoundError(f"The file {input} does not exist.")

        with open_data_file(input, mode="rb") as audio_file:
            transcription = litellm.transcription(
                model=self.transcription_model,
                file=audio_file,
                api_key=self.api_key,
                api_base=self.endpoint,
                api_version=self.api_version,
                max_retries=self.MAX_RETRIES,
            )

        return transcription

    @rate_limit_sync
    def transcribe_image(self, input) -> BaseModel:
        with open_data_file(input, mode="rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_tokens=300,
            max_retries=self.MAX_RETRIES,
        )

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")
        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )
        return formatted_prompt
