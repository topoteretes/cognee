import os
import base64
from pathlib import Path
from typing import Type

import litellm
import instructor
from pydantic import BaseModel
from cognee.shared.data_models import MonitoringTool
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.base_config import get_base_config

monitoring = get_base_config().monitoring_tool
if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


class OpenAIAdapter(LLMInterface):
    name = "OpenAI"
    model: str
    api_key: str
    api_version: str

    """Adapter for OpenAI's GPT-3, GPT=4 API"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        model: str,
        transcription_model: str,
        streaming: bool = False,
    ):
        self.aclient = instructor.from_litellm(litellm.acompletion)
        self.client = instructor.from_litellm(litellm.completion)
        self.transcription_model = transcription_model
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.streaming = streaming

    @observe(as_type="generation")
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
            max_retries=5,
        )

    @observe
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
            max_retries=5,
        )

    def create_transcript(self, input):
        """Generate a audio transcript from a user query."""

        if not os.path.isfile(input):
            raise FileNotFoundError(f"The file {input} does not exist.")

        # with open(input, 'rb') as audio_file:
        #     audio_data = audio_file.read()

        transcription = litellm.transcription(
            model=self.transcription_model,
            file=Path(input),
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_retries=5,
        )

        return transcription

    def transcribe_image(self, input) -> BaseModel:
        with open(input, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?",
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
            max_retries=5,
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
