from typing import Type
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from openai import OpenAI
import base64
from pathlib import Path
import os


class OllamaAPIAdapter(LLMInterface):
    """Adapter for an Ollama API LLM provider using instructor with an OpenAI backend."""

    MAX_RETRIES = 5

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        name: str,
        max_tokens: int,
        api_version: str = None,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.api_version = api_version

        self.aclient = instructor.from_openai(
            OpenAI(base_url=self.endpoint, api_key=self.api_key), mode=instructor.Mode.JSON
        )

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a structured output from the LLM using the provided text and system prompt."""

        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ],
            max_tokens=self.max_tokens,
        )

        return response_model.parse_raw(response.choices[0].message.content)

    def create_transcript(self, input: str):
        """Generate an audio transcript from a user query."""

        if not os.path.isfile(input):
            raise FileNotFoundError(f"The file {input} does not exist.")

        with open(input, "rb") as audio_file:
            transcription = self.aclient.audio.transcriptions.create(
                model="whisper-1",  # Ensure the correct model for transcription
                file=audio_file,
                language="en",
            )

        return transcription.text

    def transcribe_image(self, input: str) -> str:
        """Transcribe content from an image using base64 encoding."""

        if not os.path.isfile(input):
            raise FileNotFoundError(f"The file {input} does not exist.")

        with open(input, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content
