from typing import Type
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.config import get_llm_config
from openai import OpenAI
import base64
from pathlib import Path
import os

class OllamaAPIAdapter(LLMInterface):
    """Adapter for a Ollama API LLM provider using instructor with an OpenAI backend."""

    api_version: str

    MAX_RETRIES = 5

    def __init__(self, endpoint: str, api_key: str, model: str, name: str, max_tokens: int, api_version: str = None) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.api_version= api_version

        self.aclient = instructor.from_openai(
            OpenAI(base_url=self.endpoint, api_key=self.api_key), mode=instructor.Mode.JSON
        )

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


    def create_transcript(self, input):
        """Generate a audio transcript from a user query."""

        if not os.path.isfile(input):
            raise FileNotFoundError(f"The file {input} does not exist.")

        # with open(input, 'rb') as audio_file:
        #     audio_data = audio_file.read()

        transcription = self.aclient.transcription(
            model=self.transcription_model,
            file=Path(input),
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_retries=self.MAX_RETRIES,
        )

        return transcription

    def transcribe_image(self, input) -> BaseModel:
        with open(input, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return self.aclient.completion(
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
            max_retries=self.MAX_RETRIES,
        )