from typing import Type, Optional
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from openai import AsyncOpenAI  # Use AsyncOpenAI for async compatibility
import base64
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
        api_version: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.api_version = api_version

        # Use AsyncOpenAI for proper async handling
        self.aclient = instructor.from_openai(
            AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key), mode=instructor.Mode.JSON
        )

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a structured output from the LLM using the provided text and system prompt."""

        # Ensure the API method is async
        if not callable(getattr(self.aclient.chat.completions, "create", None)):
            raise TypeError("self.aclient.chat.completions.create is not callable!")

        # Call the API with the expected parameters
        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ],
            max_tokens=self.max_tokens,
        )

        # Ensure the response is valid before passing to Pydantic model
        if not isinstance(response, dict):
            raise ValueError(f"Unexpected response format: {response}")

        # Use instructor's parse_response method instead of passing response_model in API call
        return instructor.parse_response(response, response_model)

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

        # Ensure response is valid before accessing .choices[0].message.content
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("Image transcription failed. No response received.")

        return response.choices[0].message.content
