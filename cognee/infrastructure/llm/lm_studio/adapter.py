"""Adapter for LM Studio API

LM Studio provides an OpenAI-compatible API that can be used with the OpenAI client.
This adapter implements the LLMInterface for LM Studio, supporting both structured outputs
and streaming capabilities.

For more information about LM Studio API, see:
https://lmstudio.ai/docs/api/
"""

from typing import Type, List
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import logging
import json
from json import JSONDecodeError

from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.rate_limiter import (
    rate_limit_async,
    rate_limit_sync,
    sleep_and_retry_async,
    sleep_and_retry_sync,
)
import base64
import os

logger = logging.getLogger(__name__)

class LMStudioAdapter(LLMInterface):
    """Adapter for LM Studio API using instructor with an OpenAI-compatible backend.

    LM Studio provides an OpenAI-compatible API that supports the following parameters:
    - model: The model to use
    - messages: The conversation history
    - temperature: Controls randomness (0-1)
    - max_tokens: Maximum number of tokens to generate
    - top_p: Controls diversity via nucleus sampling
    - top_k: Controls diversity via top-k sampling
    - stream: Whether to stream the response
    - stop: Sequences where the API will stop generating
    - presence_penalty: Penalizes repeated tokens
    - frequency_penalty: Penalizes frequent tokens
    - logit_bias: Modifies likelihood of specific tokens
    - repeat_penalty: Penalizes repetition (specific to LM Studio)
    - seed: Random seed for reproducibility
    """

    name = "LM Studio"
    model: str

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        max_tokens: int,
        streaming: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
    ):
        """Initialize the LM Studio adapter.

        Args:
            endpoint: The LM Studio API endpoint (e.g., http://localhost:1234/v1)
            api_key: The API key (can be any string for LM Studio, often "lm-studio")
            model: The model identifier
            max_tokens: Maximum number of tokens to generate
            streaming: Whether to enable streaming responses
            temperature: Controls randomness (0-1)
            top_p: Controls diversity via nucleus sampling (0-1)
            top_k: Controls diversity via top-k sampling
            presence_penalty: Penalizes repeated tokens (-2.0 to 2.0)
            frequency_penalty: Penalizes frequent tokens (-2.0 to 2.0)
            repeat_penalty: Penalizes repetition (specific to LM Studio)
        """
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repeat_penalty = repeat_penalty

        # Initialize OpenAI client with LM Studio endpoint
        self.client = OpenAI(base_url=self.endpoint, api_key=self.api_key)

        logger.info(f"Initialized LM Studio adapter with model: {model}, endpoint: {endpoint}")

    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel] | type
    ) -> BaseModel | str:
        """Generate a structured output from LM Studio using the provided text and system prompt.

        Note: Although this method is async, it internally uses the synchronous OpenAI client
        because LM Studio's API may not fully support async operations.

        Args:
            text_input: The input text to process
            system_prompt: The system prompt to guide the model
            response_model: The Pydantic model to structure the output, or str for plain text responses

        Returns:
            A structured response according to the provided response_model

        Raises:
            JSONDecodeError: If the response cannot be parsed as JSON
            ValidationError: If the response doesn't match the expected schema
            Exception: For other API errors
        """
        try:
            # Check if response_model is a string (simple text response) or a Pydantic model
            if response_model is str or isinstance(response_model, type) and issubclass(response_model, str):
                # For string responses, we don't need a JSON schema
                response_format = None
                logger.debug("Using plain text response format (no JSON schema)")
            else:
                # Get the JSON schema from the Pydantic model
                schema = response_model.model_json_schema()

                # Format the response_format parameter according to LM Studio's expectations
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema
                    }
                }
                logger.debug(f"Using JSON schema for {response_model.__name__}: {json.dumps(response_format)}")

            # Make the API call directly with the OpenAI client instead of using instructor
            # Use the synchronous version since LM Studio may not support async properly
            # Prepare the API call parameters
            params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Use the given format to extract information from the following input: {text_input}",
                    },
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                # Exclude LM Studio specific parameters that aren't supported by OpenAI client
                # top_k, repeat_penalty are excluded
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }

            # Only add response_format if it's not None
            if response_format is not None:
                params["response_format"] = response_format

            # Make the API call
            response = self.client.chat.completions.create(**params)

            # Get the response content
            content = response.choices[0].message.content
            logger.debug(f"Raw response from LM Studio: {content}")

            # Handle different response types
            if response_model is str or isinstance(response_model, type) and issubclass(response_model, str):
                # For string responses, just return the content
                return content
            else:
                try:
                    # Parse the JSON response
                    parsed_content = json.loads(content)
                    # Validate against the model
                    result = response_model.model_validate(parsed_content)
                    return result
                except JSONDecodeError as e:
                    logger.error(f"JSON decode error in LM Studio response: {str(e)}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Failed to parse JSON response: {str(e)}")
                except ValidationError as e:
                    logger.error(f"Schema validation failed: {str(e)}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Response failed schema validation: {str(e)}")

        except Exception as e:
            logger.error(f"Error in LM Studio API call: {str(e)}")
            raise

    @sleep_and_retry_sync()
    @rate_limit_sync
    def create_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel] | type
    ) -> BaseModel | str:
        """Generate a synchronous structured output from LM Studio.

        This method formats the Pydantic model into a JSON schema that LM Studio can understand,
        then parses and validates the response.

        Args:
            text_input: The input text to process
            system_prompt: The system prompt to guide the model
            response_model: The Pydantic model to structure the output, or str for plain text responses

        Returns:
            A structured response according to the provided response_model

        Raises:
            JSONDecodeError: If the response cannot be parsed as JSON
            ValidationError: If the response doesn't match the expected schema
            Exception: For other API errors
        """
        try:
            # Check if response_model is a string (simple text response) or a Pydantic model
            if response_model is str or isinstance(response_model, type) and issubclass(response_model, str):
                # For string responses, we don't need a JSON schema
                response_format = None
                logger.debug("Using plain text response format (no JSON schema)")
            else:
                # Get the JSON schema from the Pydantic model
                schema = response_model.model_json_schema()

                # Format the response_format parameter according to LM Studio's expectations
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema
                    }
                }
                logger.debug(f"Using JSON schema for {response_model.__name__}: {json.dumps(response_format)}")

            # Make the API call directly with the OpenAI client
            # Prepare the API call parameters
            params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Use the given format to extract information from the following input: {text_input}",
                    },
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                # Exclude LM Studio specific parameters that aren't supported by OpenAI client
                # top_k, repeat_penalty are excluded
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }

            # Only add response_format if it's not None
            if response_format is not None:
                params["response_format"] = response_format

            # Make the API call
            response = self.client.chat.completions.create(**params)

            # Get the response content
            content = response.choices[0].message.content
            logger.debug(f"Raw response from LM Studio: {content}")

            # Handle different response types
            if response_model is str or isinstance(response_model, type) and issubclass(response_model, str):
                # For string responses, just return the content
                return content
            else:
                try:
                    # Parse the JSON response
                    parsed_content = json.loads(content)
                    # Validate against the model
                    result = response_model.model_validate(parsed_content)
                    return result
                except JSONDecodeError as e:
                    logger.error(f"JSON decode error in LM Studio response: {str(e)}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Failed to parse JSON response: {str(e)}")
                except ValidationError as e:
                    logger.error(f"Schema validation failed: {str(e)}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Response failed schema validation: {str(e)}")

        except Exception as e:
            logger.error(f"Error in LM Studio API call: {str(e)}")
            raise

    @rate_limit_sync
    def create_transcript(self, input_file: str) -> str:
        """Generate an audio transcript from a user query.

        Note: LM Studio may not support audio transcription natively.
        This implementation attempts to use the OpenAI-compatible Whisper endpoint
        if available, but may not work with all LM Studio configurations.

        Args:
            input_file: Path to the audio file

        Returns:
            Transcribed text from the audio file

        Raises:
            FileNotFoundError: If the input file does not exist
            NotImplementedError: If LM Studio does not support audio transcription
            Exception: For other API errors
        """
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")

        try:
            with open(input_file, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",  # Use appropriate model
                    file=audio_file,
                )

            return transcription.text
        except Exception as e:
            logger.error(f"Error in LM Studio audio transcription: {str(e)}")
            raise NotImplementedError(
                "Audio transcription may not be supported by your LM Studio configuration. "
                f"Original error: {str(e)}"
            )

    @rate_limit_sync
    def transcribe_image(self, input_file: str, prompt: str = "What's in this image?") -> str:
        """Transcribe content from an image using base64 encoding.

        This method uses the multimodal capabilities of LM Studio to analyze images.
        It requires a model that supports vision capabilities.

        Args:
            input_file: Path to the image file
            prompt: The prompt to guide the image analysis (default: "What's in this image?")

        Returns:
            Text description of the image content

        Raises:
            FileNotFoundError: If the input file does not exist
            Exception: If the LM Studio API returns an error or the model doesn't support vision
        """
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")

        try:
            # Determine image format from file extension
            file_ext = os.path.splitext(input_file)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif file_ext == '.png':
                mime_type = 'image/png'
            elif file_ext == '.webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'  # Default to JPEG

            with open(input_file, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LM Studio image transcription: {str(e)}")
            if "multimodal" in str(e).lower() or "vision" in str(e).lower():
                raise NotImplementedError(
                    "The selected model does not support vision capabilities. "
                    "Please use a multimodal model that supports image analysis."
                )
            raise

    def get_available_models(self) -> list:
        """Get a list of available models from the LM Studio API.

        Returns:
            List of available model information dictionaries

        Raises:
            Exception: If the LM Studio API returns an error
        """
        try:
            response = self.client.models.list()
            return response.data
        except Exception as e:
            logger.error(f"Error getting available models from LM Studio: {str(e)}")
            raise

    def check_connection(self) -> bool:
        """Check if the connection to LM Studio API is working.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"LM Studio connection check failed: {str(e)}")
            return False

    def create_embeddings(self, text: str) -> list:
        """Create embeddings for the given text using LM Studio's embeddings API.

        Args:
            text: The text to create embeddings for

        Returns:
            List of embedding values

        Raises:
            Exception: If the LM Studio API returns an error or embeddings are not supported
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embeddings with LM Studio: {str(e)}")
            if "embedding" in str(e).lower():
                raise NotImplementedError(
                    "The selected model does not support embeddings. "
                    "Please use a model that supports embeddings or specify an embedding model."
                )
            raise

