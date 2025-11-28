import base64
import litellm
import logging
import instructor
from typing import Type
from openai import OpenAI
from pydantic import BaseModel
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

logger = get_logger()
observe = get_observe()


class OllamaAPIAdapter(GenericAPIAdapter):
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

    default_instructor_mode = "json_mode"

    def __init__(
        self,
        api_key: str,
        model: str,
        name: str,
        max_completion_tokens: int,
        endpoint: str,
        instructor_mode: str = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            max_completion_tokens=max_completion_tokens,
            name="Ollama",
            endpoint=endpoint,
        )

        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.from_openai(
            OpenAI(base_url=self.endpoint, api_key=self.api_key),
            mode=instructor.Mode(self.instructor_mode),
        )

    @observe(as_type="generation")
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
