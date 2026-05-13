"""Adapter for Instructor-backed Structured Output Framework for Llama CPP"""

import logging
from typing import Any, cast

import instructor
import litellm
from instructor.core.patch import InstructorChatCompletionCreate
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

logger = get_logger()


class LlamaCppAPIAdapter(LLMInterface):
    """
    Adapter for Llama CPP LLM provider with support for TWO modes:

    1. SERVER MODE (OpenAI-compatible):
       - Connects to llama-cpp-python server via HTTP (local or remote)
       - Uses instructor.from_openai()
       - Requires: endpoint, api_key, model

    2. LOCAL MODE (In-process):
       - Loads model directly using llama-cpp-python library
       - Uses instructor.patch() on llama.Llama object
       - Requires: model_path

    Public methods:
    - acreate_structured_output

    Instance variables:
    - name
    - model (for server mode) or model_path (for local mode)
    - mode_type: "server" or "local"
    - max_completion_tokens
    - aclient
    """

    name: str
    model: str | None
    model_path: str | None
    mode_type: str  # "server" or "local"
    default_instructor_mode = instructor.Mode.JSON

    def __init__(
        self,
        name: str = "LlamaCpp",
        max_completion_tokens: int = 2048,
        instructor_mode: str | None = None,
        # Server mode parameters
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        # Local mode parameters
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        chat_format: str = "chatml",
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.max_completion_tokens = max_completion_tokens
        self.llm_args: dict[str, Any] = llm_args or {}
        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        # Determine which mode to use
        if model_path:
            self._init_local_mode(model_path, n_ctx, n_gpu_layers, chat_format)
        elif endpoint:
            self._init_server_mode(endpoint, api_key, model)
        else:
            raise ValueError(
                "Must provide either 'model_path' (for local mode) or 'endpoint' (for server mode)"
            )

    def _init_local_mode(
        self, model_path: str, n_ctx: int, n_gpu_layers: int, chat_format: str
    ) -> None:
        """Initialize local mode using llama-cpp-python library directly"""
        try:
            import llama_cpp  # ty:ignore[unresolved-import]
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
            )

        logger.info(f"Initializing LlamaCpp in LOCAL mode with model: {model_path}")

        self.mode_type = "local"
        self.model_path = model_path
        self.model = None

        # Initialize llama-cpp-python with the model
        self.llama = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 for all GPU, 0 for CPU only
            chat_format=chat_format,
            n_ctx=n_ctx,
            verbose=False,
        )

        self.aclient = instructor.patch(
            create=self.llama.create_chat_completion_openai_v1,
            mode=instructor.Mode(self.instructor_mode),
        )

    def _init_server_mode(self, endpoint: str, api_key: str | None, model: str | None) -> None:
        """Initialize server mode connecting to llama-cpp-python server"""
        logger.info(f"Initializing LlamaCpp in SERVER mode with endpoint: {endpoint}")

        self.mode_type = "server"
        self.model = model
        self.model_path = None
        self.endpoint = endpoint
        self.api_key = api_key

        # Use instructor.from_openai() for server mode (OpenAI-compatible API)
        self.aclient = instructor.from_openai(
            AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key),
            mode=instructor.Mode(self.instructor_mode),
        )

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(
            (litellm.exceptions.NotFoundError, litellm.exceptions.AuthenticationError)
        ),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: type[BaseModel], **kwargs
    ) -> BaseModel:
        """
        Generate a structured output from the LLM using the provided text and system prompt.

        Works in both local and server modes transparently.

        Parameters:
        -----------
            - text_input (str): The input text provided by the user.
            - system_prompt (str): The system prompt that guides the response generation.
            - response_model (Type[BaseModel]): The model type that the response should conform to.

        Returns:
        --------
            - BaseModel: A structured output that conforms to the specified response model.
        """
        async with llm_rate_limiter_context_manager():
            # Prepare messages (system first, then user is more standard)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ]

            merged_kwargs = {**self.llm_args, **kwargs}
            if self.mode_type == "server":
                response = await cast(
                    instructor.AsyncInstructor, self.aclient
                ).chat.completions.create(
                    model=self.model,
                    messages=messages,  # ty:ignore[invalid-argument-type]
                    response_model=response_model,
                    max_retries=2,
                    **merged_kwargs,
                )

            else:
                import asyncio

                def _call_sync():
                    return cast(InstructorChatCompletionCreate, self.aclient)(
                        messages=messages,
                        response_model=response_model,
                        **merged_kwargs,
                    )

                # Run sync function in thread pool to avoid blocking
                response = await asyncio.to_thread(_call_sync)

        return response
