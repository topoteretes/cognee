"""Adapter for Instructor-backed Structured Output Framework for Llama CPP"""

import litellm
import logging
import instructor
from typing import Type, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

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
    model: Optional[str]
    model_path: Optional[str]
    mode_type: str  # "server" or "local"
    default_instructor_mode = instructor.Mode.JSON

    def __init__(
        self,
        name: str = "LlamaCpp",
        max_completion_tokens: int = 2048,
        instructor_mode: Optional[str] = None,
        # Server mode parameters
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        # Local mode parameters
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        chat_format: str = "chatml",
    ):
        self.name = name
        self.max_completion_tokens = max_completion_tokens
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

    def _init_local_mode(self, model_path: str, n_ctx: int, n_gpu_layers: int, chat_format: str):
        """Initialize local mode using llama-cpp-python library directly"""
        try:
            import llama_cpp
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

    def _init_server_mode(self, endpoint: str, api_key: Optional[str], model: Optional[str]):
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
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel], **kwargs
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

            if self.mode_type == "server":
                # Server mode: use async client with OpenAI-compatible API
                response = await self.aclient.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=response_model,
                    max_retries=2,
                    max_completion_tokens=self.max_completion_tokens,
                    **kwargs,
                )

            else:
                import asyncio

                # Local mode: instructor.patch() returns a SYNC callable
                # Per docs: https://python.useinstructor.com/integrations/llama-cpp-python/
                def _call_sync():
                    return self.aclient(
                        messages=messages,
                        response_model=response_model,
                        max_tokens=self.max_completion_tokens,
                        **kwargs,
                    )

                # Run sync function in thread pool to avoid blocking
                response = await asyncio.to_thread(_call_sync)

        return response
