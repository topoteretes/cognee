"""Adapter for Azure OpenAI with managed identity and API key support."""

import logging
from typing import Any

import instructor
import litellm
from instructor.core import InstructorRetryException
from litellm.exceptions import ContentPolicyViolationError
from openai import (
    AsyncAzureOpenAI,
    ContentFilterFinishReasonError,
)
from openai import (
    AzureOpenAI as AzureOpenAIClient,
)
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
    OpenAIAdapter,
)
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

logger = get_logger()
observe = get_observe()


class AzureOpenAIAdapter(OpenAIAdapter):
    """Adapter for Azure OpenAI with managed identity and API key authentication.

    When `use_managed_identity=True`, uses DefaultAzureCredential to obtain
    a bearer token for Azure Cognitive Services, bypassing API key auth.
    Falls back to standard API key auth otherwise.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        endpoint: str | None = None,
        api_version: str | None = None,
        transcription_model: str | None = None,
        instructor_mode: str | None = None,
        streaming: bool = False,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_endpoint: str | None = None,
        llm_args: dict[str, Any] | None = None,
        use_managed_identity: bool = False,
    ) -> None:
        if use_managed_identity:
            self._init_managed_identity(
                model=model,
                max_completion_tokens=max_completion_tokens,
                endpoint=endpoint,
                api_version=api_version,
                transcription_model=transcription_model,
                instructor_mode=instructor_mode,
                streaming=streaming,
                fallback_model=fallback_model,
                fallback_api_key=fallback_api_key,
                fallback_endpoint=fallback_endpoint,
                llm_args=llm_args,
            )
        else:
            super().__init__(
                api_key=api_key,
                model=model,
                max_completion_tokens=max_completion_tokens,
                endpoint=endpoint,
                api_version=api_version,
                transcription_model=transcription_model,
                instructor_mode=instructor_mode,
                streaming=streaming,
                fallback_model=fallback_model,
                fallback_api_key=fallback_api_key,
                fallback_endpoint=fallback_endpoint,
                llm_args=llm_args,
            )

        self.use_managed_identity = use_managed_identity

    def _init_managed_identity(
        self,
        model: str,
        max_completion_tokens: int,
        endpoint: str | None,
        api_version: str | None,
        transcription_model: str | None,
        instructor_mode: str | None,
        streaming: bool,
        fallback_model: str | None,
        fallback_api_key: str | None,
        fallback_endpoint: str | None,
        llm_args: dict[str, Any] | None,
    ) -> None:
        """Initialize using Azure managed identity (DefaultAzureCredential)."""
        try:
            from azure.identity import (  # ty:ignore[unresolved-import]
                DefaultAzureCredential,
                get_bearer_token_provider,
            )
        except ImportError:
            raise ImportError(
                "azure-identity is required for managed identity authentication. "
                "Install it with: pip install azure-identity"
            )

        logger.info("Attempting to use Azure managed identity")

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )

        if not endpoint:
            raise ValueError(
                "LLM_ENDPOINT must be set to your Azure OpenAI endpoint "
                "(e.g. https://YOUR-RESOURCE.openai.azure.com) when using Azure provider."
            )

        # Set instance attributes directly (skip parent __init__ to avoid litellm client setup)
        self.name = "AzureOpenAI"
        self.model = model
        self.api_key = "managed-identity"  # placeholder, not used for auth
        self.api_version = api_version or "2024-12-01-preview"
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.transcription_model = transcription_model or model
        self.image_transcribe_model = model
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint
        self.llm_args: dict[str, Any] = llm_args or {}
        self.streaming = streaming

        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        # Create native Azure OpenAI clients with managed identity token provider
        azure_endpoint = endpoint.rstrip("/")
        azure_client = AzureOpenAIClient(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )
        azure_aclient = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )

        self.client = instructor.from_openai(
            azure_client, mode=instructor.Mode(self.instructor_mode)
        )
        self.aclient = instructor.from_openai(
            azure_aclient, mode=instructor.Mode(self.instructor_mode)
        )

        logger.info("Using managed identity for Azure OpenAI")

    @staticmethod
    def _extract_deployment(model: str) -> str:
        """Extract deployment name from model string (e.g. 'azure/gpt-4o' -> 'gpt-4o')."""
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    @observe(as_type="generation")
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
        if not self.use_managed_identity:
            return await super().acreate_structured_output(
                text_input, system_prompt, response_model, **kwargs
            )

        # Managed identity path: use native OpenAI client (not litellm)
        merged_kwargs = {**self.llm_args, **kwargs}
        # Remove litellm-specific kwargs that OpenAI client doesn't understand
        merged_kwargs.pop("api_key", None)
        merged_kwargs.pop("api_base", None)
        merged_kwargs.pop("api_version", None)

        try:
            async with llm_rate_limiter_context_manager():
                return await self.aclient.chat.completions.create(
                    model=self._extract_deployment(self.model),
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": f"""{text_input}""",
                        },
                    ],
                    response_model=response_model,
                    max_retries=self.MAX_RETRIES,
                    **merged_kwargs,
                )
        except (
            ContentFilterFinishReasonError,
            ContentPolicyViolationError,
            InstructorRetryException,
        ) as e:
            if not (self.fallback_model and self.fallback_api_key):
                raise e
            # Fall back to litellm for fallback model
            try:
                fallback_aclient = instructor.from_litellm(litellm.acompletion)
                async with llm_rate_limiter_context_manager():
                    return await fallback_aclient.chat.completions.create(
                        model=self.fallback_model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": f"""{text_input}""",
                            },
                        ],
                        api_key=self.fallback_api_key,
                        response_model=response_model,
                        max_retries=self.MAX_RETRIES,
                        **merged_kwargs,
                    )
            except (
                ContentFilterFinishReasonError,
                ContentPolicyViolationError,
                InstructorRetryException,
            ) as error:
                if (
                    isinstance(error, InstructorRetryException)
                    and "content management policy" not in str(error).lower()
                ):
                    raise error
                else:
                    raise ContentPolicyFilterError(
                        f"The provided input contains content that is not aligned with our content policy: {text_input}"
                    ) from error
