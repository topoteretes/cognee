import litellm
import instructor
from typing import Type
from pydantic import BaseModel
from litellm.exceptions import ContentPolicyViolationError
from instructor.exceptions import InstructorRetryException

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.exceptions import (
    ContentPolicyFilterError,
    MissingSystemPromptPathError,
)
from cognee.infrastructure.files.storage.s3_config import get_s3_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.rate_limiter import (
    rate_limit_async,
    rate_limit_sync,
    sleep_and_retry_async,
    sleep_and_retry_sync,
)
from cognee.modules.observability.get_observe import get_observe

observe = get_observe()


class BedrockAdapter(LLMInterface):
    """
    Adapter for AWS Bedrock API with support for three authentication methods:
    1. API Key (Bearer Token)
    2. AWS Credentials (access key + secret key)
    3. AWS Profile (boto3 credential chain)
    """

    name = "Bedrock"
    model: str
    api_key: str
    default_instructor_mode = "json_schema_mode"

    MAX_RETRIES = 5

    def __init__(
        self,
        model: str,
        api_key: str = None,
        max_completion_tokens: int = 16384,
        streaming: bool = False,
        instructor_mode: str = None,
    ):
        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode(self.instructor_mode)
        )
        self.client = instructor.from_litellm(litellm.completion)
        self.model = model
        self.api_key = api_key
        self.max_completion_tokens = max_completion_tokens
        self.streaming = streaming

    def _create_bedrock_request(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> dict:
        """Create Bedrock request with authentication."""

        request_params = {
            "model": self.model,
            "custom_llm_provider": "bedrock",
            "drop_params": True,
            "messages": [
                {"role": "user", "content": text_input},
                {"role": "system", "content": system_prompt},
            ],
            "response_model": response_model,
            "max_retries": self.MAX_RETRIES,
            "max_completion_tokens": self.max_completion_tokens,
            "stream": self.streaming,
        }

        s3_config = get_s3_config()

        # Add authentication parameters
        if self.api_key:
            request_params["api_key"] = self.api_key
        elif s3_config.aws_access_key_id and s3_config.aws_secret_access_key:
            request_params["aws_access_key_id"] = s3_config.aws_access_key_id
            request_params["aws_secret_access_key"] = s3_config.aws_secret_access_key
            if s3_config.aws_session_token:
                request_params["aws_session_token"] = s3_config.aws_session_token
        elif s3_config.aws_profile_name:
            request_params["aws_profile_name"] = s3_config.aws_profile_name

        if s3_config.aws_region:
            request_params["aws_region_name"] = s3_config.aws_region

        # Add optional parameters
        if s3_config.aws_bedrock_runtime_endpoint:
            request_params["aws_bedrock_runtime_endpoint"] = s3_config.aws_bedrock_runtime_endpoint

        return request_params

    @observe(as_type="generation")
    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate structured output from AWS Bedrock API."""

        try:
            request_params = self._create_bedrock_request(text_input, system_prompt, response_model)
            return await self.aclient.chat.completions.create(**request_params)

        except (
            ContentPolicyViolationError,
            InstructorRetryException,
        ) as error:
            if (
                isinstance(error, InstructorRetryException)
                and "content management policy" not in str(error).lower()
            ):
                raise error

            raise ContentPolicyFilterError(
                f"The provided input contains content that is not aligned with our content policy: {text_input}"
            )

    @observe
    @sleep_and_retry_sync()
    @rate_limit_sync
    def create_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate structured output from AWS Bedrock API (synchronous)."""

        request_params = self._create_bedrock_request(text_input, system_prompt, response_model)
        return self.client.chat.completions.create(**request_params)

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise MissingSystemPromptPathError()
        system_prompt = LLMGateway.read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )
        return formatted_prompt
