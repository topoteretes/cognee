import litellm
import instructor
from typing import Type, Optional
from pydantic import BaseModel
from litellm.exceptions import ContentPolicyViolationError
from instructor.exceptions import InstructorRetryException

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.files.utils.open_data_file import open_data_file
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
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region_name: str
    aws_profile_name: str

    MAX_RETRIES = 5

    def __init__(
        self,
        model: str,
        api_key: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        aws_region_name: str = "us-east-1",
        aws_profile_name: str = None,
        aws_bedrock_runtime_endpoint: str = None,
        max_tokens: int = 16384,
        streaming: bool = False,
    ):
        self.aclient = instructor.from_litellm(litellm.acompletion)
        self.client = instructor.from_litellm(litellm.completion)
        self.model = model
        self.api_key = api_key
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.aws_bedrock_runtime_endpoint = aws_bedrock_runtime_endpoint
        self.max_tokens = max_tokens
        self.streaming = streaming

    def _create_bedrock_request(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> dict:
        """Create Bedrock request with authentication and enhanced JSON formatting."""
        enhanced_system_prompt = f"""{system_prompt}

IMPORTANT: You must respond with valid JSON only. Do not include any text before or after the JSON. The response must be a valid JSON object that can be parsed directly."""

        request_params = {
            "model": self.model,
            "custom_llm_provider": "bedrock",
            "drop_params": True,
            "messages": [
                {"role": "user", "content": text_input},
                {"role": "system", "content": enhanced_system_prompt},
            ],
            "response_model": response_model,
            "max_retries": self.MAX_RETRIES,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
        }

        # Add authentication parameters
        if self.api_key:
            request_params["api_key"] = self.api_key
        elif self.aws_access_key_id and self.aws_secret_access_key:
            request_params["aws_access_key_id"] = self.aws_access_key_id
            request_params["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                request_params["aws_session_token"] = self.aws_session_token
        elif self.aws_profile_name:
            request_params["aws_profile_name"] = self.aws_profile_name

        # Add optional parameters
        if self.aws_region_name:
            request_params["aws_region_name"] = self.aws_region_name
        if self.aws_bedrock_runtime_endpoint:
            request_params["aws_bedrock_runtime_endpoint"] = self.aws_bedrock_runtime_endpoint

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
            raise InvalidValueError(message="No system prompt path provided.")
        system_prompt = LLMGateway.read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )
        return formatted_prompt
