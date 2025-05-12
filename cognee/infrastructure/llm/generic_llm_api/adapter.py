"""Adapter for Generic API LLM provider API"""

import litellm
import instructor
from typing import Type
from pydantic import BaseModel
from openai import ContentFilterFinishReasonError
from litellm.exceptions import ContentPolicyViolationError
from instructor.exceptions import InstructorRetryException

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.rate_limiter import rate_limit_async, sleep_and_retry_async


class GenericAPIAdapter(LLMInterface):
    """Adapter for Generic API LLM provider API"""

    name: str
    model: str
    api_key: str

    def __init__(
        self,
        endpoint,
        api_key: str,
        model: str,
        name: str,
        max_tokens: int,
        fallback_model: str = None,
        fallback_api_key: str = None,
        fallback_endpoint: str = None,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens

        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint

        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode.JSON, api_key=api_key
        )

    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        try:
            return await self.aclient.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Use the given format to
                    extract information from the following input: {text_input}. """,
                    },
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                ],
                max_retries=5,
                api_base=self.endpoint,
                response_model=response_model,
            )
        except (
            ContentFilterFinishReasonError,
            ContentPolicyViolationError,
            InstructorRetryException,
        ) as error:
            if (
                isinstance(error, InstructorRetryException)
                and not "content management policy" in str(error).lower()
            ):
                raise error

            if not (self.fallback_model and self.fallback_api_key and self.fallback_endpoint):
                raise ContentPolicyFilterError(
                    f"The provided input contains content that is not aligned with our content policy: {text_input}"
                )

            try:
                return await self.aclient.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Use the given format to
                        extract information from the following input: {text_input}. """,
                        },
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                    ],
                    max_retries=5,
                    api_key=self.fallback_api_key,
                    api_base=self.fallback_endpoint,
                    response_model=response_model,
                )
            except (
                ContentFilterFinishReasonError,
                ContentPolicyViolationError,
                InstructorRetryException,
            ) as error:
                if (
                    isinstance(error, InstructorRetryException)
                    and not "content management policy" in str(error).lower()
                ):
                    raise error
                else:
                    raise ContentPolicyFilterError(
                        f"The provided input contains content that is not aligned with our content policy: {text_input}"
                    )
