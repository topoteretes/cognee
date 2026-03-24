from typing import Type, Optional, Coroutine
from pydantic import BaseModel
from cognee.infrastructure.llm import get_llm_config


class LLMGateway:
    """
    Class handles selection of structured output frameworks and LLM functions.
    Class used as a namespace for LLM related functions, should not be instantiated, all methods are static.
    """

    @staticmethod
    def acreate_structured_output(
        text_input: str, system_prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> Coroutine:
        llm_config = get_llm_config()

        from examples.demos.simple_agent_v2.agentic_context_trace import get_current_agent_context_trace
        context_trace = get_current_agent_context_trace()
        if context_trace is not None and context_trace.with_memory:
            memory_context = context_trace.memory_context
            text_input = f"Additional Memory Context: {memory_context} Original input: \n\n{text_input}"

        if llm_config.structured_output_framework.upper() == "BAML":
            from cognee.infrastructure.llm.structured_output_framework.baml.baml_src.extraction import (
                acreate_structured_output,
            )

            return acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
            )
        else:
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
                get_llm_client,
            )

            llm_client = get_llm_client()
            return llm_client.acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
                **kwargs,
            )

    @staticmethod
    def create_transcript(input) -> Coroutine:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.create_transcript(input=input)

    @staticmethod
    def transcribe_image(input) -> Coroutine:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.transcribe_image(input=input)
