from typing import Type, Optional, Coroutine
from pydantic import BaseModel
from cognee.infrastructure.llm import get_llm_config


class LLMGateway:
    """
    Class handles selection of structured output frameworks and LLM functions.
    Class used as a namespace for LLM related functions, should not be instantiated, all methods are static.
    """

    @staticmethod
    def render_prompt(filename: str, context: dict, base_directory: str = None):
        from cognee.infrastructure.llm.prompts import render_prompt

        return render_prompt(filename=filename, context=context, base_directory=base_directory)

    @staticmethod
    def acreate_structured_output(
        text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> Coroutine:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.acreate_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=response_model
        )

    @staticmethod
    def create_structured_output(
        text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.create_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=response_model
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

    @staticmethod
    def show_prompt(text_input: str, system_prompt: str) -> str:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.show_prompt(text_input=text_input, system_prompt=system_prompt)

    @staticmethod
    def read_query_prompt(prompt_file_name: str, base_directory: str = None):
        from cognee.infrastructure.llm.prompts import (
            read_query_prompt,
        )

        return read_query_prompt(prompt_file_name=prompt_file_name, base_directory=base_directory)

    @staticmethod
    def extract_content_graph(
        content: str,
        response_model: Type[BaseModel],
        mode: str = "simple",
        custom_prompt: Optional[str] = None,
    ) -> Coroutine:
        llm_config = get_llm_config()
        if llm_config.structured_output_framework.upper() == "BAML":
            from cognee.infrastructure.llm.structured_output_framework.baml.baml_src.extraction import (
                extract_content_graph,
            )

            return extract_content_graph(
                content=content,
                response_model=response_model,
                mode=mode,
                custom_prompt=custom_prompt,
            )
        else:
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.extraction import (
                extract_content_graph,
            )

            return extract_content_graph(
                content=content, response_model=response_model, custom_prompt=custom_prompt
            )

    @staticmethod
    def extract_categories(content: str, response_model: Type[BaseModel]) -> Coroutine:
        # TODO: Add BAML version of category and extraction and update function
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.extraction import (
            extract_categories,
        )

        return extract_categories(content=content, response_model=response_model)

    @staticmethod
    def extract_code_summary(content: str) -> Coroutine:
        llm_config = get_llm_config()
        if llm_config.structured_output_framework.upper() == "BAML":
            from cognee.infrastructure.llm.structured_output_framework.baml.baml_src.extraction import (
                extract_code_summary,
            )

            return extract_code_summary(content=content)
        else:
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.extraction import (
                extract_code_summary,
            )

            return extract_code_summary(content=content)

    @staticmethod
    def extract_summary(content: str, response_model: Type[BaseModel]) -> Coroutine:
        llm_config = get_llm_config()
        if llm_config.structured_output_framework.upper() == "BAML":
            from cognee.infrastructure.llm.structured_output_framework.baml.baml_src.extraction import (
                extract_summary,
            )

            return extract_summary(content=content, response_model=response_model)
        else:
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.extraction import (
                extract_summary,
            )

            return extract_summary(content=content, response_model=response_model)
