import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, create_model

from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt


def _get_graph_system_prompt(custom_prompt: str | None = None) -> str:
    if custom_prompt:
        return custom_prompt

    llm_config = get_llm_config()
    prompt_path = llm_config.graph_prompt_path

    # Check if the prompt path is an absolute path or just a filename
    if os.path.isabs(prompt_path):
        # directory containing the file
        base_directory = os.path.dirname(prompt_path)
        # just the filename itself
        prompt_path = os.path.basename(prompt_path)
    else:
        base_directory = None

    return render_prompt(prompt_path, {}, base_directory=base_directory)


def _get_graph_summary_system_prompt(custom_prompt: str | None = None) -> str:
    summary_prompt = read_query_prompt("summarize_content.txt") or ""
    return (
        f"{_get_graph_system_prompt(custom_prompt)}\n\n"
        "Also produce a concise summary of the same input content.\n"
        f"{summary_prompt}"
    )


@lru_cache(maxsize=128)
def get_graph_summary_response_model(
    graph_model: type[BaseModel], summarization_model: type[BaseModel]
) -> type[BaseModel]:
    graph_model_name = getattr(graph_model, "__name__", "Graph")
    summarization_model_name = getattr(summarization_model, "__name__", "Summary")

    return create_model(
        f"{graph_model_name}{summarization_model_name}Extraction",
        graph=(
            graph_model,
            Field(..., description="Knowledge graph extracted from the input content."),
        ),
        summary=(
            summarization_model,
            Field(..., description="Concise summary of the input content."),
        ),
    )


async def extract_content_graph(
    content: str, response_model: type[BaseModel], custom_prompt: str | None = None, **kwargs: Any
) -> BaseModel:
    system_prompt = _get_graph_system_prompt(custom_prompt)

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model, **kwargs
    )

    return content_graph


async def extract_content_graph_and_summary(
    content: str,
    graph_model: type[BaseModel],
    summarization_model: type[BaseModel],
    custom_prompt: str | None = None,
    **kwargs: Any,
) -> BaseModel:
    response_model = get_graph_summary_response_model(graph_model, summarization_model)
    system_prompt = _get_graph_summary_system_prompt(custom_prompt)

    return await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model, **kwargs
    )
