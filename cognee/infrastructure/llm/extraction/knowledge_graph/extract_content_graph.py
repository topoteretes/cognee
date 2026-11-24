import os
from typing import Any

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.llm_graph_model import (
    content_graph_to_data_point,
    datapoint_model_to_basemodel,
)


async def extract_content_graph(
    content: str,
    response_model: type[BaseModel],
    custom_prompt: str | None = None,
    temporal_hints: list[str] | None = None,
    **kwargs: Any,
) -> BaseModel:
    """Extract a graph from ``content`` with one structured-output LLM call.

    ``temporal_hints`` are rendered into the default graph prompt's
    TEMPORAL_NORMALIZATION_HINTS block (see ``engine/utils/temporal_hints.py``).
    A ``custom_prompt`` is sent verbatim and receives no hints.
    ``GRAPH_EXTRACTION_REASONING_EFFORT``, when set, is sent as ``reasoning_effort``
    on this call only; a ``reasoning_effort`` kwarg overrides it.
    """
    llm_config = get_llm_config()
    if llm_config.graph_extraction_reasoning_effort:
        kwargs.setdefault("reasoning_effort", llm_config.graph_extraction_reasoning_effort)

    if custom_prompt:
        system_prompt = custom_prompt
    else:
        prompt_path = llm_config.graph_prompt_path

        # Check if the prompt path is an absolute path or just a filename
        if os.path.isabs(prompt_path):
            # directory containing the file
            base_directory = os.path.dirname(prompt_path)
            # just the filename itself
            prompt_path = os.path.basename(prompt_path)
        else:
            base_directory = None

        system_prompt = render_prompt(
            prompt_path, {"temporal_hints": temporal_hints or []}, base_directory=base_directory
        )

    simplified_response_model = response_model
    if isinstance(response_model, type) and issubclass(response_model, DataPoint):
        simplified_response_model = datapoint_model_to_basemodel(
            response_model, strip_metadata=True
        )

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, simplified_response_model, **kwargs
    )

    if isinstance(response_model, type) and issubclass(response_model, DataPoint):
        return await content_graph_to_data_point(content_graph, response_model)
    return content_graph
