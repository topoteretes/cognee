import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.config import (
    get_llm_config,
    get_llm_context_config,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.graph_model_utils import datapoint_model_to_basemodel


def _note_extraction_config(system_prompt: str) -> None:
    """Record the effective extraction model and prompt fingerprint on the run manifest.

    Eval capture (SDK-529). The model is the one every LLM call in this context is
    routed to — the stage-merged config that ``pipeline_stage("extraction")`` bound,
    or the global config outside a stage — resolved exactly as ``LLMGateway`` resolves
    it. The import is lazy so ``import cognee`` never loads the capture package, and
    the fingerprint (a sha256 over the rendered prompt) is only computed while capture
    is active - and once per distinct prompt, not once per chunk; ``note()`` itself is
    a no-op without a run scope.
    """
    from cognee.modules.observability import capture as eval_capture

    if not eval_capture.is_active():
        return
    eval_capture.note("extraction.model", get_llm_context_config().llm_model)
    eval_capture.note("extraction.prompt_fingerprint", _prompt_fingerprint(system_prompt))


@lru_cache(maxsize=8)
def _prompt_fingerprint(system_prompt: str) -> str:
    """Fingerprint of one rendered extraction prompt, memoized per distinct prompt text.

    ``extract_content_graph`` runs once per chunk and every chunk of a batch renders
    the same prompt, so the sha256 is paid once per prompt rather than once per chunk
    (the economy ``summarize_text`` applies to its prompt). Only reached while capture
    is active; a ``custom_prompt`` gets its own entry.
    """
    from cognee.modules.observability import capture as eval_capture

    return eval_capture.prompt_fingerprint(system_prompt)


async def extract_content_graph(
    content: str, response_model: type[BaseModel], custom_prompt: str | None = None, **kwargs: Any
) -> BaseModel:
    if custom_prompt:
        system_prompt = custom_prompt
    else:
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

        system_prompt = render_prompt(prompt_path, {}, base_directory=base_directory)

    _note_extraction_config(system_prompt)

    simplified_response_model = response_model
    if isinstance(response_model, type) and issubclass(response_model, DataPoint):
        simplified_response_model = datapoint_model_to_basemodel(
            response_model, strip_metadata=True
        )

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, simplified_response_model, **kwargs
    )

    if simplified_response_model is not response_model:
        return response_model.model_validate(content_graph.model_dump())
    return content_graph
