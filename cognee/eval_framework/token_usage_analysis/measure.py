"""Measure the real token cost of ingesting a chunk with cognee.

This is the only module that calls the LLM. Each sampled chunk is run through
cognee's summary and graph-extraction calls, and the real prompt/completion
token usage is read off each response (so instruction and schema overhead are
included). Multiple llm_models are run sequentially, switching cognee's config
between them.
"""

from __future__ import annotations

import asyncio
import os

import litellm
import tiktoken
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.extraction import extract_content_graph, extract_summary
from cognee.shared.data_models import KnowledgeGraph, SummarizedContent

from cost_model import ChunkMeasurement

FALLBACK_ENCODING = "o200k_base"


def infer_provider(llm_model: str) -> str:
    """Map an llm_model name to its provider (only openai/anthropic are used)."""
    leaf = llm_model.split("/")[-1].lower()
    if llm_model.lower().startswith("anthropic/") or leaf.startswith("claude"):
        return "anthropic"
    return "openai"


def count_tokens(text: str, llm_model: str) -> int:
    """Local token count for the chunk content and the corpus (no API call)."""
    if infer_provider(llm_model) == "anthropic":
        return int(litellm.token_counter(model=llm_model, text=text))
    return len(_openai_encoding(llm_model).encode(text))


def run_measurements(chunks: list[str], llm_models: list[str]) -> list[ChunkMeasurement]:
    """Measure every chunk under every llm_model. Seals the asyncio entrypoint."""
    return asyncio.run(_measure_all(chunks, llm_models))


async def measure_chunk(chunk: str, llm_model: str) -> ChunkMeasurement:
    """Run cognee's two ingestion calls on one chunk and record real token usage."""
    summary, graph = await asyncio.gather(
        extract_summary(chunk, SummarizedContent),
        extract_content_graph(chunk, KnowledgeGraph),
    )
    summary_prompt, summary_completion = _usage(summary)
    graph_prompt, graph_completion = _usage(graph)
    return ChunkMeasurement(
        llm_model=llm_model,
        input_tokens=count_tokens(chunk, llm_model),
        summary_prompt_tokens=summary_prompt,
        summary_completion_tokens=summary_completion,
        graph_prompt_tokens=graph_prompt,
        graph_completion_tokens=graph_completion,
    )


async def _measure_all(chunks: list[str], llm_models: list[str]) -> list[ChunkMeasurement]:
    measurements: list[ChunkMeasurement] = []
    for llm_model in llm_models:
        _configure_llm_model(llm_model)
        measurements += await asyncio.gather(*(measure_chunk(chunk, llm_model) for chunk in chunks))
    return measurements


def _configure_llm_model(llm_model: str) -> None:
    """Switch cognee onto llm_model: set env, then clear the cached config."""
    provider = infer_provider(llm_model)
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = llm_model
    api_key = os.environ.get(f"{provider.upper()}_API_KEY") or os.environ.get("LLM_API_KEY")
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
    get_llm_config.cache_clear()


def _usage(result) -> tuple[int, int]:
    """Read (prompt, completion) tokens from an instructor result's raw response.

    litellm names them prompt_tokens/completion_tokens; the Anthropic client names
    them input_tokens/output_tokens.
    """
    usage = result._raw_response.usage
    prompt = _pick(usage, "prompt_tokens", "input_tokens")
    completion = _pick(usage, "completion_tokens", "output_tokens")
    return prompt, completion


def _pick(usage, *field_names: str) -> int:
    """Return the first present usage field (as int), across provider naming."""
    for field_name in field_names:
        value = getattr(usage, field_name, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"usage exposes none of {field_names}: {usage!r}")


def _openai_encoding(llm_model: str) -> tiktoken.Encoding:
    model_name = llm_model.split("/")[-1]
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)
