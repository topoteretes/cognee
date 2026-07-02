import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Type
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel

from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import (
    AudioDocument,
    DltRowDocument,
    ImageDocument,
    TextDocument,
)
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.session_lifecycle.usage_tracking import estimate_cost_usd
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import datapoint_model_to_basemodel
from cognee.tasks.documents import classify_documents


SUMMARY_OUTPUT_TOKENS_PER_CHUNK = 150
GRAPH_OUTPUT_TOKEN_RATIO = 0.5
MIN_GRAPH_OUTPUT_TOKENS_PER_CHUNK = 256
TEXT_EXTENSIONS = {
    "txt",
    "md",
    "markdown",
    "csv",
    "json",
    "jsonl",
    "xml",
    "html",
    "htm",
}


@dataclass
class DryRunStageEstimate:
    name: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.cost_usd,
        }


@dataclass
class DryRunEstimate:
    operation: str
    model: str
    chunks: int
    stages: list[DryRunStageEstimate]
    chunk_tokens: int
    skipped_items: int = 0
    warnings: list[str] | None = None

    @property
    def input_tokens(self) -> int:
        return sum(stage.input_tokens for stage in self.stages)

    @property
    def output_tokens(self) -> int:
        return sum(stage.output_tokens for stage in self.stages)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return sum(stage.cost_usd for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "dry_run": True,
            "model": self.model,
            "chunks": self.chunks,
            "chunk_tokens": self.chunk_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "skipped_items": self.skipped_items,
            "warnings": self.warnings or [],
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def __str__(self) -> str:
        return format_dry_run_estimate(self)


def _model_name() -> str:
    return get_llm_config().llm_model


def _llm_tokenizer():
    model = _model_name().split("/", 1)[-1]
    try:
        return TikTokenTokenizer(model=model)
    except Exception:
        return TikTokenTokenizer(model=None)


def _count_tokens(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    tokenizer = tokenizer or _llm_tokenizer()
    try:
        return tokenizer.count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _schema_tokens(model: Type[BaseModel], tokenizer) -> int:
    if isinstance(model, type) and issubclass(model, BaseModel):
        schema = model.model_json_schema()
        return _count_tokens(json.dumps(schema, sort_keys=True), tokenizer)
    return 0


def _graph_prompt(custom_prompt: Optional[str]) -> str:
    if custom_prompt:
        return custom_prompt

    llm_config = get_llm_config()
    prompt_path = llm_config.graph_prompt_path
    if os.path.isabs(prompt_path):
        base_directory = os.path.dirname(prompt_path)
        prompt_path = os.path.basename(prompt_path)
    else:
        base_directory = None
    return render_prompt(prompt_path, {}, base_directory=base_directory)


def _simplify_graph_model(graph_model: Type[BaseModel]) -> Type[BaseModel]:
    from cognee.infrastructure.engine import DataPoint

    if isinstance(graph_model, type) and issubclass(graph_model, DataPoint):
        return datapoint_model_to_basemodel(graph_model, strip_metadata=True)
    return graph_model


def _chunk_texts(
    texts: Iterable[str],
    *,
    chunker=TextChunker,
    chunk_size: int,
) -> list[Any]:
    chunkers: list[Any] = []
    for index, text in enumerate(texts):
        if not text:
            continue

        async def get_text(text=text):
            yield text

        document = TextDocument(
            id=uuid5(NAMESPACE_OID, f"dry-run-{index}"),
            name=f"dry-run-{index}",
            raw_data_location="",
            external_metadata="{}",
            importance_weight=0.5,
        )
        try:
            chunker_instance = chunker(document, max_chunk_size=chunk_size, get_text=get_text)
        except TypeError:
            chunker_instance = chunker(document, get_text=get_text, max_chunk_tokens=chunk_size)
        chunkers.append(chunker_instance)

    return chunkers


async def _chunks_from_texts(
    texts: Iterable[str],
    *,
    chunker=TextChunker,
    chunk_size: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for chunker_instance in _chunk_texts(texts, chunker=chunker, chunk_size=chunk_size):
        async for chunk in chunker_instance.read():
            chunks.append(chunk)
    return chunks


async def _chunks_from_data_items(
    data_items: list[Data],
    *,
    chunker=TextChunker,
    chunk_size: int,
) -> tuple[list[DocumentChunk], int]:
    documents = await classify_documents(data_items)
    chunks: list[DocumentChunk] = []
    skipped = 0

    for document in documents:
        if isinstance(document, (AudioDocument, ImageDocument)):
            skipped += 1
            continue
        async for chunk in document.read(max_chunk_size=chunk_size, chunker_cls=chunker):
            chunks.append(chunk)

    return chunks, skipped


def estimate_chunks(
    chunks: list[DocumentChunk],
    *,
    operation: str,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
    skipped_items: int = 0,
) -> DryRunEstimate:
    tokenizer = _llm_tokenizer()
    model = _model_name()
    graph_model = _simplify_graph_model(graph_model)
    summarization_model = get_cognify_config().summarization_model

    llm_chunks = [
        chunk
        for chunk in chunks
        if not isinstance(getattr(chunk, "is_part_of", None), DltRowDocument)
    ]
    skipped_dlt_chunks = len(chunks) - len(llm_chunks)

    chunk_token_counts = [_count_tokens(chunk.text, tokenizer) for chunk in llm_chunks]
    chunk_tokens = sum(chunk_token_counts)

    graph_overhead = _count_tokens(_graph_prompt(custom_prompt), tokenizer) + _schema_tokens(
        graph_model, tokenizer
    )
    summary_prompt = read_query_prompt("summarize_content.txt") or ""
    summary_overhead = _count_tokens(summary_prompt, tokenizer) + _schema_tokens(
        summarization_model, tokenizer
    )

    graph_input = sum(tokens + graph_overhead for tokens in chunk_token_counts)
    graph_output = sum(
        max(MIN_GRAPH_OUTPUT_TOKENS_PER_CHUNK, int(tokens * GRAPH_OUTPUT_TOKEN_RATIO))
        for tokens in chunk_token_counts
    )
    summary_input = sum(tokens + summary_overhead for tokens in chunk_token_counts)
    summary_output = len(llm_chunks) * SUMMARY_OUTPUT_TOKENS_PER_CHUNK

    stages = [
        DryRunStageEstimate(
            name="structured_graph_extraction",
            calls=len(llm_chunks),
            input_tokens=graph_input,
            output_tokens=graph_output,
            cost_usd=estimate_cost_usd(model, graph_input, graph_output),
        ),
        DryRunStageEstimate(
            name="chunk_summarization",
            calls=len(llm_chunks),
            input_tokens=summary_input,
            output_tokens=summary_output,
            cost_usd=estimate_cost_usd(model, summary_input, summary_output),
        ),
    ]

    warnings = []
    if skipped_items:
        warnings.append(
            f"Skipped {skipped_items} audio/image item(s) because estimating them would "
            "require LLM transcription."
        )

    if skipped_dlt_chunks:
        warnings.append(
            f"Skipped {skipped_dlt_chunks} DLT row chunk(s) because they do not use LLM "
            "extraction or summarization."
        )

    return DryRunEstimate(
        operation=operation,
        model=model,
        chunks=len(llm_chunks),
        stages=stages,
        chunk_tokens=chunk_tokens,
        skipped_items=skipped_items + skipped_dlt_chunks,
        warnings=warnings,
    )


async def _read_file_like_text(value: Any) -> str:
    position = None
    name = getattr(value, "filename", None) or getattr(value, "name", None)
    if name:
        try:
            suffix = Path(str(name)).suffix.lower().lstrip(".")
        except ValueError:
            suffix = ""
        if suffix and suffix not in TEXT_EXTENSIONS:
            raise ValueError(f"dry_run supports text-like file inputs only, got {name!r}.")

    if hasattr(value, "tell") and hasattr(value, "seek"):
        try:
            position = value.tell()
            seek_result = value.seek(0)
            if inspect.isawaitable(seek_result):
                await seek_result
        except Exception:
            position = None
    payload = value.read()
    if inspect.isawaitable(payload):
        payload = await payload
    if position is not None:
        try:
            seek_result = value.seek(position)
            if inspect.isawaitable(seek_result):
                await seek_result
        except Exception:
            pass
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def _read_supported_text_path(value: str) -> str | None:
    if "\n" in value or "\r" in value or len(value) > 4096:
        return None
    try:
        path = Path(value)
        is_file = path.exists() and path.is_file()
    except (OSError, ValueError):
        return None
    if not is_file:
        return None

    suffix = path.suffix.lower().lstrip(".")
    if suffix not in TEXT_EXTENSIONS:
        raise ValueError(f"dry_run supports text-like file inputs only, got {value!r}.")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


async def _input_to_texts(data: Any) -> list[str]:
    if data is None:
        return []
    if isinstance(data, list):
        texts: list[str] = []
        for item in data:
            texts.extend(await _input_to_texts(item))
        return texts
    if isinstance(data, bytes):
        return [data.decode("utf-8", errors="replace")]
    if isinstance(data, (str, Path)):
        value = str(data)
        file_text = _read_supported_text_path(value)
        return [file_text if file_text is not None else value]
    if hasattr(data, "read"):
        return [await _read_file_like_text(data)]
    return [str(data)]


async def estimate_remember_dry_run(
    data: Any,
    *,
    chunker=TextChunker,
    chunk_size: int,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
) -> DryRunEstimate:
    chunks = await _chunks_from_texts(
        await _input_to_texts(data),
        chunker=chunker,
        chunk_size=chunk_size,
    )
    return estimate_chunks(
        chunks,
        operation="remember",
        graph_model=graph_model,
        custom_prompt=custom_prompt,
    )


async def estimate_cognify_dry_run(
    datasets,
    *,
    user=None,
    chunker=TextChunker,
    chunk_size: int,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
) -> DryRunEstimate:
    user, authorized_datasets = await resolve_authorized_user_datasets(datasets, user)

    chunks: list[DocumentChunk] = []
    skipped = 0
    for dataset in authorized_datasets:
        dataset_data = await get_dataset_data(dataset.id)
        dataset_chunks, skipped_items = await _chunks_from_data_items(
            dataset_data,
            chunker=chunker,
            chunk_size=chunk_size,
        )
        chunks.extend(dataset_chunks)
        skipped += skipped_items

    return estimate_chunks(
        chunks,
        operation="cognify",
        graph_model=graph_model,
        custom_prompt=custom_prompt,
        skipped_items=skipped,
    )


def format_dry_run_estimate(estimate: DryRunEstimate | dict[str, Any]) -> str:
    if isinstance(estimate, dict):
        operation = estimate.get("operation", "unknown")
        model = estimate.get("model", "unknown")
        chunks = int(estimate.get("chunks") or 0)
        input_tokens = int(estimate.get("input_tokens") or 0)
        output_tokens = int(estimate.get("output_tokens") or 0)
        total_tokens = int(estimate.get("total_tokens") or input_tokens + output_tokens)
        estimated_cost_usd = float(estimate.get("estimated_cost_usd") or 0.0)
        stages = estimate.get("stages") or []
        warnings = estimate.get("warnings") or []
    else:
        operation = estimate.operation
        model = estimate.model
        chunks = estimate.chunks
        input_tokens = estimate.input_tokens
        output_tokens = estimate.output_tokens
        total_tokens = estimate.total_tokens
        estimated_cost_usd = estimate.estimated_cost_usd
        stages = [stage.to_dict() for stage in estimate.stages]
        warnings = estimate.warnings or []

    lines = [
        "Dry-run token estimate",
        f"  Operation: {operation}",
        f"  Model: {model}",
        f"  Chunks: {chunks}",
        f"  Input tokens: {input_tokens:,}",
        f"  Output tokens: {output_tokens:,}",
        f"  Total tokens: {total_tokens:,}",
        f"  Estimated cost: ${estimated_cost_usd:.6f}",
        "",
        "  Stage                         Calls   Input tokens   Output tokens   Cost",
    ]
    for stage in stages:
        name = str(stage.get("name", "unknown"))
        calls = int(stage.get("calls") or 0)
        stage_input = int(stage.get("input_tokens") or 0)
        stage_output = int(stage.get("output_tokens") or 0)
        stage_cost = float(stage.get("estimated_cost_usd") or stage.get("cost_usd") or 0.0)
        lines.append(
            f"  {name:<29} {calls:>5}   {stage_input:>12,}   "
            f"{stage_output:>13,}   ${stage_cost:.6f}"
        )
    for warning in warnings:
        lines.append(f"  Warning: {warning}")
    return "\n".join(lines)
