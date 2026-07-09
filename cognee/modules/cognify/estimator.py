"""Dry-run token and cost estimation for the default cognify pipeline.

Estimates the two LLM-heavy stages — structured graph extraction and chunk
summarization — without making LLM calls, ingesting data, or writing graph
results. Reuses the real pipeline pieces (document classifiers, chunker,
prompt templates, graph-model simplification) so chunk and call counts track
what a real run would do.

Known approximations:
- A real ``cognify`` run with ``incremental_loading=True`` skips documents that
  were already processed, so estimates for re-runs are upper bounds.
- LLM calls outside the two estimated stages (``remember`` self-improvement,
  transcription of audio/image items) and embedding costs are not included.
- Input tokens use the real tokenizer, prompt templates, and response-model
  JSON schema; output tokens use fixed heuristics.
"""

import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Type
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import NAMESPACE_OID, UUID, uuid5

from pydantic import BaseModel

from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import (
    AudioDocument,
    DltRowDocument,
    ImageDocument,
    PdfDocument,
    TextDocument,
    UnstructuredDocument,
)
from cognee.modules.session_lifecycle.usage_tracking import estimate_cost_usd
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import datapoint_model_to_basemodel
from cognee.tasks.documents import classify_documents
from cognee.tasks.documents.classify_documents import EXTENSION_TO_DOCUMENT_CLASS
from cognee.tasks.ingestion.data_item import DataItem

SUMMARY_OUTPUT_TOKENS_PER_CHUNK = 150
GRAPH_OUTPUT_TOKEN_RATIO = 0.5
MIN_GRAPH_OUTPUT_TOKENS_PER_CHUNK = 256

# Formats whose bytes are not directly tokenizable text: images and audio would
# need LLM transcription, PDF/Office formats a loader pass over stored files.
_BINARY_DOCUMENT_TYPES = (AudioDocument, ImageDocument, PdfDocument, UnstructuredDocument)

# Remote inputs a real run would fetch and ingest; estimating the URL string
# instead would be silently, badly wrong.
_UNSUPPORTED_SCHEMES = ("http", "https", "s3")


@dataclass
class DryRunStageEstimate:
    """Token/cost estimate for one LLM stage of the pipeline."""

    name: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class DryRunEstimate:
    """Aggregated estimate returned by ``remember``/``cognify`` with ``dry_run=True``."""

    operation: str
    model: str
    chunks: int
    chunk_tokens: int
    stages: list[DryRunStageEstimate]
    skipped_items: int = 0
    warnings: list[str] = field(default_factory=list)

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
            "warnings": list(self.warnings),
            "stages": [
                {
                    "name": stage.name,
                    "calls": stage.calls,
                    "input_tokens": stage.input_tokens,
                    "output_tokens": stage.output_tokens,
                    "total_tokens": stage.total_tokens,
                    "estimated_cost_usd": stage.cost_usd,
                }
                for stage in self.stages
            ],
        }

    def __str__(self) -> str:
        lines = [
            "Dry-run token estimate",
            f"  Operation: {self.operation}",
            f"  Model: {self.model}",
            f"  Chunks: {self.chunks}",
            f"  Input tokens: {self.input_tokens:,}",
            f"  Output tokens: {self.output_tokens:,}",
            f"  Total tokens: {self.total_tokens:,}",
            f"  Estimated cost: ${self.estimated_cost_usd:.6f}",
            "",
            "  Stage                         Calls   Input tokens   Output tokens   Cost",
        ]
        for stage in self.stages:
            lines.append(
                f"  {stage.name:<29} {stage.calls:>5}   {stage.input_tokens:>12,}   "
                f"{stage.output_tokens:>13,}   ${stage.cost_usd:.6f}"
            )
        lines.extend(f"  Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _llm_tokenizer() -> TikTokenTokenizer:
    model = get_llm_config().llm_model.split("/", 1)[-1]
    try:
        return TikTokenTokenizer(model=model)
    except Exception:
        # Model unknown to tiktoken — fall back to its default encoding.
        return TikTokenTokenizer(model=None)


def _count_tokens(text: str, tokenizer: TikTokenTokenizer) -> int:
    return tokenizer.count_tokens(text) if text else 0


def _schema_tokens(model: Type[BaseModel], tokenizer: TikTokenTokenizer) -> int:
    if isinstance(model, type) and issubclass(model, BaseModel):
        return _count_tokens(json.dumps(model.model_json_schema(), sort_keys=True), tokenizer)
    return 0


def _graph_prompt(custom_prompt: Optional[str]) -> str:
    """The graph-extraction system prompt — mirrors ``extract_content_graph``."""
    if custom_prompt:
        return custom_prompt

    prompt_path = get_llm_config().graph_prompt_path
    base_directory = None
    if os.path.isabs(prompt_path):
        base_directory = os.path.dirname(prompt_path)
        prompt_path = os.path.basename(prompt_path)
    return render_prompt(prompt_path, {}, base_directory=base_directory)


def _simplify_graph_model(graph_model: Type[BaseModel]) -> Type[BaseModel]:
    """DataPoint models are sent simplified — mirrors ``extract_content_graph``."""
    from cognee.infrastructure.engine import DataPoint

    if isinstance(graph_model, type) and issubclass(graph_model, DataPoint):
        return datapoint_model_to_basemodel(graph_model, strip_metadata=True)
    return graph_model


def estimate_chunks(
    chunks: list[DocumentChunk],
    *,
    operation: str,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
    skipped_items: int = 0,
) -> DryRunEstimate:
    """Estimate the per-chunk LLM stages of the default cognify pipeline."""
    tokenizer = _llm_tokenizer()
    model = get_llm_config().llm_model
    summarization_model = get_cognify_config().summarization_model

    # The real pipeline skips LLM extraction and summarization for DLT row
    # chunks (see extract_graph_from_data / summarize_text).
    llm_chunks = [
        chunk
        for chunk in chunks
        if not isinstance(getattr(chunk, "is_part_of", None), DltRowDocument)
    ]
    skipped_dlt_chunks = len(chunks) - len(llm_chunks)

    chunk_token_counts = [_count_tokens(chunk.text, tokenizer) for chunk in llm_chunks]
    chunk_tokens = sum(chunk_token_counts)

    graph_overhead = _count_tokens(_graph_prompt(custom_prompt), tokenizer) + _schema_tokens(
        _simplify_graph_model(graph_model), tokenizer
    )
    summary_overhead = _count_tokens(
        read_query_prompt("summarize_content.txt") or "", tokenizer
    ) + _schema_tokens(summarization_model, tokenizer)

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
            f"Skipped {skipped_items} audio/image item(s) because estimating them "
            "would require LLM transcription."
        )
    if skipped_dlt_chunks:
        warnings.append(
            f"Skipped {skipped_dlt_chunks} DLT row chunk(s) because they do not use "
            "LLM extraction or summarization."
        )
    total_tokens = sum(stage.total_tokens for stage in stages)
    if total_tokens and not sum(stage.cost_usd for stage in stages):
        warnings.append(f"Cost unavailable: no pricing entry for model {model!r}.")

    return DryRunEstimate(
        operation=operation,
        model=model,
        chunks=len(llm_chunks),
        chunk_tokens=chunk_tokens,
        stages=stages,
        skipped_items=skipped_items + skipped_dlt_chunks,
        warnings=warnings,
    )


def _require_text_like(name: str) -> None:
    """Reject formats the estimator cannot read as raw text.

    Mirrors ``classify_documents``: unknown extensions are treated as text.
    """
    try:
        suffix = Path(name).suffix.lower().lstrip(".")
    except ValueError:
        return
    if not suffix:
        return
    document_class = EXTENSION_TO_DOCUMENT_CLASS.get(suffix, TextDocument)
    if issubclass(document_class, _BINARY_DOCUMENT_TYPES):
        raise ValueError(f"dry_run supports text-like file inputs only, got {name!r}.")


def _accept_local_file_path() -> bool:
    """The same ACCEPT_LOCAL_FILE_PATH gate real ingestion applies."""
    from cognee.tasks.ingestion.save_data_item_to_storage import settings

    return settings.accept_local_file_path


def _path_candidate(value: str) -> Optional[Path]:
    """The local path this string refers to, or None when it is raw text.

    Mirrors ``save_data_item_to_storage``: remote URLs and missing absolute
    paths are loud errors, ``file://`` URIs resolve to local paths, everything
    else that is not an existing file is raw text. Scheme is checked before the
    newline guard so a trailing-newline URL cannot slip through as text. When
    ``ACCEPT_LOCAL_FILE_PATH`` is disabled, path references are rejected and
    relative strings are raw text, exactly as in a real run.
    """
    scheme = urlparse(value).scheme.lower()
    if scheme in _UNSUPPORTED_SCHEMES:
        raise ValueError(
            f"dry_run supports local text inputs only, got {value!r}. "
            "A real run would fetch and ingest this URL."
        )
    if scheme == "file":
        if not _accept_local_file_path():
            raise ValueError(f"Local files are not accepted, got {value!r}.")
        return Path(url2pathname(urlparse(value).path))

    if "\n" in value or "\r" in value or len(value) > 4096:
        return None

    if not _accept_local_file_path():
        if value.startswith("/"):
            raise ValueError(f"Local files are not accepted, got {value!r}.")
        return None

    try:
        path = Path(value)
        if path.exists():
            return path
    except (OSError, ValueError):
        return None
    if value.startswith("/"):
        # A real run treats absolute paths as file references and would fail too.
        raise ValueError(f"dry_run file input does not exist: {value!r}.")
    return None


def _read_text_path(path: Path, original: str) -> str:
    if path.is_dir():
        raise ValueError(
            f"dry_run does not support directory inputs, got {original!r}. "
            "Pass individual file paths or raw text."
        )
    if not path.is_file():
        raise ValueError(f"dry_run file input does not exist: {original!r}.")
    _require_text_like(str(path))
    return path.read_text(encoding="utf-8", errors="replace")


async def _read_file_like(value: Any) -> str:
    """Read a file-like object without consuming it for a later real run."""
    name = getattr(value, "filename", None) or getattr(value, "name", None)
    if name:
        _require_text_like(str(name))

    position = None
    if hasattr(value, "tell") and hasattr(value, "seek"):
        try:
            position = value.tell()
            value.seek(0)
        except (OSError, ValueError):
            position = None
    payload = value.read()
    if inspect.isawaitable(payload):
        payload = await payload
    if position is not None:
        value.seek(position)
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


async def _input_to_texts(data: Any) -> list[str]:
    """Resolve ``remember()`` input into raw texts, mirroring ingestion routing."""
    if data is None:
        return []
    if isinstance(data, DataItem):
        # save_data_item_to_storage recurses into the wrapped payload.
        return await _input_to_texts(data.data)
    if isinstance(data, list):
        texts: list[str] = []
        for item in data:
            texts.extend(await _input_to_texts(item))
        return texts
    if isinstance(data, bytes):
        return [data.decode("utf-8", errors="replace")]
    if isinstance(data, (str, Path)):
        value = str(data)
        path = _path_candidate(value)
        return [_read_text_path(path, value) if path is not None else value]
    if hasattr(data, "read"):
        return [await _read_file_like(data)]
    return [str(data)]


async def _chunks_from_texts(
    texts: Iterable[str],
    *,
    chunker: Type[Any],
    chunk_size: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
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
        chunker_instance = chunker(document, get_text=get_text, max_chunk_size=chunk_size)
        async for chunk in chunker_instance.read():
            chunks.append(chunk)
    return chunks


async def _chunks_from_data_items(
    data_items: list[Data],
    *,
    chunker: Type[Any],
    chunk_size: int,
) -> tuple[list[DocumentChunk], int]:
    documents = await classify_documents(data_items)
    chunks: list[DocumentChunk] = []
    skipped = 0
    for document in documents:
        if isinstance(document, (AudioDocument, ImageDocument)):
            skipped += 1
            continue
        async for chunk in document.read(chunker_cls=chunker, max_chunk_size=chunk_size):
            chunks.append(chunk)
    return chunks, skipped


async def estimate_remember_dry_run(
    data: Any,
    *,
    chunker: Type[Any] = TextChunker,
    chunk_size: int,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
) -> DryRunEstimate:
    """Estimate ``remember(data)`` for permanent add+cognify inputs."""
    chunks = await _chunks_from_texts(
        await _input_to_texts(data), chunker=chunker, chunk_size=chunk_size
    )
    return estimate_chunks(
        chunks, operation="remember", graph_model=graph_model, custom_prompt=custom_prompt
    )


async def estimate_cognify_dry_run(
    datasets,
    *,
    user=None,
    chunker: Type[Any] = TextChunker,
    chunk_size: int,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
) -> DryRunEstimate:
    """Estimate ``cognify(datasets)`` over all data in the authorized datasets.

    Resolves datasets read-only: unlike a real run it never creates a missing
    dataset, so estimating a typo'd name fails loudly instead of writing one.
    """
    if user is None:
        user = await get_default_user()
    if isinstance(datasets, (str, UUID)):
        datasets = [datasets]

    authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)
    if not authorized_datasets:
        raise DatasetNotFoundError("There are no datasets to estimate.")

    chunks: list[DocumentChunk] = []
    skipped = 0
    for dataset in authorized_datasets:
        dataset_chunks, skipped_items = await _chunks_from_data_items(
            await get_dataset_data(dataset.id), chunker=chunker, chunk_size=chunk_size
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
