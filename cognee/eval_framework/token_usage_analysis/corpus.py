"""Turn an input (file, directory, or raw string) into sampled chunks.

No LLM calls and no cost math here — just reading text and running cognee's
TextChunker so the sampled chunks match what ingestion would actually see.
"""

from __future__ import annotations

import asyncio
import importlib
import random
from pathlib import Path
from types import SimpleNamespace

import tiktoken
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types.Document import Document

DEFAULT_MAX_CHUNK_SIZE = 4095  # half of cognee's 8191 default
CHUNKER_TOKENIZER_MODEL = "gpt-4o"  # fixed, only decides where chunk boundaries fall


def read_source(args) -> str:
    """Resolve exactly one of --file / --dir / --text into raw text."""
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    files = sorted(Path(args.dir).glob("*.txt"))
    return "\n\n".join(path.read_text(encoding="utf-8") for path in files)


def chunk_text(text: str, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[str]:
    """Split text into chunks using cognee's TextChunker."""
    _patch_chunker_tokenizer()
    return asyncio.run(_chunk_text(text, max_chunk_size))


def sample_chunks(chunks: list[str], sample_size: int = 3, seed: int = 42) -> list[str]:
    """Pick sample_size chunks deterministically; use all if there are fewer."""
    if len(chunks) <= sample_size:
        return chunks
    return random.Random(seed).sample(chunks, sample_size)


def sampled_chunks_from(text: str, args) -> list[str]:
    """Chunk then sample. A raw --text string is already the single chunk."""
    if args.text is not None:
        return [text]
    return sample_chunks(chunk_text(text, args.max_chunk_size), args.samples, args.seed)


def _patch_chunker_tokenizer(tokenizer_model: str = CHUNKER_TOKENIZER_MODEL) -> None:
    """Point cognee's chunker at a local tokenizer so it needs no embedding model."""
    encoding = tiktoken.encoding_for_model(tokenizer_model)
    tokenizer = SimpleNamespace(count_tokens=lambda text: len(encoding.encode(text)))
    engine = SimpleNamespace(tokenizer=tokenizer)
    chunk_by_sentence = importlib.import_module("cognee.tasks.chunks.chunk_by_sentence")
    chunk_by_sentence.get_embedding_engine = lambda: engine


async def _chunk_text(text: str, max_chunk_size: int) -> list[str]:
    async def text_stream():
        yield text

    document = Document(
        name="input",
        raw_data_location="input",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunker = TextChunker(document, lambda: text_stream(), max_chunk_size)
    return [chunk.text async for chunk in chunker.read()]
