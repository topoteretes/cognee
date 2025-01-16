import logging
from typing import AsyncGenerator, Generator
from uuid import NAMESPACE_OID, uuid5

import parso
import tiktoken

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint
from cognee.shared.CodeGraphEntities import CodeFile, CodePart, SourceCodeChunk

logger = logging.getLogger(__name__)


def _count_tokens(tokenizer: tiktoken.Encoding, source_code: str) -> int:
    return len(tokenizer.encode(source_code))


def _get_naive_subchunk_token_counts(
    tokenizer: tiktoken.Encoding, source_code: str, max_subchunk_tokens: int = 8000
) -> list[tuple[str, int]]:
    """Splits source code into subchunks of up to max_subchunk_tokens and counts tokens."""

    token_ids = tokenizer.encode(source_code)
    subchunk_token_counts = []

    for start_idx in range(0, len(token_ids), max_subchunk_tokens):
        subchunk_token_ids = token_ids[start_idx : start_idx + max_subchunk_tokens]
        token_count = len(subchunk_token_ids)
        subchunk = "".join(
            tokenizer.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
            for token_id in subchunk_token_ids
        )
        subchunk_token_counts.append((subchunk, token_count))

    return subchunk_token_counts


def _get_subchunk_token_counts(
    tokenizer: tiktoken.Encoding,
    source_code: str,
    max_subchunk_tokens: int = 8000,
    depth: int = 0,
    max_depth: int = 100,
) -> list[tuple[str, int]]:
    """Splits source code into subchunk and counts tokens for each subchunk."""
    if depth > max_depth:
        return _get_naive_subchunk_token_counts(tokenizer, source_code, max_subchunk_tokens)

    try:
        module = parso.parse(source_code)
    except Exception as e:
        logger.error(f"Error parsing source code: {e}")
        return []

    if not module.children:
        logger.warning("Parsed module has no children (empty or invalid source code).")
        return []

    # Handle cases with only one real child and an EndMarker to prevent infinite recursion.
    if len(module.children) <= 2:
        module = module.children[0]

    subchunk_token_counts = []
    for child in module.children:
        subchunk = child.get_code()
        token_count = _count_tokens(tokenizer, subchunk)

        if token_count == 0:
            continue

        if token_count <= max_subchunk_tokens:
            subchunk_token_counts.append((subchunk, token_count))
            continue

        if child.type == "string":
            subchunk_token_counts.extend(
                _get_naive_subchunk_token_counts(tokenizer, subchunk, max_subchunk_tokens)
            )
            continue

        subchunk_token_counts.extend(
            _get_subchunk_token_counts(
                tokenizer, subchunk, max_subchunk_tokens, depth=depth + 1, max_depth=max_depth
            )
        )

    return subchunk_token_counts


def _get_chunk_source_code(
    code_token_counts: list[tuple[str, int]], overlap: float, max_tokens: int
) -> tuple[list[tuple[str, int]], str]:
    """Generates a chunk of source code from tokenized subchunks with overlap handling."""
    current_count = 0
    cumulative_counts = []
    current_source_code = ""

    for i, (child_code, token_count) in enumerate(code_token_counts):
        current_count += token_count
        cumulative_counts.append(current_count)
        if current_count > max_tokens:
            break
        current_source_code += f"\n{child_code}"

    if current_count <= max_tokens:
        return [], current_source_code.strip()

    cutoff = 1
    for i, cum_count in enumerate(cumulative_counts):
        if cum_count > (1 - overlap) * max_tokens:
            break
        cutoff = i

    return code_token_counts[cutoff:], current_source_code.strip()


def get_source_code_chunks_from_code_part(
    code_file_part: CodePart,
    max_tokens: int = 8192,
    overlap: float = 0.25,
    granularity: float = 0.1,
    model_name: str = "text-embedding-3-large",
) -> Generator[SourceCodeChunk, None, None]:
    """Yields source code chunks from a CodePart object, with configurable token limits and overlap."""
    if not code_file_part.source_code:
        logger.error(f"No source code in CodeFile {code_file_part.id}")
        return

    vector_engine = get_vector_engine()
    embedding_model = vector_engine.embedding_engine.model
    model_name = embedding_model.split("/")[-1]
    tokenizer = tiktoken.encoding_for_model(model_name)
    max_subchunk_tokens = max(1, int(granularity * max_tokens))
    subchunk_token_counts = _get_subchunk_token_counts(
        tokenizer, code_file_part.source_code, max_subchunk_tokens
    )

    previous_chunk = None
    while subchunk_token_counts:
        subchunk_token_counts, chunk_source_code = _get_chunk_source_code(
            subchunk_token_counts, overlap, max_tokens
        )
        if not chunk_source_code:
            continue
        current_chunk = SourceCodeChunk(
            id=uuid5(NAMESPACE_OID, chunk_source_code),
            code_chunk_of=code_file_part,
            source_code=chunk_source_code,
            previous_chunk=previous_chunk,
        )
        yield current_chunk
        previous_chunk = current_chunk


async def get_source_code_chunks(
    data_points: list[DataPoint],
) -> AsyncGenerator[list[DataPoint], None]:
    """Processes code graph datapoints, create SourceCodeChink datapoints."""
    # TODO: Add support for other embedding models, with max_token mapping
    for data_point in data_points:
        try:
            yield data_point
            if not isinstance(data_point, CodeFile):
                continue
            if not data_point.contains:
                logger.warning(f"CodeFile {data_point.id} contains no code parts")
                continue
            for code_part in data_point.contains:
                try:
                    yield code_part
                    for source_code_chunk in get_source_code_chunks_from_code_part(code_part):
                        yield source_code_chunk
                except Exception as e:
                    logger.error(f"Error processing code part: {e}")
        except Exception as e:
            logger.error(f"Error processing data point: {e}")
