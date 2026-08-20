import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from os.path import basename
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.data.processing.document_types.Document import Document

logger = get_logger()

PathToken: TypeAlias = str | int
SelectorToken: TypeAlias = str | int | None


@dataclass(frozen=True)
class _ArrayOccurrence:
    path: tuple[PathToken, ...]
    items: list[Any]
    context: dict[str, Any]


def _format_json_path(path: tuple[PathToken, ...], *, normalized: bool = False) -> str:
    """Render a tokenized JSON path without conflating object keys and array indexes."""
    formatted = ""
    for token in path:
        if isinstance(token, int):
            formatted += "[*]" if normalized else f"[{token}]"
        else:
            formatted += f".{token}" if formatted else token
    return formatted


def _scalar_context(container: dict[str, Any]) -> dict[str, Any]:
    """Return scalar siblings that describe descendants of ``container``."""
    return {key: value for key, value in container.items() if not isinstance(value, (dict, list))}


def _find_arrays(
    node: Any,
    path: tuple[PathToken, ...] = (),
    context: dict[str, Any] | None = None,
) -> list[_ArrayOccurrence]:
    """Find arrays at any depth while retaining scalar values from every ancestor."""
    context = context or {}
    occurrences: list[_ArrayOccurrence] = []

    if isinstance(node, dict):
        child_context = {**context, **_scalar_context(node)}
        for key, value in node.items():
            child_path = (*path, key)
            if isinstance(value, list):
                occurrences.append(_ArrayOccurrence(child_path, value, child_context))
                occurrences.extend(_find_arrays(value, child_path, child_context))
            elif isinstance(value, dict):
                occurrences.extend(_find_arrays(value, child_path, child_context))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            occurrences.extend(_find_arrays(value, (*path, index), context))

    return occurrences


def _is_descendant(path: tuple[PathToken, ...], ancestor: tuple[PathToken, ...]) -> bool:
    return len(path) > len(ancestor) and path[: len(ancestor)] == ancestor


def _leaf_array_occurrences(occurrences: list[_ArrayOccurrence]) -> list[_ArrayOccurrence]:
    """Exclude container arrays when they contain a more specific child array."""
    return [
        occurrence
        for occurrence in occurrences
        if not any(
            _is_descendant(other.path, occurrence.path)
            for other in occurrences
            if other is not occurrence
        )
    ]


def _parse_json_path(json_path: str) -> tuple[SelectorToken, ...]:
    """Parse dotted object keys plus ``[index]`` or ``[*]`` array selectors."""
    if not json_path:
        raise ValueError("json_path must not be empty.")

    tokens: list[SelectorToken] = []
    for segment in json_path.split("."):
        match = re.fullmatch(r"([^\[\].]+)((?:\[(?:\*|\d+)\])*)", segment)
        if match is None:
            raise ValueError(
                f"json_path '{json_path}' must use dotted keys and optional [index] or [*] selectors."
            )

        tokens.append(match.group(1))
        for array_selector in re.findall(r"\[(\*|\d+)\]", match.group(2)):
            tokens.append(None if array_selector == "*" else int(array_selector))

    return tuple(tokens)


def _path_matches(selector: tuple[SelectorToken, ...], path: tuple[PathToken, ...]) -> bool:
    return len(selector) == len(path) and all(
        expected is None or expected == actual for expected, actual in zip(selector, path)
    )


def _select_array_occurrences(
    data: dict[str, Any], json_path: str | None
) -> list[_ArrayOccurrence]:
    occurrences = _find_arrays(data)

    if json_path is not None:
        selector = _parse_json_path(json_path)
        selected = [
            occurrence for occurrence in occurrences if _path_matches(selector, occurrence.path)
        ]
        if not selected:
            raise ValueError(
                f"json_path '{json_path}' does not point to a JSON list in the document."
            )
        return selected

    leaves = _leaf_array_occurrences(occurrences)
    candidates: dict[str, list[_ArrayOccurrence]] = {}
    for occurrence in leaves:
        candidates.setdefault(_format_json_path(occurrence.path, normalized=True), []).append(
            occurrence
        )

    if not candidates:
        raise ValueError(
            "JsonListChunker could not find any JSON list in the document. "
            "Pass json_path explicitly to point at the array to chunk."
        )
    if len(candidates) > 1:
        paths = ", ".join(f"'{path}'" for path in candidates)
        raise ValueError(
            "JsonListChunker found multiple leaf JSON lists in the document: "
            f"{paths}. Pass json_path to specify which one to chunk."
        )

    return next(iter(candidates.values()))


def _contextual_text(item: Any, context: dict[str, Any], array_path: str) -> str:
    """Put source context and structural path in the indexed text, not metadata alone."""
    return json.dumps(
        {"_context": context, "_json_path": array_path, "_record": item},
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )


class JsonListChunker(Chunker):
    """Chunk a JSON list or a selected nested JSON array into record chunks."""

    json_path: str | None = None

    def __init__(
        self,
        document: "Document",
        get_text: Callable[[], AsyncIterator[str | None]],
        max_chunk_size: int,
        json_path: str | None = None,
    ):
        super().__init__(document, get_text, max_chunk_size)
        self.json_path = json_path if json_path is not None else type(self).json_path

    @classmethod
    def with_json_path(cls, json_path: str) -> type["JsonListChunker"]:
        """Return a chunker class usable by ``cognify(chunker=...)`` with a path selector."""
        _parse_json_path(json_path)
        return type(f"{cls.__name__}AtPath", (cls,), {"json_path": json_path})

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        data = json.loads(content)
        if isinstance(data, list):
            occurrences = [_ArrayOccurrence((), data, {})]
        elif isinstance(data, dict):
            occurrences = _select_array_occurrences(data, self.json_path)
        else:
            raise ValueError(
                "JsonListChunker expects the document content to be a JSON list "
                "or a JSON object containing one."
            )

        max_observed_chunk_size = 0
        chunk_index = 0
        for occurrence in occurrences:
            array_path = _format_json_path(occurrence.path, normalized=True)
            for json_list_index, item in enumerate(occurrence.items):
                item_path = (*occurrence.path, json_list_index)
                formatted_item_path = _format_json_path(item_path)
                is_nested = bool(occurrence.path)
                text = (
                    _contextual_text(item, occurrence.context, array_path)
                    if is_nested
                    else str(item)
                )
                chunk_size = len(text.split())
                max_observed_chunk_size = max(max_observed_chunk_size, chunk_size)

                if chunk_size > self.max_chunk_size:
                    logger.warning(
                        "JsonListChunker item exceeds max_chunk_size",
                        chunk_index=chunk_index,
                        chunk_size=chunk_size,
                        max_chunk_size=self.max_chunk_size,
                        document_name=document_name,
                    )

                metadata = {
                    "index_fields": ["text"],
                    "json_list_index": json_list_index,
                    "json_path": formatted_item_path,
                }
                if is_nested:
                    metadata["json_array_path"] = array_path
                    if occurrence.context:
                        metadata["json_context"] = occurrence.context

                yield DocumentChunk(
                    id=uuid5(
                        NAMESPACE_OID,
                        f"{document_id}-{formatted_item_path}"
                        if is_nested
                        else f"{document_id}-{json_list_index}",
                    ),
                    text=text,
                    chunk_size=chunk_size,
                    is_part_of=self.document,
                    chunk_index=chunk_index,
                    cut_type="json_list_item",
                    contains=[],
                    importance_weight=self.document.importance_weight,
                    document_id=document_id,
                    document_name=document_name,
                    metadata=metadata,
                )
                chunk_index += 1

        if max_observed_chunk_size > self.max_chunk_size:
            logger.warning(
                "JsonListChunker max item size exceeds max_chunk_size",
                max_observed_chunk_size=max_observed_chunk_size,
                max_chunk_size=self.max_chunk_size,
                document_name=document_name,
            )
