import json
from os.path import basename
from typing import Any, Optional
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _find_top_level_lists(obj: Any, path: str = "") -> list[tuple[str, list]]:
    """Find all list values that are direct children of dict keys.

    Recurses into nested dicts but NOT into list items, so that nested
    arrays inside individual records are not mistaken for the primary
    data list.

    Returns a list of ``(dot_separated_path, list_value)`` tuples.
    """
    results = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, list):
                results.append((child_path, value))
            elif isinstance(value, dict):
                results.extend(_find_top_level_lists(value, child_path))
    return results


def _resolve_json_path(obj: Any, json_path: str) -> Any:
    """Resolve a dot-separated path (optional bracket indexes) to a nested value.

    Examples::
        "records.items"
        "data[0].items"
    """
    normalized = json_path.replace("[", ".").replace("]", "")
    parts = normalized.split(".")

    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Path '{json_path}' not found in JSON: key '{part}' missing")
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(
                    f"Path '{json_path}' not found in JSON: cannot index list with '{part}'"
                )
            if idx >= len(current):
                raise ValueError(
                    f"Path '{json_path}' not found in JSON: "
                    f"index {idx} out of range (list length {len(current)})"
                )
            current = current[idx]
        else:
            raise ValueError(
                f"Path '{json_path}' not found in JSON: cannot traverse {type(current).__name__}"
            )

    return current


def _extract_parent_context(obj: Any, json_path: str) -> dict[str, Any]:
    """Collect primitive key-value pairs from the ancestors of ``json_path``.

    These values give each chunk its enclosing structural context (e.g.
    ``{"company": "Acme", "department": "Engineering"}``) without pulling
    in sibling lists or nested dicts.
    """
    normalized = json_path.replace("[", ".").replace("]", "")
    parts = normalized.split(".")

    context: dict[str, Any] = {}
    current = obj

    # Walk to the parent container of the target list.
    # For path "a.b.c" -> traverse to "a.b", then collect primitives from each ancestor.
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, dict):
            # Collect all primitive siblings at this level (excluding the path key)
            for key, value in current.items():
                if key == part:
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    context[key] = value
            if part in current:
                current = current[part]
            else:
                break
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                break
        else:
            break

    # Finally, collect primitives from the direct parent of the list.
    # When path is a single segment (e.g. "employees"), `current` is still `obj`,
    # so we collect all primitive keys except the list key itself.
    if isinstance(current, dict):
        list_key = parts[-1]
        for key, value in current.items():
            if key == list_key:
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                context[key] = value

    return context


class JsonListChunker(Chunker):
    """Chunk a JSON document into one stringified item per chunk.

    Supports both flat top-level JSON arrays and nested JSON structures
    where the target list lives inside one or more dict layers.

    When the parsed JSON is a dict, the chunker either resolves an
    explicit ``json_path`` or auto-detects the most likely data list
    (the longest top-level list found in the structure).
    """

    def __init__(
        self,
        document,
        get_text: callable,
        max_chunk_size: int,
        json_path: Optional[str] = None,
    ):
        super().__init__(document, get_text, max_chunk_size)
        self.json_path = json_path

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        items = json.loads(content)

        if isinstance(items, list):
            # Flat list at root — preserve original behavior
            list_to_chunk = items
            list_path = ""
            parent_context: dict[str, Any] = {}
        elif isinstance(items, dict):
            if self.json_path:
                list_to_chunk = _resolve_json_path(items, self.json_path)
                if not isinstance(list_to_chunk, list):
                    raise ValueError(
                        f"json_path '{self.json_path}' does not point to a list "
                        f"(found {type(list_to_chunk).__name__})"
                    )
                list_path = self.json_path
            else:
                nested_lists = _find_top_level_lists(items)
                if not nested_lists:
                    raise ValueError(
                        "JsonListChunker expects the document content to be a "
                        "JSON list or a dict containing at least one nested list."
                    )
                # Pick the longest list — most likely the primary data
                list_path, list_to_chunk = max(nested_lists, key=lambda x: len(x[1]))
                if len(nested_lists) > 1:
                    logger.info(
                        "Multiple nested lists found in JSON document, using longest",
                        document_name=document_name,
                        selected_path=list_path,
                        list_length=len(list_to_chunk),
                        all_paths=[p for p, _ in nested_lists],
                    )

            parent_context = _extract_parent_context(items, list_path)
        else:
            raise ValueError(
                f"JsonListChunker expects JSON list or dict, got {type(items).__name__}."
            )

        max_observed_chunk_size = 0
        for index, item in enumerate(list_to_chunk):
            text = str(item)
            chunk_size = len(text.split())
            max_observed_chunk_size = max(max_observed_chunk_size, chunk_size)

            if chunk_size > self.max_chunk_size:
                logger.warning(
                    "JsonListChunker item exceeds max_chunk_size",
                    chunk_index=index,
                    chunk_size=chunk_size,
                    max_chunk_size=self.max_chunk_size,
                    document_name=document_name,
                )

            # Build the per-item json_path, e.g. "records.items[0]"
            item_json_path = f"{list_path}[{index}]" if list_path else f"[{index}]"

            # Keep original ID format for flat lists (backward compat)
            id_seed = (
                f"{document_id}-{index}" if not list_path else f"{document_id}-{list_path}-{index}"
            )

            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, id_seed),
                text=text,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=index,
                cut_type="json_list_item",
                contains=[],
                importance_weight=self.document.importance_weight,
                document_id=document_id,
                document_name=document_name,
                metadata={
                    "index_fields": ["text"],
                    "json_list_index": index,
                    "json_path": item_json_path,
                    **parent_context,
                },
            )

        if max_observed_chunk_size > self.max_chunk_size:
            logger.warning(
                "JsonListChunker max item size exceeds max_chunk_size",
                max_observed_chunk_size=max_observed_chunk_size,
                max_chunk_size=self.max_chunk_size,
                document_name=document_name,
            )
