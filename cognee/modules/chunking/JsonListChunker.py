import json
from os.path import basename
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _sibling_context(container: dict) -> dict:
    """Scalar values living next to an array in its parent dict, kept as chunk context."""
    return {key: value for key, value in container.items() if not isinstance(value, (list, dict))}


def _find_arrays(obj, path=""):
    """Recursively find every JSON array nested inside a dict.

    Returns a list of (path, array, context) tuples, where `context` is the
    sibling scalar values living alongside the array in its parent dict.
    """
    found = []
    if isinstance(obj, dict):
        context = _sibling_context(obj)
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, list):
                found.append((new_path, value, context))
            elif isinstance(value, dict):
                found.extend(_find_arrays(value, new_path))
    return found


def _resolve_json_path(data, json_path: str):
    """Navigate a dotted key path (e.g. 'records.items') to a specific array."""
    keys = json_path.split(".")
    current = data
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"json_path '{json_path}' does not exist in the document.")
        current = current[key]

    last_key = keys[-1]
    if not isinstance(current, dict) or last_key not in current:
        raise ValueError(f"json_path '{json_path}' does not exist in the document.")

    array = current[last_key]
    if not isinstance(array, list):
        raise ValueError(f"json_path '{json_path}' does not point to a JSON list.")

    return array, _sibling_context(current)


class JsonListChunker(Chunker):
    """Chunk a JSON list document into one stringified item per chunk.

    Supports three document shapes:
    - A flat top-level list: `[{...}, {...}]` (original behavior, unchanged).
    - A dict with exactly one array nested anywhere inside it: auto-detected.
    - A dict with more than one nested array: the caller must pass `json_path`
      (e.g. "records.items") to say which array to chunk.
    """

    def __init__(self, document, get_text: callable, max_chunk_size: int, json_path: str = None):
        super().__init__(document, get_text, max_chunk_size)
        self.json_path = json_path

    chunker_id = "json_list_chunker_v1"

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        data = json.loads(content)

        base_path = ""
        context = {}

        if isinstance(data, list):
            items = data  # original, flat-list behavior — unchanged
        elif isinstance(data, dict):
            if self.json_path:
                items, context = _resolve_json_path(data, self.json_path)
                base_path = self.json_path
            else:
                candidates = _find_arrays(data)
                if len(candidates) == 0:
                    raise ValueError(
                        "JsonListChunker could not find any JSON list in the document. "
                        "Pass json_path explicitly to point at the array to chunk."
                    )
                if len(candidates) > 1:
                    found_paths = ", ".join(f"'{p}'" for p, _, _ in candidates)
                    raise ValueError(
                        "JsonListChunker found multiple JSON lists in the document: "
                        f"{found_paths}. Pass json_path to specify which one to chunk."
                    )
                base_path, items, context = candidates[0]
        else:
            raise ValueError(
                "JsonListChunker expects the document content to be a JSON list "
                "or a JSON object containing one."
            )

        max_observed_chunk_size = 0
        for index, item in enumerate(items):
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

            item_path = f"{base_path}[{index}]" if base_path else f"[{index}]"
            metadata = {
                "index_fields": ["text"],
                "json_list_index": index,
                "json_path": item_path,
            }
            if context:
                metadata["json_context"] = context

            yield DocumentChunk(
                chunker_id=self.chunker_id,
                id=uuid5(NAMESPACE_OID, f"{document_id}-{index}"),
                text=text,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=index,
                cut_type="json_list_item",
                contains=[],
                importance_weight=self.document.importance_weight,
                document_id=document_id,
                document_name=document_name,
                metadata=metadata,
            )

        if max_observed_chunk_size > self.max_chunk_size:
            logger.warning(
                "JsonListChunker max item size exceeds max_chunk_size",
                max_observed_chunk_size=max_observed_chunk_size,
                max_chunk_size=self.max_chunk_size,
                document_name=document_name,
            )
