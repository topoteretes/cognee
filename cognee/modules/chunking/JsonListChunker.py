import json
from os.path import basename
from typing import Any
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def find_json_arrays(obj: Any, path: str = "") -> list[tuple[str, list]]:
    """
    Recursively find all arrays in a JSON object.

    Returns list of (json_path, array) tuples.
    """
    arrays = []

    if isinstance(obj, list):
        arrays.append((path or "$", obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else f"$.{key}"
            if isinstance(value, list):
                arrays.append((new_path, value))
            elif isinstance(value, dict):
                arrays.extend(find_json_arrays(value, new_path))

    return arrays


def get_parent_context(obj: Any, target_path: str) -> dict[str, Any]:
    """
    Extract sibling values from all ancestor levels of the target array.

    Returns a dict of simple (non-list, non-dict) key-value pairs from all levels
    leading to the target array, excluding the target array itself.
    """
    if not isinstance(obj, dict):
        return {}

    # Navigate to the parent of the target, collecting context from each level
    parts = target_path.strip("$.").split(".")
    if len(parts) <= 1:
        # Target is at root level
        target_key = parts[0] if parts else ""
        context = {}
        for key, value in obj.items():
            if key != target_key and not isinstance(value, (list, dict)):
                context[key] = value
        return context

    # Navigate through the path, collecting context from each level
    context = {}
    current = obj
    for i, part in enumerate(parts[:-1]):  # All parts except the last (target key)
        if isinstance(current, dict) and part in current:
            # Collect simple siblings at this level
            for key, value in current.items():
                if key != part and not isinstance(value, (list, dict)):
                    context[key] = value
            current = current[part]
        else:
            return {}

    # At the parent level, collect siblings of the target array
    target_key = parts[-1]
    if isinstance(current, dict):
        for key, value in current.items():
            if key != target_key and not isinstance(value, (list, dict)):
                context[key] = value

    return context


class JsonListChunker(Chunker):
    """Chunk a JSON list document into one stringified item per chunk.

    Supports both flat JSON lists at root level and nested arrays within objects.
    Can be configured with a json_path to target a specific nested array.

    Configuration (via document.metadata or class attributes):
    - json_path: Optional JSONPath string (e.g., "records.items") to target a specific nested array
    - auto_detect: If True and no json_path provided, auto-detect nested arrays (default: True)
    """

    # Class-level defaults (can be overridden per subclass or instance)
    auto_detect: bool = True
    json_path: str | None = None

    def __init__(self, document, get_text: callable, max_chunk_size: int):
        super().__init__(document, get_text, max_chunk_size)
        # Allow per-instance config override via document.metadata
        meta = getattr(document, "metadata", {}) or {}
        self.json_path = meta.get("json_path", self.json_path)
        self.auto_detect = meta.get("auto_detect", self.auto_detect)

    chunker_id = "json_list_chunker_v1"

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in document {document_name}: {e}")

        # Determine the target array(s) to chunk
        target_arrays = self._resolve_target_arrays(data, document_name)

        if not target_arrays:
            raise ValueError(
                f"No suitable JSON array found in document {document_name}. "
                f"Provide a valid json_path or ensure the document contains a JSON array."
            )

        if len(target_arrays) > 1:
            paths = [path for path, _ in target_arrays]
            raise ValueError(
                f"Multiple JSON arrays found in document {document_name}: {paths}. "
                f"Specify a json_path to select one (e.g., '$.records.items')."
            )

        target_path, items = target_arrays[0]

        if not isinstance(items, list):
            raise ValueError(f"Target path {target_path} does not resolve to a JSON list.")

        # Extract parent context (sibling values) for metadata enrichment
        parent_context = get_parent_context(data, target_path)

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

            # Build json_path for this specific item (e.g., "$.records.items[0]")
            item_json_path = f"{target_path}[{index}]"

            yield DocumentChunk(
                chunker_id=self.chunker_id,
                id=uuid5(NAMESPACE_OID, f"{document_id}-{index}"),
                text=str(item),
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
                    "parent_context": parent_context,
                },
            )

        if max_observed_chunk_size > self.max_chunk_size:
            logger.warning(
                "JsonListChunker max item size exceeds max_chunk_size",
                max_observed_chunk_size=max_observed_chunk_size,
                max_chunk_size=self.max_chunk_size,
                document_name=document_name,
            )

    def _resolve_target_arrays(self, data: Any, document_name: str) -> list[tuple[str, list]]:
        """Resolve the target array(s) based on json_path config or auto-detection."""
        # If explicit json_path provided, use it
        if self.json_path:
            path = self.json_path if self.json_path.startswith("$") else f"$.{self.json_path}"
            try:
                target = self._get_by_path(data, path)
                if isinstance(target, list):
                    return [(path, target)]
                else:
                    raise ValueError(
                        f"json_path '{self.json_path}' does not resolve to a list in {document_name}."
                    )
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Invalid json_path '{self.json_path}' for {document_name}: {e}")

        # Auto-detect: find all arrays in the JSON structure
        if self.auto_detect:
            arrays = find_json_arrays(data)
            if arrays:
                return arrays

        # Fallback: try root-level list (original behavior)
        if isinstance(data, list):
            return [("$", data)]

        return []

    def _get_by_path(self, obj: Any, path: str) -> Any:
        """Navigate to a value using a JSONPath-like string (e.g., '$.records.items')."""
        if not path or path == "$":
            return obj

        parts = path.strip("$.").split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise KeyError(f"Path '{path}' not found at part '{part}'")
        return current
