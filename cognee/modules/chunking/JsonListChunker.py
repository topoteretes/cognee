import json
from os.path import basename
from typing import Any, Optional
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _flatten_json_arrays(
    data: Any,
    current_path: str = "",
    parent_keys: Optional[list[str]] = None,
) -> list[tuple[Any, str, list[str]]]:
    """Recursively traverse a JSON structure and extract list items with their paths.

    The function walks dicts and lists depth-first.  When it encounters a **list**,
    each element of that list is emitted as a result tuple.  Elements that are
    themselves dicts or lists are **not** further decomposed — the whole element
    is kept as a single chunk so that structural context is preserved.

    Parameters
    ----------
    data:
        The parsed JSON data (dict, list, or scalar).
    current_path:
        JSONPath-style path accumulator (e.g. ``"records.items"``).
    parent_keys:
        Stack of enclosing dictionary keys, used to preserve high-level
        contextual parent keys across all generated record chunks.

    Returns
    -------
    list[tuple[item, json_path, parent_keys]]
        Each tuple contains the extracted item, its JSONPath-style location,
        and the list of enclosing parent keys.
    """
    if parent_keys is None:
        parent_keys = []

    results: list[tuple[Any, str, list[str]]] = []

    if isinstance(data, dict):
        # Traverse into each value to find nested arrays.
        for key, value in data.items():
            child_path = f"{current_path}.{key}" if current_path else key
            child_parents = parent_keys + [key]
            results.extend(_flatten_json_arrays(value, child_path, child_parents))
    elif isinstance(data, list):
        # Emit each element of the list as a chunk.
        for idx, item in enumerate(data):
            element_path = f"{current_path}[{idx}]" if current_path else f"[{idx}]"
            # If the element is itself a dict/list, we still emit it as one chunk
            # (keeping the whole element as the chunk payload) — we do NOT recurse
            # further so structural context is preserved.
            results.append((item, element_path, list(parent_keys)))
    else:
        # Scalar value at the top level or a dict-only branch — emit as a single chunk.
        if current_path:
            results.append((data, current_path, list(parent_keys)))

    return results


class JsonListChunker(Chunker):
    """Chunk a JSON document into one chunk per list element.

    Supports three modes of operation:

    1. **Flat top-level array** (original behaviour):
       ``[{"a": 1}, {"a": 2}]`` → two chunks, one per element.

    2. **Array-root nested flattening** (new):
       ``{"records": [{"items": [1, 2]}]}`` → the chunker automatically
       detects nested arrays and flattens each element into its own chunk,
       preserving the JSONPath (e.g. ``"records[0].items[0]"``) as metadata.

    3. **Key/JSONPath selector** (new):
       The caller can pass ``json_path`` to restrict extraction to a specific
       sub-tree of the document.

    Each emitted chunk carries:
    - ``json_path``: the JSONPath-style location of the element.
    - ``parent_keys``: list of enclosing dictionary keys for structural context.
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

    @staticmethod
    def _extract_by_path(data: Any, path: str) -> Any:
        """Extract a sub-tree of *data* identified by a dot/bracket path.

        ``"records.items"`` → ``data["records"]["items"]``
        ``"records[0].name"`` → ``data["records"][0]["name"]``
        """
        tokens: list[str | int] = []
        remaining = path
        while remaining:
            if remaining.startswith("["):
                end = remaining.index("]")
                tokens.append(int(remaining[1:end]))
                remaining = remaining[end + 1:]
            else:
                dot = remaining.find(".")
                bracket = remaining.find("[")
                if dot == -1 and bracket == -1:
                    tokens.append(remaining)
                    remaining = ""
                elif dot != -1 and (bracket == -1 or dot < bracket):
                    tokens.append(remaining[:dot])
                    remaining = remaining[dot + 1:]
                else:
                    tokens.append(remaining[:bracket])
                    remaining = remaining[bracket:]
        node = data
        for token in tokens:
            if isinstance(token, int):
                node = node[token]
            else:
                node = node[token]
        return node

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        data = json.loads(content)

        # If a json_path selector is provided, extract that sub-tree.
        if self.json_path:
            data = self._extract_by_path(data, self.json_path)

        # Determine the list of items to chunk.
        if isinstance(data, list):
            # Original behaviour: flat top-level array — one chunk per element.
            items_with_paths = [
                (item, f"[{idx}]", []) for idx, item in enumerate(data)
            ]
        elif isinstance(data, dict):
            # New behaviour: traverse the dict to find and flatten nested arrays.
            items_with_paths = _flatten_json_arrays(data)
        else:
            # Scalar JSON — emit as a single chunk.
            items_with_paths = [(data, "$", [])]

        max_observed_chunk_size = 0
        for index, (item, json_path, parent_keys) in enumerate(items_with_paths):
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
                    json_path=json_path,
                )

            chunk_metadata = {
                "index_fields": ["text"],
                "json_list_index": index,
                "json_path": json_path,
                "parent_keys": parent_keys,
            }

            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{document_id}-{index}-{json_path}"),
                text=text,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=index,
                cut_type="json_list_item",
                contains=[],
                importance_weight=self.document.importance_weight,
                document_id=document_id,
                document_name=document_name,
                metadata=chunk_metadata,
            )

        if max_observed_chunk_size > self.max_chunk_size:
            logger.warning(
                "JsonListChunker max item size exceeds max_chunk_size",
                max_observed_chunk_size=max_observed_chunk_size,
                max_chunk_size=self.max_chunk_size,
                document_name=document_name,
            )
