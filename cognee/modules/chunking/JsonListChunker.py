import json
from os.path import basename
from typing import Any
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _flatten_nested(obj: Any, prefix: str = "", separator: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dict or list into a dot-separated key/value mapping.

    Arrays of scalars are joined as comma-separated strings.
    Arrays of objects are each flattened with an integer index suffix.
    This preserves every leaf value while making relationships traversable as text.

    Examples::

        >>> _flatten_nested({"a": {"b": 1}})
        {"a.b": 1}

        >>> _flatten_nested({"tags": ["x", "y"]})
        {"tags": "x, y"}

        >>> _flatten_nested({"users": [{"name": "Alice"}, {"name": "Bob"}]})
        {"users.0.name": "Alice", "users.1.name": "Bob"}
    """
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}{separator}{key}" if prefix else key
            result.update(_flatten_nested(value, full_key, separator))
    elif isinstance(obj, list):
        if all(isinstance(el, (str, int, float, bool, type(None))) for el in obj):
            # Scalar list → join as a human-readable string
            result[prefix] = ", ".join(str(el) for el in obj)
        else:
            # Mixed/object list → index each element
            for i, el in enumerate(obj):
                full_key = f"{prefix}{separator}{i}" if prefix else str(i)
                result.update(_flatten_nested(el, full_key, separator))
    else:
        result[prefix] = obj
    return result


def _item_to_text(item: Any, flatten_nested: bool) -> str:
    """Convert a single JSON list item to a searchable text representation.

    When ``flatten_nested`` is True and the item is a dict, each key/value pair is
    rendered on its own line as ``key: value`` so the LLM and the vector index see
    natural attribute sentences rather than raw Python repr strings.

    For scalar values the string representation is returned unchanged.
    """
    if flatten_nested and isinstance(item, dict):
        flat = _flatten_nested(item)
        lines = [f"{k}: {v}" for k, v in flat.items()]
        return "\n".join(lines)
    return str(item)


def _find_nested_array(data: Any, json_path: str | None) -> tuple[list, dict]:
    """Locate the array to chunk and collect its parent-context fields.

    If ``json_path`` is provided (a dot-separated key path, e.g. ``"data.items"``),
    the function walks the parsed JSON object along those keys and returns the array
    found at that location together with any sibling scalar fields at every level as
    parent context.

    If ``json_path`` is None and the root is a dict, the function auto-detects the
    first top-level value that is a non-empty list of objects and uses it, capturing
    all remaining top-level scalar fields as parent context.

    Returns ``(items_list, parent_context_dict)``.

    Raises ``ValueError`` when:
    - ``json_path`` points to something that is not a list.
    - No suitable nested array is found (auto-detection mode).
    - The root is neither a list nor a dict.
    """
    if isinstance(data, list):
        return data, {}

    if not isinstance(data, dict):
        raise ValueError(
            f"JsonListChunker received unsupported root type {type(data).__name__}; "
            "expected a JSON list or object."
        )

    # --- Explicit json_path walk ---
    if json_path is not None:
        keys = json_path.split(".")
        node = data
        parent_context: dict[str, Any] = {}
        for depth, key in enumerate(keys):
            if not isinstance(node, dict) or key not in node:
                raise ValueError(
                    f"json_path {json_path!r}: key {key!r} not found at depth {depth}."
                )
            # Collect scalar siblings as context
            for sibling_key, sibling_val in node.items():
                if sibling_key != key and not isinstance(sibling_val, (dict, list)):
                    ctx_key = ".".join(keys[:depth] + [sibling_key]) if depth else sibling_key
                    parent_context[ctx_key] = sibling_val
            node = node[key]
        if not isinstance(node, list):
            raise ValueError(
                f"json_path {json_path!r} points to a {type(node).__name__}, not a list."
            )
        return node, parent_context

    # --- Auto-detect: find the first top-level array of objects ---
    parent_context = {}
    array_key: str | None = None
    array_val: list | None = None
    for key, value in data.items():
        if (
            isinstance(value, list)
            and value
            and any(isinstance(el, dict) for el in value)
            and array_key is None
        ):
            array_key = key
            array_val = value
        elif not isinstance(value, (dict, list)):
            parent_context[key] = value

    if array_val is None:
        raise ValueError(
            "JsonListChunker could not auto-detect a nested array of objects in the "
            "JSON object. Supply json_path to point to the target array explicitly."
        )
    logger.debug(
        "JsonListChunker auto-detected nested array",
        array_key=array_key,
    )
    return array_val, parent_context


class JsonListChunker(Chunker):
    """Chunk a JSON document into one item-per-chunk, with full nested-JSON support.

    **Supported root shapes**

    - ``[...]`` — A JSON array at the root (original behaviour preserved).
    - ``{..., "key": [...], ...}`` — A JSON object containing one or more arrays.
      The first top-level array of objects is auto-selected; pass ``json_path`` to
      override.
    - Any nesting depth reachable via a dot-separated ``json_path``.

    **Constructor parameters** (passed through ``get_text`` / document contract):

    ``json_path`` (``str | None``, default ``None``)
        Dot-separated path to the target array inside a root object,
        e.g. ``"data.items"``.  Ignored when the root is already a list.

    ``flatten_nested`` (``bool``, default ``True``)
        When True, nested dict/list values inside each item are recursively
        flattened and rendered as ``"key: value"`` lines so every attribute
        becomes independently searchable.  Disable for items that must stay
        as Python repr strings (backwards-compatible mode).

    ``include_parent_context`` (``bool``, default ``True``)
        When True and parent-context fields were collected (e.g. a wrapper
        object's scalar siblings), prepend them to each chunk's text so the
        retriever sees the full provenance of each item.

    Backwards compatibility
    -----------------------
    The chunker continues to work without any extra arguments when the
    document content is a plain JSON list.  Existing stored documents carry
    ``chunker_id = "json_list_chunker_v1"`` and are only re-chunked if the
    document changes, so no migration is required.
    """

    chunker_id = "json_list_chunker_v1"

    def __init__(
        self,
        document,
        get_text: callable,
        max_chunk_size: int,
        *,
        json_path: str | None = None,
        flatten_nested: bool = True,
        include_parent_context: bool = True,
    ):
        super().__init__(document, get_text, max_chunk_size)
        self.json_path = json_path
        self.flatten_nested = flatten_nested
        self.include_parent_context = include_parent_context

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        content = ""

        async for content_text in self.get_text():
            if content_text is not None:
                content += content_text

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JsonListChunker could not parse document {document_name!r} as JSON: {exc}"
            ) from exc

        items, parent_context = _find_nested_array(data, self.json_path)

        # Build a one-line parent-context prefix reused across all chunks.
        parent_prefix = ""
        if self.include_parent_context and parent_context:
            parent_prefix = (
                "Context: " + "; ".join(f"{k}={v}" for k, v in parent_context.items()) + "\n"
            )

        max_observed_chunk_size = 0
        for index, item in enumerate(items):
            item_text = _item_to_text(item, self.flatten_nested)
            text = parent_prefix + item_text if parent_prefix else item_text
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

            metadata: dict[str, Any] = {
                "index_fields": ["text"],
                "json_list_index": index,
            }
            if parent_context:
                metadata["json_parent_context"] = parent_context
            if self.json_path:
                metadata["json_path"] = self.json_path

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
