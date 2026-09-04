import inspect
from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from typing import Any, Optional

from cognee.tasks.ingestion.data_item import DataItem


def _normalize_filename(filename: Optional[str], index: int, keep_path: bool = False) -> str:
    """The buffered item's filename.

    An upload's ``filename`` is kept whole (separators normalised) because a
    relative path in it means a folder upload, which resolve_data_directories
    reassembles later in the pipeline. A plain stream's ``name`` is a path on
    this machine, so only its basename is kept.
    """
    if not filename:
        return f"upload_{index}.bin"
    normalized = str(filename).replace("\\", "/")
    if not keep_path:
        normalized = normalized.split("/")[-1]
    return normalized or f"upload_{index}.bin"


async def _read_stream_bytes(stream: Any) -> bytes:
    if not hasattr(stream, "read"):
        raise TypeError(f"Expected stream-like object, got: {type(stream)}")

    # Best effort to read from the start of the stream.
    if hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except Exception:
            pass

    data = stream.read()
    if inspect.isawaitable(data):
        data = await data

    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Unsupported stream payload type: {type(data)}")

    return bytes(data)


async def materialize_stream_for_background(data_item: Any, index: int = 0) -> Any:
    if isinstance(data_item, DataItem):
        # Copy EVERY DataItem field: dropping one here silently breaks the
        # background path only (system_metadata carries the DLT routing stamp).
        return DataItem(
            data=await materialize_stream_for_background(data_item.data, index=index),
            label=data_item.label,
            external_metadata=data_item.external_metadata,
            system_metadata=data_item.system_metadata,
            data_id=data_item.data_id,
        )

    if isinstance(data_item, list):
        return [
            await materialize_stream_for_background(item, index=i)
            for i, item in enumerate(data_item)
        ]

    # Keep stable primitives untouched.
    if isinstance(data_item, str):
        return data_item

    stream = getattr(data_item, "file", data_item if hasattr(data_item, "read") else None)
    if stream is None:
        return data_item

    payload = await _read_stream_bytes(stream)
    buffer = SpooledTemporaryFile(mode="w+b")
    buffer.write(payload)
    buffer.seek(0)

    upload_name = getattr(data_item, "filename", None)
    filename = (
        _normalize_filename(upload_name, index=index, keep_path=True)
        if upload_name
        else _normalize_filename(getattr(stream, "name", None), index=index)
    )

    # Ingestion path supports objects exposing `.file` and `.filename`.
    return SimpleNamespace(file=buffer, filename=filename)
