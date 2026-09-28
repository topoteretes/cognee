import inspect
from dataclasses import replace
from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem

logger = get_logger()


def _normalize_filename(filename: str | None, index: int) -> str:
    if not filename:
        return f"upload_{index}.bin"
    normalized = str(filename).replace("\\", "/").split("/")[-1]
    return normalized or f"upload_{index}.bin"


async def _read_stream_bytes(stream: Any) -> bytes:
    if not hasattr(stream, "read"):
        raise TypeError(f"Expected stream-like object, got: {type(stream)}")

    # Best effort to read from the start of the stream.
    if hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except Exception:
            logger.debug("Ignoring exception in _read_stream_bytes", exc_info=True)

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
        # dataclasses.replace copies every other field (label, external_metadata,
        # system_metadata, data_id, node_set, ...) so a new DataItem field can
        # never be silently dropped from the background path again.
        return replace(
            data_item,
            data=await materialize_stream_for_background(data_item.data, index=index),
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
    buffer = SpooledTemporaryFile(mode="w+b")  # noqa: SIM115 - returned to the caller, who owns it
    buffer.write(payload)
    buffer.seek(0)

    filename = _normalize_filename(
        getattr(data_item, "filename", None) or getattr(stream, "name", None),
        index=index,
    )

    # Ingestion path supports objects exposing `.file` and `.filename`.
    return SimpleNamespace(file=buffer, filename=filename)
