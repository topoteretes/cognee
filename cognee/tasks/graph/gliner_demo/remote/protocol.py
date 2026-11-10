"""What a transport does, and the parts of the wire contract every transport shares.

A transport sends one request and answers readiness. Everything else (batching,
retries, the process-wide request budget, readiness caching, limit and model
checks, failure scope, spans) lives in :mod:`.adapter`, the same for HTTP, gRPC
and RabbitMQ.

Requests are the worker's ``POST /v1/extract`` JSON body (``inputs``, ``schema``,
``options``), which HTTP and RabbitMQ send as is and gRPC maps onto its
protobuf messages. Replies come back as :class:`WorkerReply` whose items carry
gliner2's own result dicts, so they are interchangeable with local extraction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .errors import GlinerWorkerRuntimeError

# What a worker advertises when it windows long inputs itself.
WINDOWING_FEATURE = "windowing"


@dataclass(frozen=True)
class WorkerLimits:
    """The worker's published request limits; ``None`` where it does not say."""

    max_inputs: int | None = None
    max_text_chars: int | None = None
    max_batch_size: int | None = None


@dataclass(frozen=True)
class WorkerReadiness:
    """A ready worker. ``features`` is ``None`` when the transport cannot tell
    (gRPC, RabbitMQ): the adapter then checks each reply instead."""

    model: str | None = None
    limits: WorkerLimits = WorkerLimits()
    features: frozenset[str] | None = None


@dataclass(frozen=True)
class WorkerReply:
    items: list[tuple[str, dict[str, Any]]]
    model: str
    elapsed_ms: float | None
    windowed: bool


class WorkerTransport(Protocol):
    #: "http", "grpc" or "amqp".
    name: str
    #: The endpoint with any credentials masked, for logs, spans and errors.
    endpoint_label: str

    async def extract(self, payload: Mapping[str, Any], *, request_id: str) -> WorkerReply:
        """Send one request. Raises a :class:`GlinerWorkerError` subclass on failure."""
        ...

    async def readiness(self) -> WorkerReadiness:
        """Probe the worker once. Raises :class:`GlinerWorkerUnavailableError` when not ready."""
        ...

    async def aclose(self) -> None: ...


def redact_url(url: str) -> str:
    """``scheme://user:pass@host`` -> ``scheme://***@host``; each comma-separated part."""
    parts = []
    for part in url.split(","):
        part = part.strip()
        try:
            split = urlsplit(part)
        except ValueError:
            parts.append("<invalid url>")
            continue
        if "@" in split.netloc:
            # The last "@" ends the userinfo, so a password containing "@" stays hidden.
            host = split.netloc.rsplit("@", 1)[1]
            part = urlunsplit(split._replace(netloc=f"***@{host}"))
        parts.append(part)
    return ",".join(parts)


def parse_json_reply(body: bytes) -> WorkerReply:
    """Parse the HTTP / RabbitMQ success body; a malformed one is a runtime fault."""
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GlinerWorkerRuntimeError(f"The GLiNER worker sent invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise GlinerWorkerRuntimeError("The GLiNER worker's reply is not a JSON object.")
    items = data.get("items")
    model = data.get("model")
    elapsed_ms = data.get("elapsed_ms")
    windowed = data.get("windowed", False)
    if not isinstance(items, list) or not isinstance(model, str):
        raise GlinerWorkerRuntimeError("The GLiNER worker's reply lacks `items` or `model`.")
    if elapsed_ms is not None and not isinstance(elapsed_ms, (int, float)):
        raise GlinerWorkerRuntimeError("The GLiNER worker's `elapsed_ms` is not a number.")
    if not isinstance(windowed, bool):
        raise GlinerWorkerRuntimeError("The GLiNER worker's `windowed` is not a boolean.")
    parsed = []
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not isinstance(item.get("extraction"), dict)
        ):
            raise GlinerWorkerRuntimeError(
                "The GLiNER worker sent an item without a string `id` and an `extraction` object."
            )
        parsed.append((item["id"], item["extraction"]))
    return WorkerReply(
        items=parsed,
        model=model,
        elapsed_ms=float(elapsed_ms) if elapsed_ms is not None else None,
        windowed=windowed,
    )
