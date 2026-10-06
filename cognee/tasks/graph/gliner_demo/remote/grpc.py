"""gRPC transport: ``gliner_worker.v1.GlinerWorker/Extract`` and ``grpc.health.v1``.

Needs the ``gliner-remote-grpc`` extra (grpcio). Message classes come from the
serialized descriptors in :mod:`.grpc_descriptors`, loaded into a private pool,
so no protoc-generated module pins the protobuf runtime.

Several comma-separated endpoints are balanced round-robin, one long-lived
HTTP/2 channel each. A single URL behind an L4 load balancer pins every request
to one replica; list the replicas or use a gRPC-aware (L7) balancer.

The typed reply is turned back into the dicts gliner2 produces (and HTTP passes
through), labels in schema order, so the rest of cognify cannot tell the
transports apart.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from functools import cache
from typing import Any
from urllib.parse import urlsplit

from .errors import (
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from .grpc_descriptors import GLINER_WORKER_PROTO, HEALTH_PROTO
from .protocol import WorkerReadiness, WorkerReply, redact_url

try:
    import grpc
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise GlinerRemoteConfigError(
        "COGNEE_GLINER_TRANSPORT=grpc needs grpcio, which is not installed.",
        remediation='Install it with: pip install "cognee[gliner-remote-grpc]"',
    ) from error

EXTRACT_METHOD = "/gliner_worker.v1.GlinerWorker/Extract"
HEALTH_METHOD = "/grpc.health.v1.Health/Check"
SERVICE_NAME = "gliner_worker.v1.GlinerWorker"


@cache
def _messages() -> dict[str, Any]:
    from google.protobuf import descriptor_pool, message_factory

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(GLINER_WORKER_PROTO)
    pool.AddSerializedFile(HEALTH_PROTO)
    names = {
        "ExtractRequest": "gliner_worker.v1.ExtractRequest",
        "ExtractResponse": "gliner_worker.v1.ExtractResponse",
        "Input": "gliner_worker.v1.Input",
        "Schema": "gliner_worker.v1.Schema",
        "Label": "gliner_worker.v1.Label",
        "Options": "gliner_worker.v1.Options",
        "HealthCheckRequest": "grpc.health.v1.HealthCheckRequest",
        "HealthCheckResponse": "grpc.health.v1.HealthCheckResponse",
    }
    return {
        short: message_factory.GetMessageClass(pool.FindMessageTypeByName(full))
        for short, full in names.items()
    }


_RETRYABLE = frozenset(
    {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        # The worker never answers CANCELLED itself: a client or proxy cut the call.
        "CANCELLED",
        "RESOURCE_EXHAUSTED",
        "ABORTED",
    }
)
_UNAUTHORIZED = frozenset({"UNAUTHENTICATED", "PERMISSION_DENIED"})
_REJECTED = frozenset({"INVALID_ARGUMENT", "OUT_OF_RANGE", "FAILED_PRECONDITION"})


def classify_status(code_name: str, message: str) -> Exception:
    """Map a gRPC status onto the same failure classes HTTP statuses map to."""
    if code_name in _RETRYABLE:
        return GlinerWorkerUnavailableError(message)
    if code_name in _UNAUTHORIZED:
        return GlinerWorkerUnauthorizedError(message)
    if code_name in _REJECTED:
        return GlinerWorkerRejectedError(message)
    return GlinerWorkerRuntimeError(message)


def _target(endpoint: str) -> tuple[str, bool]:
    """``http(s)://host:port`` or ``host:port`` -> (gRPC target, use TLS)."""
    if "://" not in endpoint:
        return endpoint, False
    split = urlsplit(endpoint)
    if split.scheme not in ("http", "https") or not split.netloc or "@" in split.netloc:
        raise GlinerRemoteConfigError(
            "Each gRPC COGNEE_GLINER_ENDPOINT must be http(s)://host:port or host:port, "
            f"got {redact_url(endpoint)!r}."
        )
    return split.netloc, split.scheme == "https"


def _mention(mention: Any) -> Any:
    """A typed Mention back into gliner2's shape: a plain string when it carries
    neither confidence nor span, else a dict with what it has."""
    if not mention.HasField("confidence") and not mention.HasField("start"):
        return mention.text
    result: dict[str, Any] = {"text": mention.text}
    if mention.HasField("confidence"):
        result["confidence"] = mention.confidence
    if mention.HasField("start"):
        result["start"] = mention.start
    if mention.HasField("end"):
        result["end"] = mention.end
    return result


def _relation(relation: Any) -> Any:
    head, tail = _mention(relation.head), _mention(relation.tail)
    if isinstance(head, str) and isinstance(tail, str):
        return [head, tail]
    return {"head": head, "tail": tail}


def _ordered(mapping: Mapping[str, Any], order: list[str]) -> list[str]:
    # Protobuf maps carry no order; gliner2 (and HTTP) report labels in schema order.
    return [label for label in order if label in mapping] + sorted(
        label for label in mapping if label not in order
    )


def response_to_reply(response: Any, entity_order: list[str], relation_order: list[str]):
    items = []
    for item in response.items:
        extraction: dict[str, Any] = {}
        if item.entities:
            extraction["entities"] = {
                label: [_mention(m) for m in item.entities[label].mentions]
                for label in _ordered(item.entities, entity_order)
            }
        if item.relations:
            extraction["relation_extraction"] = {
                label: [_relation(r) for r in item.relations[label].relations]
                for label in _ordered(item.relations, relation_order)
            }
        items.append((item.id, extraction))
    return WorkerReply(
        items=items,
        model=response.model,
        elapsed_ms=response.elapsed_ms,
        windowed=response.windowed,
    )


def payload_to_request(payload: Mapping[str, Any]) -> Any:
    messages = _messages()
    schema = payload["schema"]

    def labels(section: Mapping[str, str | None]) -> list[Any]:
        return [
            messages["Label"](name=name)
            if description is None
            else messages["Label"](name=name, description=description)
            for name, description in section.items()
        ]

    return messages["ExtractRequest"](
        inputs=[messages["Input"](id=item["id"], text=item["text"]) for item in payload["inputs"]],
        schema=messages["Schema"](
            entities=labels(schema.get("entities", {})),
            relations=labels(schema.get("relations", {})),
        ),
        options=messages["Options"](
            **{key: value for key, value in payload.get("options", {}).items() if value is not None}
        ),
    )


class GrpcWorkerTransport:
    name = "grpc"

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None,
        connect_timeout: float,
        request_timeout: float,
        max_response_bytes: int,
    ):
        endpoints = [part.strip() for part in endpoint.split(",") if part.strip()]
        if not endpoints:
            raise GlinerRemoteConfigError("COGNEE_GLINER_ENDPOINT lists no gRPC endpoint.")
        self.endpoint_label = redact_url(endpoint)
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._metadata = (("authorization", f"Bearer {api_key}"),) if api_key else ()
        options = [
            ("grpc.max_receive_message_length", max_response_bytes),
            ("grpc.max_send_message_length", -1),
            ("grpc.keepalive_time_ms", 30_000),
        ]
        self._channels = []
        for part in endpoints:
            target, secure = _target(part)
            self._channels.append(
                grpc.aio.secure_channel(target, grpc.ssl_channel_credentials(), options=options)
                if secure
                else grpc.aio.insecure_channel(target, options=options)
            )
        self._next = itertools.cycle(range(len(self._channels)))

    async def extract(self, payload: Mapping[str, Any], *, request_id: str) -> WorkerReply:
        messages = _messages()
        channel = self._channels[next(self._next)]
        call = channel.unary_unary(
            EXTRACT_METHOD,
            request_serializer=messages["ExtractRequest"].SerializeToString,
            response_deserializer=messages["ExtractResponse"].FromString,
        )
        try:
            response = await call(
                payload_to_request(payload),
                timeout=self._request_timeout,
                metadata=(*self._metadata, ("x-request-id", request_id)),
            )
        except grpc.aio.AioRpcError as error:
            raise self._error(error) from error
        return response_to_reply(
            response,
            list(payload["schema"].get("entities", {})),
            list(payload["schema"].get("relations", {})),
        )

    async def readiness(self) -> WorkerReadiness:
        """Ready when at least one endpoint's health service reports SERVING.

        The probe names no model and publishes no limits: the model is checked
        on the first reply, and windowing through each reply's ``windowed``.
        """
        messages = _messages()
        failure: Exception | None = None
        for channel in self._channels:
            check = channel.unary_unary(
                HEALTH_METHOD,
                request_serializer=messages["HealthCheckRequest"].SerializeToString,
                response_deserializer=messages["HealthCheckResponse"].FromString,
            )
            try:
                response = await check(
                    messages["HealthCheckRequest"](service=SERVICE_NAME),
                    timeout=self._connect_timeout,
                    metadata=self._metadata,
                )
            except grpc.aio.AioRpcError as error:
                failure = self._error(error)
                continue
            if response.status == messages["HealthCheckResponse"].SERVING:
                return WorkerReadiness()
            failure = GlinerWorkerUnavailableError(
                f"The GLiNER worker at {self.endpoint_label} is not serving yet."
            )
        assert failure is not None
        raise failure

    async def aclose(self) -> None:
        for channel in self._channels:
            await channel.close()

    def _error(self, error: Any) -> Exception:
        code = error.code()
        name = code.name if code is not None else "UNKNOWN"
        return classify_status(
            name,
            f"The GLiNER worker at {self.endpoint_label} answered {name}: {error.details() or ''}",
        )
