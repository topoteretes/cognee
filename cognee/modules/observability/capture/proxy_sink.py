"""Proxy sink (SDK-529): ship capture records through the analytics proxy.

Same road as product telemetry (``cognee.shared.utils.send_telemetry``): a POST
to the proxy on Vercel, which forwards to a Firehose stream, and a daily loader
copies the stream into MotherDuck. Nothing here talks to the warehouse and no
credential leaves the process — the proxy holds the AWS role.

Selected with ``COGNEE_CAPTURE_SINK=proxy`` (or ``both``, next to the storage
sink). Two switches have to be on for a record to leave the machine — capture
itself (``COGNEE_CAPTURE_ENABLED``) and this sink — and the telemetry kill
switches (``TELEMETRY_DISABLED``, ``ENV=test|dev``) block it like they block
telemetry (``proxy_sink_blocked``).

What leaves, and what does not
------------------------------
- One identity block per request: the ids telemetry sends (``anonymous_id``,
  ``persistent_id``, ``api_key_tracking_id``, ``telemetry_origin``,
  ``cognee_version``, ``sdk_runtime``). ``user_id`` / ``tenant_id`` ride on
  each record, from its run scope.
- Every record's envelope, ids included. ``run_id`` is the ``pipeline_run_id``
  telemetry already sends and ``dataset_id`` a dataset UUID — the rule
  telemetry applies to datasets is that ids pass through and user-chosen
  names are fingerprinted (``_mask_dataset_name``); capture never carries a
  name. The warehouse joins capture rows to telemetry rows on both.
- Metadata payloads as they are: ids, counts, hashes, scores, provider and
  model names — a model setting that is a filesystem path reduced to
  ``local_path`` exactly like telemetry does (``telemetry_model_label``).
- The content kinds (``CONTENT_KINDS``: chunk graphs, dropped duplicate ids,
  ontology lookups, retrieval candidates) leave as join keys, counts, scores
  and type histograms built from an allowlist (``metadata_only_payload``);
  the extracted strings — names, descriptions, relationship labels, matched
  terms — never do. The full payload leaves only with
  ``COGNEE_CAPTURE_PROXY_CONTENT=true``, and every record says which form it
  is (``content_included``), so the proxy can enforce its own policy on what
  it accepts.

Delivery follows the flusher's contract: a failed POST drops the batch and
counts it on the run (``events_dropped``); a POST cut off by a drain budget is
retried, and the warehouse dedupes on ``event_id``. Requests are capped at
``MAX_BATCH_BYTES`` (the proxy runs as a serverless function with a body
limit) and records at ``MAX_RECORD_BYTES`` (the Firehose record limit); an
oversize record is skipped and logged, never split.

The encode and the POST both run on a worker thread (``run_off_loop``): a sync
client, so the sink is usable from any loop as the sink contract requires, and
the flusher's ``SINK_TIMEOUT_S`` can abandon a stalled request.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cognee.shared.logging_utils import get_logger

from .events import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_DROPPED_DUPLICATES,
    KIND_EXTRACTION_FUZZY_MATCH,
    KIND_RETRIEVAL_CANDIDATES,
)
from .sinks import run_off_loop

if TYPE_CHECKING:
    from .config import CaptureConfig

logger = get_logger("eval_capture")

# The route on the telemetry proxy; the proxy writes it to its own stream.
CAPTURE_ROUTE = "/capture"
# Vercel serverless functions reject bodies past 4.5 MB; stay well under.
MAX_BATCH_BYTES = 3 * 1024 * 1024
# Firehose rejects a record past 1,000 KiB; the proxy drops those too.
MAX_RECORD_BYTES = 900 * 1024
# The kinds whose payload carries derived user content: something the LLM
# extracted from the user's documents. Chunk graphs (entity names, descriptions
# and relationship labels verbatim); the graph-local ids of dropped duplicate
# nodes (the extraction prompt asks for human-readable ids, so they are entity
# names); the names looked up against the ontology; and the relationship label
# on each entry of a retrieval cut. Every other kind is ids, hashes, counts,
# scores, provider and model names by construction — ``test_capture_proxy_sink``
# serializes one record of every kind with content off and checks that.
CONTENT_KINDS = frozenset(
    {
        KIND_EXTRACTION_CHUNK_GRAPH,
        KIND_EXTRACTION_DROPPED_DUPLICATES,
        KIND_EXTRACTION_FUZZY_MATCH,
        KIND_RETRIEVAL_CANDIDATES,
    }
)
# What survives of a content kind's payload with content off: an ALLOWLIST of
# join keys, counts and flags per kind, plus the histograms and reduced entries
# ``metadata_only_payload`` derives. A payload key not listed here never leaves,
# which is the right default for a key added to an emit point later.
METADATA_KEYS: dict[str, tuple[str, ...]] = {
    KIND_EXTRACTION_CHUNK_GRAPH: ("chunk_id", "chunk_index", "chunk_size_chars"),
    KIND_EXTRACTION_DROPPED_DUPLICATES: ("chunk_id", "chunk_index", "count"),
    KIND_EXTRACTION_FUZZY_MATCH: ("chunk_id", "chunk_index", "lookups", "count"),
    KIND_RETRIEVAL_CANDIDATES: ("query_index", "pool_size", "pool_truncated", "cut_size"),
}
# A retrieval pool entry: a vector-row id, its collection (a schema name such as
# ``Entity_name``) and the distance. A cut entry: the two node ids and the
# summed distance — never ``rel``, the relationship label.
POOL_ENTRY_KEYS = ("id", "collection", "score")
CUT_ENTRY_KEYS = ("source", "target", "score")
# Payload keys that hold a model setting; masked like telemetry's model labels.
MODEL_KEYS = frozenset(
    {"model", "llm.model", "embedding.model", "extraction.model", "summarization.model"}
)
SDK_RUNTIME = "python"

# (url, body, timeout_s) -> None; raises on any failure, including a non-2xx answer.
Poster = Callable[[str, bytes, float], None]


def proxy_sink_blocked() -> str | None:
    """The telemetry kill switches, applied to this sink: the reason, or ``None``."""
    if os.getenv("TELEMETRY_DISABLED"):
        return "TELEMETRY_DISABLED is set"
    env = os.getenv("ENV")
    if env in ("test", "dev"):
        return f"ENV={env}"
    return None


def default_proxy_url() -> str:
    """The telemetry proxy's capture route — one host for all outbound analytics."""
    from cognee.shared.utils import proxy_url

    return proxy_url.rstrip("/") + CAPTURE_ROUTE


def telemetry_identity() -> dict[str, Any]:
    """The identity block telemetry sends, minus user/tenant (those ride on records)."""
    from cognee.shared.utils import (
        _cognee_version,
        _get_api_key_tracking_id,
        get_anonymous_id,
        get_persistent_id,
        telemetry_origin,
    )

    return {
        "anonymous_id": get_anonymous_id(),
        "persistent_id": get_persistent_id(),
        "api_key_tracking_id": _get_api_key_tracking_id(),
        "telemetry_origin": telemetry_origin(),
        "cognee_version": _cognee_version(),
        "sdk_runtime": SDK_RUNTIME,
    }


def mask_model_labels(value: Any, key: str | None = None) -> Any:
    """A copy of ``value`` with every model setting under a ``MODEL_KEYS`` key
    passed through ``telemetry_model_label`` (a filesystem path -> ``local_path``).

    Handles the three shapes the payloads use: a flat ``{"model": ...}``
    (``summary.generated``), the dotted keys of a setup snapshot
    (``{"llm.model": ...}``) and a ``setup.changed`` diff entry
    (``{"llm.model": {"from": ..., "to": ...}}``).
    """
    from cognee.shared.utils import telemetry_model_label

    if isinstance(value, dict):
        if key in MODEL_KEYS and value and set(value) <= {"from", "to"}:
            return {edge: telemetry_model_label(label) for edge, label in value.items()}
        return {
            child_key: mask_model_labels(child, child_key) for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [mask_model_labels(item, key) for item in value]
    if key in MODEL_KEYS:
        return telemetry_model_label(value)
    return value


def metadata_only_payload(kind: str, payload: Any) -> dict[str, Any]:
    """The content-free form of a content kind's payload: join keys, counts, scores, histograms.

    Built from the allowlists above, never by removing keys from the input.
    ``extraction.chunk_graph`` keeps the chunk join keys and becomes node and
    edge counts plus a node-type histogram (the type labels are the
    extraction's categories — ``Person``, ``Organization`` — not document
    text); ``extraction.dropped_duplicates`` keeps the chunk join keys and the
    count, the dropped ids being entity names; ``extraction.fuzzy_match`` keeps
    the lookup and match counts plus a per-category histogram;
    ``retrieval.candidates`` keeps the pool as vector-row ids, collection names
    and scores and the cut as node ids and scores. Names, descriptions,
    relationship labels and matched terms never appear.
    """
    if not isinstance(payload, dict):
        return {}
    reduced: dict[str, Any] = {key: payload.get(key) for key in METADATA_KEYS.get(kind, ())}
    if kind == KIND_EXTRACTION_CHUNK_GRAPH:
        graph = payload.get("graph")
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        edges = graph.get("edges") if isinstance(graph, dict) else None
        reduced["node_count"] = len(nodes) if isinstance(nodes, list) else None
        reduced["edge_count"] = len(edges) if isinstance(edges, list) else None
        reduced["node_types"] = (
            dict(Counter(str(node.get("type")) for node in nodes if isinstance(node, dict)))
            if isinstance(nodes, list)
            else None
        )
    elif kind == KIND_EXTRACTION_FUZZY_MATCH:
        categories: dict[str, dict[str, int]] = {}
        matches = payload.get("matches")
        if isinstance(matches, list):
            for match in matches:
                if not isinstance(match, dict):
                    continue
                entry = categories.setdefault(
                    str(match.get("category")), {"lookups": 0, "matched": 0}
                )
                entry["lookups"] += 1
                if match.get("matched"):
                    entry["matched"] += 1
        reduced["categories"] = categories
    elif kind == KIND_RETRIEVAL_CANDIDATES:
        reduced["pool"] = _entries(payload.get("pool"), POOL_ENTRY_KEYS)
        reduced["top_k"] = _entries(payload.get("top_k"), CUT_ENTRY_KEYS)
    return reduced


def _entries(value: Any, keys: tuple[str, ...]) -> list[dict[str, Any]] | None:
    """The dict entries of a list reduced to ``keys``; ``None`` when ``value`` is no list."""
    if not isinstance(value, list):
        return None
    return [{key: entry.get(key) for key in keys} for entry in value if isinstance(entry, dict)]


def prepare_record(record: dict, *, include_content: bool) -> dict:
    """The record as it may leave the process; a new dict, the input untouched."""
    kind = record.get("kind")
    payload = record.get("payload")
    content_kind = kind in CONTENT_KINDS
    if content_kind and not include_content:
        payload = metadata_only_payload(kind, payload)
    else:
        payload = mask_model_labels(payload)
    return {**record, "content_included": content_kind and include_content, "payload": payload}


def post_json(url: str, body: bytes, timeout_s: float) -> None:
    """One POST; a non-2xx answer raises (``urllib`` raises ``HTTPError`` for it)."""
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "cognee-eval-capture"},
    )
    with urllib.request.urlopen(
        request, timeout=timeout_s, context=ssl.create_default_context()
    ) as response:
        status = response.status
    if not 200 <= status < 300:
        raise RuntimeError(f"capture proxy answered {status}")


class ProxySink:
    """POST capture records to the analytics proxy, in byte-capped batches.

    Body shape: ``{"identity": {...}, "records": [...]}`` — one identity block
    per request (see ``telemetry_identity``), each record prepared by
    ``prepare_record``. ``poster`` is the transport (``post_json``); tests pass
    a recorder.
    """

    def __init__(
        self,
        url: str,
        *,
        include_content: bool = False,
        timeout_s: float = 5.0,
        poster: Poster = post_json,
        max_batch_bytes: int = MAX_BATCH_BYTES,
        max_record_bytes: int = MAX_RECORD_BYTES,
    ) -> None:
        self.url = url
        self.include_content = include_content
        self.timeout_s = timeout_s
        self._poster = poster
        self._max_batch_bytes = max_batch_bytes
        self._max_record_bytes = max_record_bytes

    @classmethod
    def from_config(cls, config: CaptureConfig) -> ProxySink:
        return cls(
            config.cognee_capture_proxy_url or default_proxy_url(),
            include_content=config.cognee_capture_proxy_content,
            timeout_s=config.cognee_capture_proxy_timeout_s,
        )

    async def __call__(self, records: list[dict]) -> None:
        bodies = await run_off_loop(self.encode, records)
        for body in bodies:
            await run_off_loop(self._poster, self.url, body, self.timeout_s)

    def encode(self, records: list[dict]) -> list[bytes]:
        """Prepare and JSON-encode ``records`` into request bodies under the byte caps."""
        prefix = (
            b'{"identity":' + json.dumps(telemetry_identity(), default=str).encode("utf-8")
        ) + b',"records":['
        suffix = b"]}"
        frame = len(prefix) + len(suffix)

        lines: list[bytes] = []
        for record in records:
            try:
                line = json.dumps(
                    prepare_record(record, include_content=self.include_content), default=str
                ).encode("utf-8")
            except Exception as exc:
                logger.debug("capture proxy record skipped, not serializable (%s)", exc)
                continue
            if len(line) > self._max_record_bytes:
                logger.debug(
                    "capture proxy record skipped, %d bytes over the %d byte record limit",
                    len(line),
                    self._max_record_bytes,
                )
                continue
            lines.append(line)

        bodies: list[bytes] = []
        current: list[bytes] = []
        size = frame
        for line in lines:
            extra = len(line) + (1 if current else 0)
            if current and size + extra > self._max_batch_bytes:
                bodies.append(prefix + b",".join(current) + suffix)
                current, size, extra = [], frame, len(line)
            current.append(line)
            size += extra
        if current:
            bodies.append(prefix + b",".join(current) + suffix)
        return bodies
