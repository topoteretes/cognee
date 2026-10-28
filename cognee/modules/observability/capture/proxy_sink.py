"""Proxy sink (SDK-529): ship capture records through the analytics proxy.

Same road as product telemetry (``cognee.shared.utils.send_telemetry``): a POST
to the proxy on Vercel, which forwards to a Firehose stream, and a daily loader
copies the stream into MotherDuck. Nothing here talks to the warehouse and no
credential leaves the process — the proxy holds the AWS role.

Selected with ``COGNEE_CAPTURE_SINK=proxy`` (or ``both``, next to the storage
sink). Two switches have to be on for a record to leave the machine — capture
itself (``COGNEE_CAPTURE_ENABLED``) and this sink — and the telemetry kill
switches (``TELEMETRY_DISABLED``, ``ENV=test|dev``) block it like they block
telemetry (``proxy_sink_blocked``): when the sink is selected, and again on
every request, so ``TELEMETRY_DISABLED=1`` is the one switch that promises
nothing leaves the machine, whenever it is set.

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
  ``local_path`` exactly like telemetry does (``telemetry_model_label``) —
  except graph and chunk ids, which are deterministic hashes of names and
  text and leave as keyed pseudonyms under a key this installation never
  sends (``GRAPH_ID_KEYS``, ``graph_id_pseudonym``).
- The content kinds (``CONTENT_KINDS``: chunk graphs, dropped duplicate ids,
  ontology lookups, retrieval candidates) leave as join keys, counts, scores
  and type histograms built from an allowlist (``metadata_only_payload``);
  the extracted strings — names, descriptions, relationship labels, matched
  terms — never do. The full payload leaves only with
  ``COGNEE_CAPTURE_PROXY_CONTENT=true``, and every record says which form it
  is (``content_included``), so the proxy can enforce its own policy on what
  it accepts.

Delivery follows the flusher's contract. A batch becomes one or more requests
(capped at ``MAX_BATCH_BYTES``: the proxy runs as a serverless function with a
body limit); a failed POST loses the records it carried and the requests after
it, while the ones before it landed — the sink reports exactly that through
``PartialDelivery`` so the run's ``events_dropped`` / ``events_delivered`` say
what reached the proxy, and a POST that fails first charges the whole batch. A
record over ``MAX_RECORD_BYTES`` (the Firehose record limit), or one that
cannot be serialized, is left out — never split — and charged to its run the
same way. A POST cut off by a drain budget is retried, and the warehouse
dedupes on ``event_id``.

The encode and the POST both run on a worker thread (``run_off_loop``): a sync
client, so the sink is usable from any loop as the sink contract requires, and
the flusher's ``SINK_TIMEOUT_S`` can abandon a stalled request.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import ssl
import threading
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cognee.shared.logging_utils import get_logger

from .events import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_DROPPED_DUPLICATES,
    KIND_EXTRACTION_FUZZY_MATCH,
    KIND_RETRIEVAL_CANDIDATES,
    KIND_STORAGE_DELTA,
    KIND_SUMMARY_GENERATED,
)
from .sinks import PartialDelivery, run_off_loop

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
# Graph and chunk ids are deterministic — an entity's id is
# ``uuid5(NAMESPACE_OID, "Entity:<normalized name>")`` under a public
# namespace, a chunk's ``uuid5(doc_id:sha256(text):occurrence)`` — so whoever
# holds a candidate name, or the document, can compute the id and confirm it
# appears in a record: for a person's name, a dictionary attack on what is
# meant to be metadata. Before such an id leaves the machine it is keyed-hashed
# with a secret this installation keeps to itself (``graph_id_pseudonym``). The
# same id keeps the same pseudonym across records and runs of the
# installation, so the joins the warehouse makes — pool against cut, chunk
# graph against its summary — still work, and nobody without the key can go
# from a name to a row. Run, dataset, user and tenant ids are random UUIDs
# telemetry already carries and leave as they are: they are what capture rows
# are joined to telemetry rows on, and what an erasure request is resolved by.
GRAPH_ID_KEYS: dict[str, tuple[str, ...]] = {
    KIND_EXTRACTION_CHUNK_GRAPH: ("chunk_id",),
    KIND_EXTRACTION_DROPPED_DUPLICATES: ("chunk_id",),
    KIND_EXTRACTION_FUZZY_MATCH: ("chunk_id",),
    KIND_SUMMARY_GENERATED: ("chunk_id", "summary_id"),
    KIND_STORAGE_DELTA: ("node_ids",),
}
# The id fields of a retrieval pool entry and of a cut entry.
POOL_ID_KEYS = ("id",)
CUT_ID_KEYS = ("source", "target")
# Where the installation's key lives (next to telemetry's persistent id) and
# the variable that overrides it for several machines sharing one pseudonym space.
ID_KEY_FILE_NAME = ".capture_id_key"
ID_KEY_ENV = "COGNEE_CAPTURE_ID_KEY"
# 128 bits of the HMAC-SHA256, as hex.
PSEUDONYM_HEX_CHARS = 32
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


_id_key_lock = threading.Lock()
_id_key: bytes | None = None


def installation_id_key() -> bytes:
    """The secret graph ids are keyed-hashed with before they leave. Never sent.

    ``COGNEE_CAPTURE_ID_KEY`` when set (any string: a deployment spanning
    several machines that wants one pseudonym space). Otherwise
    ``~/.cognee/.capture_id_key``, next to telemetry's persistent id: 32 random
    bytes written once, readable by the owner only, and read back by every
    later process, so pseudonyms are stable across runs. A home directory that
    cannot be read or written leaves a key for this process alone — pseudonyms
    then join within the process, not across it — logged at debug. Resolved
    once per process.
    """
    global _id_key
    key = _id_key
    if key is None:
        with _id_key_lock:
            key = _id_key
            if key is None:
                key = _id_key = _load_or_create_id_key()
    return key


def _load_or_create_id_key() -> bytes:
    configured = os.getenv(ID_KEY_ENV)
    if configured:
        return configured.encode("utf-8")
    from cognee.shared.utils import _PERSISTENT_ID_DIR

    path = _PERSISTENT_ID_DIR / ID_KEY_FILE_NAME
    for _attempt in range(2):
        try:
            if path.is_file():
                stored = path.read_text(encoding="utf-8").strip()
                if stored:
                    return stored.encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create, owner-only: two processes racing here leave one
            # file, and the loser reads it on its second pass.
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            key = secrets.token_hex(32)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(key)
            return key.encode("utf-8")
        except FileExistsError:
            continue
        except OSError as exc:
            logger.debug(
                "capture id key not persisted, pseudonyms hold within this process (%s)", exc
            )
            break
    return secrets.token_bytes(32)


def _reset_id_key_for_tests() -> None:
    global _id_key
    _id_key = None


def graph_id_pseudonym(value: Any, key: bytes) -> Any:
    """``value`` keyed-hashed under ``key``: ``PSEUDONYM_HEX_CHARS`` hex chars.

    ``None`` stays ``None`` and a list is mapped element-wise; anything else is
    hashed as its ``str()``, so a ``UUID`` and its string form agree.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [graph_id_pseudonym(item, key) for item in value]
    digest = hmac.new(key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:PSEUDONYM_HEX_CHARS]


def pseudonymize_graph_ids(kind: str, payload: Any, key: bytes) -> Any:
    """A copy of ``payload`` with its graph ids replaced by their pseudonyms.

    The fields are per kind (``GRAPH_ID_KEYS``; the entries' id fields for
    ``retrieval.candidates``). A field absent from the payload stays absent,
    and a payload that is not a dict is returned as is.
    """
    if not isinstance(payload, dict):
        return payload
    id_keys = GRAPH_ID_KEYS.get(kind, ())
    if not id_keys and kind != KIND_RETRIEVAL_CANDIDATES:
        return payload
    out = dict(payload)
    for name in id_keys:
        if name in out:
            out[name] = graph_id_pseudonym(out[name], key)
    if kind == KIND_RETRIEVAL_CANDIDATES:
        for name, entry_keys in (("pool", POOL_ID_KEYS), ("top_k", CUT_ID_KEYS)):
            if name in out:
                out[name] = _pseudonymize_entries(out[name], entry_keys, key)
    return out


def _pseudonymize_entries(value: Any, keys: tuple[str, ...], key: bytes) -> Any:
    if not isinstance(value, list):
        return value
    return [
        {**entry, **{name: graph_id_pseudonym(entry[name], key) for name in keys if name in entry}}
        if isinstance(entry, dict)
        else entry
        for entry in value
    ]


def prepare_record(record: dict, *, include_content: bool, id_key: bytes | None = None) -> dict:
    """The record as it may leave the process; a new dict, the input untouched.

    ``id_key`` is what graph ids are pseudonymized under (``GRAPH_ID_KEYS``);
    ``None`` means this installation's key.
    """
    kind = record.get("kind")
    payload = record.get("payload")
    content_kind = kind in CONTENT_KINDS
    if content_kind and not include_content:
        payload = metadata_only_payload(kind, payload)
    else:
        payload = mask_model_labels(payload)
    key = installation_id_key() if id_key is None else id_key
    payload = pseudonymize_graph_ids(kind, payload, key)
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


@dataclass(slots=True)
class EncodedRequest:
    body: bytes
    # Positions, in the records handed to the sink, of the records this body carries.
    positions: list[int]


@dataclass(slots=True)
class Encoded:
    """``ProxySink.encode``'s result: the requests to send, and what was left out."""

    requests: list[EncodedRequest]
    # Positions left out: not serializable, or over the per-record byte cap.
    skipped: list[int]


class ProxySink:
    """POST capture records to the analytics proxy, in byte-capped batches.

    Body shape: ``{"identity": {...}, "records": [...]}`` — one identity block
    per request (see ``telemetry_identity``), each record prepared by
    ``prepare_record``. ``poster`` is the transport (``post_json``); tests pass
    a recorder. What the sink could not deliver — a record ``encode`` left out,
    the requests from a failed POST on — is reported as ``PartialDelivery``
    (see the module docstring), never dropped silently.
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
        id_key: bytes | None = None,
    ) -> None:
        self.url = url
        self.include_content = include_content
        self.timeout_s = timeout_s
        self._poster = poster
        self._max_batch_bytes = max_batch_bytes
        self._max_record_bytes = max_record_bytes
        # The pseudonym key; None resolves the installation's at encode time.
        self._id_key = id_key

    @classmethod
    def from_config(cls, config: CaptureConfig) -> ProxySink:
        return cls(
            config.cognee_capture_proxy_url or default_proxy_url(),
            include_content=config.cognee_capture_proxy_content,
            timeout_s=config.cognee_capture_proxy_timeout_s,
        )

    async def __call__(self, records: list[dict]) -> None:
        blocked = proxy_sink_blocked()
        if blocked:
            # Checked per request, not only when the sink was selected: the
            # switch that promises nothing leaves holds even when set mid-process.
            raise RuntimeError(f"capture proxy sink blocked ({blocked})")
        encoded = await run_off_loop(self.encode, records)
        for sent, request in enumerate(encoded.requests):
            try:
                await run_off_loop(self._poster, self.url, request.body, self.timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if sent == 0:
                    raise  # nothing landed: the whole batch is the flusher's to charge
                # The requests before this one landed; this one and the rest did not.
                unsent = [p for later in encoded.requests[sent:] for p in later.positions]
                raise PartialDelivery(
                    [*encoded.skipped, *unsent],
                    f"capture proxy POST {sent + 1} of {len(encoded.requests)} failed ({exc})",
                ) from exc
        if encoded.skipped:
            raise PartialDelivery(
                encoded.skipped, f"{len(encoded.skipped)} record(s) left out of the request"
            )

    def encode(self, records: list[dict]) -> Encoded:
        """Prepare and JSON-encode ``records`` into request bodies under the byte caps.

        A record that cannot be serialized, or that is over ``max_record_bytes``
        once prepared, is left out — never split — and reported in
        ``Encoded.skipped``, so ``__call__`` can charge it to its run.
        """
        prefix = (
            b'{"identity":' + json.dumps(telemetry_identity(), default=str).encode("utf-8")
        ) + b',"records":['
        suffix = b"]}"
        frame = len(prefix) + len(suffix)

        id_key = installation_id_key() if self._id_key is None else self._id_key
        lines: list[tuple[int, bytes]] = []
        skipped: list[int] = []
        for position, record in enumerate(records):
            try:
                line = json.dumps(
                    prepare_record(record, include_content=self.include_content, id_key=id_key),
                    default=str,
                ).encode("utf-8")
            except Exception as exc:
                logger.debug("capture proxy record skipped, not serializable (%s)", exc)
                skipped.append(position)
                continue
            if len(line) > self._max_record_bytes:
                logger.debug(
                    "capture proxy record skipped, %d bytes over the %d byte record limit",
                    len(line),
                    self._max_record_bytes,
                )
                skipped.append(position)
                continue
            lines.append((position, line))

        requests: list[EncodedRequest] = []
        current: list[bytes] = []
        positions: list[int] = []
        size = frame
        for position, line in lines:
            extra = len(line) + (1 if current else 0)
            if current and size + extra > self._max_batch_bytes:
                requests.append(EncodedRequest(prefix + b",".join(current) + suffix, positions))
                current, positions, size, extra = [], [], frame, len(line)
            current.append(line)
            positions.append(position)
            size += extra
        if current:
            requests.append(EncodedRequest(prefix + b",".join(current) + suffix, positions))
        return Encoded(requests, skipped)
