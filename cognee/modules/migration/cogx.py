"""COGX — the Cognee eXchange format for portable memory.

COGX is the hub format for memory migration: importers translate external
providers (Mem0, Zep/Graphiti, Letta, ...) into COGX records, a single loader
ingests COGX into Cognee, and a single dumper exports Cognee datasets into a
COGX archive that emitters can translate onward into other formats.

An archive is a directory containing ``manifest.json`` plus one JSONL file per
record kind. Records are Pydantic models discriminated by their ``kind`` field.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator
from typing_extensions import Self

COGX_VERSION = "0.2"


def parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp from ISO strings or epoch seconds/milli/micro/nanoseconds.

    Always returns a timezone-aware UTC datetime, or None. Inputs that carry no
    offset — ``"2026-01-01T10:00:00"``, a bare date, a naive ``datetime`` — are
    read as UTC, which is what the exporting systems store. Returning them naive
    mixes naive and aware values across a single import: comparing two of them
    raises TypeError, and sorting them applies the importing machine's local
    offset instead of the real instant.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Heuristic: values past the year ~2603 in seconds are a finer epoch
        # unit (milli/micro/nanoseconds); scale down until plausible.
        seconds = float(value)
        if not math.isfinite(seconds):
            return None
        while abs(seconds) > 2e10:
            seconds /= 1000
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


class COGXScope(BaseModel):
    """Ownership scope of a memory record in the source system."""

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None


# ── OMP header (COGX 0.2) ──────────────────────────────────────────────────
# The fields below implement the header proposed to the Open Memory Protocol
# exchange-and-runtime group (common-ground notes, Oct 14 amendments): an
# origin kept across copies, a body hash for dedupe, the memory's subject,
# its semantic kind, a validity interval with a supersession pointer, typed
# evidence, and a measured confidence record. Every field is optional so 0.1
# archives and external importers keep working unchanged.

SEMANTIC_KINDS = (
    "fact",
    "decision",
    "commitment",
    "procedure",
    "preference",
    "observation",
    "other",
)
SemanticKind = Literal[
    "fact", "decision", "commitment", "procedure", "preference", "observation", "other"
]
EvidenceRole = Literal["observed", "stated", "inferred", "derived"]
ConfidenceOutcome = Literal["unverified", "confirmed", "corrected", "contradicted"]
Verifier = Literal["human", "agent", "automatic"]
ContributorRole = Literal["initiator", "author", "approver", "system"]
ActorKind = Literal["human", "agent", "unknown"]
AIInvolvement = Literal["none", "draft", "mixed", "full"]


def content_hash_of(text: str) -> str:
    """Hash of the body alone (``sha256:<hex>``): "do I already have this?".

    Distinct from any integrity checksum of the exported record, which answers
    "did it arrive intact?". Identical bodies hash identically regardless of
    header differences, which is what dedupe across copies needs.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class COGXOrigin(BaseModel):
    """Where a record first came from, kept across copies and re-exports.

    ``external_system``/``external_id`` describe the system that wrote *this*
    archive; ``origin`` names the first system, so a memory imported from Mem0
    into Cognee and exported again still says it came from Mem0.
    """

    system: str
    id: str
    version: str | None = None


class COGXEvidence(BaseModel):
    """One piece of support for a record.

    ``role`` says how the record relates to its support: ``observed`` (seen in a
    source), ``stated`` (asserted by an actor), ``inferred`` (a model's
    conclusion) or ``derived`` (consolidated from other memories). ``ref`` is
    another record's identifier, a source locator, or an actor. ``confidence``
    is the writer's belief at write time; see :class:`COGXConfidence` for what
    happened afterwards.
    """

    role: EvidenceRole
    ref: str
    confidence: float | None = None


class COGXConfidence(BaseModel):
    """How much a memory can be trusted, measured after it was written.

    ``outcome`` records whether a later check confirmed, corrected or
    contradicted it; ``verified_by`` says whether a human, an agent or an
    automatic rule did so; ``score`` is a value in [0, 1] on an
    implementation-defined scale. A bare number given where a record expects
    this object (the COGX 0.1 fact ``confidence`` float) becomes ``score``
    with the outcome left ``unverified``.
    """

    outcome: ConfidenceOutcome = "unverified"
    verified_at: datetime | None = None
    verified_by: Verifier | None = None
    score: float | None = None


class COGXContributor(BaseModel):
    """One entry of the provenance list: who took part, in what role.

    ``initiator`` asked for the write, ``author`` produced the body,
    ``approver`` signed it off, ``system`` is the pipeline or model that ran.
    The accountable human is named separately on the record so an auditor
    never has to infer it from this list.
    """

    role: ContributorRole
    actor: str
    kind: ActorKind = "unknown"
    at: datetime | None = None


def checksum_of(record: BaseModel) -> str:
    """Integrity hash of the exported record, header and body, minus the
    checksum itself: "did it arrive intact?". Not the body hash."""
    payload = record.model_dump_json(exclude={"checksum"}, exclude_none=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_checksum(record: BaseModel) -> bool | None:
    """True/False when the record carries a checksum; None when it does not."""
    stored = getattr(record, "checksum", None)
    if not stored:
        return None
    return checksum_of(record) == stored


class COGXRecordBase(BaseModel):
    external_system: str = "unknown"
    external_id: str
    scope: COGXScope = Field(default_factory=COGXScope)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # OMP header (0.2). All optional.
    origin: COGXOrigin | None = None
    version: int | None = None
    content_type: str | None = None  # body format, e.g. text/plain, text/markdown
    content_hash: str | None = None
    checksum: str | None = None  # stamped by the writer over the whole record
    tags: list[str] = Field(default_factory=list)
    contributors: list[COGXContributor] = Field(default_factory=list)
    accountable: str | None = None  # the one human an auditor needs
    ai_involvement: AIInvolvement | None = None
    subject: list[str] = Field(default_factory=list)
    semantic_kind: SemanticKind | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    invalidated_by: str | None = None
    evidence: list[COGXEvidence] = Field(default_factory=list)
    confidence: COGXConfidence | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_from_number(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"score": float(value)}
        return value

    def body_text(self) -> str | None:
        """The text the content hash covers; None for records without a body."""
        return None

    @model_validator(mode="after")
    def _stamp_body_fields(self) -> Self:
        body = self.body_text()
        if self.content_hash is None and body:
            self.content_hash = content_hash_of(body)
        if self.content_type is None and body:
            self.content_type = "text/plain"
        return self


class COGXDocument(COGXRecordBase):
    """Raw source content: a file, archival passage, or standalone text."""

    kind: Literal["document"] = "document"
    content: str
    title: str | None = None
    mime_type: str | None = None

    def body_text(self) -> str | None:
        return self.content

    @model_validator(mode="after")
    def _content_type_from_mime(self) -> Self:
        if self.content_type in (None, "text/plain") and self.mime_type:
            self.content_type = self.mime_type
        return self


class COGXTurn(BaseModel):
    role: str
    content: str
    occurred_at: datetime | None = None


class COGXEpisode(COGXRecordBase):
    """A conversation episode: ordered turns with roles and timestamps."""

    kind: Literal["episode"] = "episode"
    turns: list[COGXTurn] = Field(default_factory=list)
    title: str | None = None

    def body_text(self) -> str | None:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self.turns) or None


class COGXEntity(COGXRecordBase):
    """An extracted entity with optional type, aliases, and description."""

    kind: Literal["entity"] = "entity"
    name: str
    entity_type: str | None = None
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    def body_text(self) -> str | None:
        return f"{self.name}: {self.description}" if self.description else self.name


class COGXFact(COGXRecordBase):
    """A triplet fact. Subject/object refer to entity external_ids or plain names.

    Temporal validity (``valid_at``/``invalid_at``) is always carried, even
    where the loader cannot yet query it natively — it is stored as edge
    properties so the data is in place when bi-temporal search lands.
    """

    kind: Literal["fact"] = "fact"
    subject_ref: str
    predicate: str
    object_ref: str
    fact_text: str | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    provenance: list[str] = Field(default_factory=list)

    def body_text(self) -> str | None:
        return self.fact_text

    @model_validator(mode="after")
    def _sync_validity(self) -> Self:
        """``valid_at``/``invalid_at`` (0.1) and ``valid_from``/``valid_to`` (0.2)
        are the same interval; whichever side is set fills the other."""
        if self.valid_from is None and self.valid_at is not None:
            self.valid_from = self.valid_at
        elif self.valid_at is None and self.valid_from is not None:
            self.valid_at = self.valid_from
        if self.valid_to is None and self.invalid_at is not None:
            self.valid_to = self.invalid_at
        elif self.invalid_at is None and self.valid_to is not None:
            self.invalid_at = self.valid_to
        return self


class COGXMemory(COGXRecordBase):
    """An atomic derived memory (Mem0-style short fact text)."""

    kind: Literal["memory"] = "memory"
    content: str
    categories: list[str] = Field(default_factory=list)

    def body_text(self) -> str | None:
        return self.content


class COGXMemoryBlock(COGXRecordBase):
    """A named, bounded core-memory block (Letta-style)."""

    kind: Literal["memory_block"] = "memory_block"
    label: str
    value: str
    limit: int | None = None

    def body_text(self) -> str | None:
        return self.value


class COGXRawNode(BaseModel):
    """A graph node persisted verbatim (full fidelity) when no typed mapping exists.

    ``properties`` is the raw node property dict, including its ``id`` and
    ``type``. On import these are rehydrated back into DataPoint instances so
    facts referencing them stay resolvable.
    """

    kind: Literal["raw_node"] = "raw_node"
    properties: dict[str, Any] = Field(default_factory=dict)


COGXRecord = (
    COGXDocument | COGXEpisode | COGXEntity | COGXFact | COGXMemory | COGXMemoryBlock | COGXRawNode
)

HEADER_FIELDS = (
    "origin",
    "version",
    "content_type",
    "content_hash",
    "checksum",
    "tags",
    "contributors",
    "accountable",
    "ai_involvement",
    "subject",
    "semantic_kind",
    "valid_from",
    "valid_to",
    "invalidated_by",
    "evidence",
    "confidence",
)


def header_to_properties(record: COGXRecordBase) -> dict[str, Any]:
    """The OMP header as flat, JSON-safe node/edge properties.

    Graph backends store scalars; nested values (origin, evidence, contributors)
    are serialized as JSON strings and lists as JSON arrays so the header
    survives a round trip through the graph and comes back on re-export.
    """
    properties: dict[str, Any] = {}
    if record.origin is not None:
        properties["origin"] = json.dumps(record.origin.model_dump(exclude_none=True))
    if record.version is not None:
        properties["version"] = record.version
    if record.content_type:
        properties["content_type"] = record.content_type
    if record.content_hash:
        properties["content_hash"] = record.content_hash
    if record.tags:
        properties["tags"] = json.dumps(record.tags)
    if record.contributors:
        properties["contributors"] = json.dumps(
            [item.model_dump(mode="json", exclude_none=True) for item in record.contributors]
        )
    if record.accountable:
        properties["accountable"] = record.accountable
    if record.ai_involvement:
        properties["ai_involvement"] = record.ai_involvement
    if record.subject:
        properties["subject"] = json.dumps(record.subject)
    if record.semantic_kind:
        properties["semantic_kind"] = record.semantic_kind
    if record.valid_from:
        properties["valid_from"] = record.valid_from.isoformat()
    if record.valid_to:
        properties["valid_to"] = record.valid_to.isoformat()
    if record.invalidated_by:
        properties["invalidated_by"] = record.invalidated_by
    if record.evidence:
        properties["evidence"] = json.dumps(
            [item.model_dump(exclude_none=True) for item in record.evidence]
        )
    if record.confidence is not None:
        # Kept scalar so the edge's ``confidence`` stays the numeric property
        # retrieval already reads; the measured part rides beside it.
        if record.confidence.score is not None:
            properties["confidence"] = record.confidence.score
        if record.confidence.outcome != "unverified":
            properties["confidence_outcome"] = record.confidence.outcome
        if record.confidence.verified_at:
            properties["confidence_verified_at"] = record.confidence.verified_at.isoformat()
        if record.confidence.verified_by:
            properties["confidence_verified_by"] = record.confidence.verified_by
    return properties


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def header_from_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Read the OMP header back from node/edge properties (inverse of
    :func:`header_to_properties`). Unknown or malformed values are dropped,
    never guessed: an absent header field means "not stated"."""
    header: dict[str, Any] = {}
    origin = _loads(properties.get("origin"))
    if isinstance(origin, dict) and origin.get("system") and origin.get("id"):
        header["origin"] = COGXOrigin(
            system=str(origin["system"]), id=str(origin["id"]), version=origin.get("version")
        )
    version = properties.get("version")
    if isinstance(version, int) and not isinstance(version, bool):
        header["version"] = version
    if properties.get("content_type"):
        header["content_type"] = str(properties["content_type"])
    if properties.get("content_hash"):
        header["content_hash"] = str(properties["content_hash"])
    tags = _loads(properties.get("tags"))
    if isinstance(tags, list):
        header["tags"] = [str(item) for item in tags]
    contributors = _loads(properties.get("contributors"))
    if isinstance(contributors, list):
        items = []
        for item in contributors:
            if isinstance(item, dict) and item.get("role") and item.get("actor"):
                try:
                    items.append(COGXContributor.model_validate(item))
                except ValueError:
                    continue
        if items:
            header["contributors"] = items
    if properties.get("accountable"):
        header["accountable"] = str(properties["accountable"])
    if properties.get("ai_involvement") in ("none", "draft", "mixed", "full"):
        header["ai_involvement"] = properties["ai_involvement"]
    subject = _loads(properties.get("subject"))
    if isinstance(subject, list):
        header["subject"] = [str(item) for item in subject]
    kind = properties.get("semantic_kind") or properties.get("lesson_kind")
    if kind in SEMANTIC_KINDS:
        header["semantic_kind"] = kind
    for key in ("valid_from", "valid_to"):
        parsed = parse_timestamp(properties.get(key))
        if parsed is not None:
            header[key] = parsed
    invalidated_by = properties.get("invalidated_by") or properties.get("superseded_by")
    if invalidated_by:
        header["invalidated_by"] = str(invalidated_by)
    evidence = _loads(properties.get("evidence"))
    if isinstance(evidence, list):
        items = []
        for item in evidence:
            if isinstance(item, dict) and item.get("role") and item.get("ref"):
                try:
                    items.append(COGXEvidence.model_validate(item))
                except ValueError:
                    continue
        if items:
            header["evidence"] = items
    confidence: dict[str, Any] = {}
    raw_score = _loads(properties.get("confidence"))
    if isinstance(raw_score, dict):
        confidence.update(raw_score)
    elif isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        confidence["score"] = float(raw_score)
    if properties.get("confidence_outcome") in (
        "unverified",
        "confirmed",
        "corrected",
        "contradicted",
    ):
        confidence["outcome"] = properties["confidence_outcome"]
    verified_at = parse_timestamp(properties.get("confidence_verified_at"))
    if verified_at is not None:
        confidence["verified_at"] = verified_at
    if properties.get("confidence_verified_by") in ("human", "agent", "automatic"):
        confidence["verified_by"] = properties["confidence_verified_by"]
    if confidence:
        try:
            header["confidence"] = COGXConfidence.model_validate(confidence)
        except ValueError:
            pass
    return header


_record_adapter: TypeAdapter = TypeAdapter(COGXRecord)

RECORD_FILES: dict[str, str] = {
    "document": "documents.jsonl",
    "episode": "episodes.jsonl",
    "entity": "entities.jsonl",
    "fact": "facts.jsonl",
    "memory": "memories.jsonl",
    "memory_block": "memory_blocks.jsonl",
}

MANIFEST_FILE = "manifest.json"
RAW_NODES_FILE = "nodes.jsonl"
PERMISSIONS_FILE = "permissions.json"


class COGXManifest(BaseModel):
    cogx_version: str = COGX_VERSION
    source_system: str = "unknown"
    exported_at: datetime | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    embedding_model: str | None = None
    # Cognee-origin archives only: the source store's stamped data-migration
    # revision at export time. Raw nodes keep their source-store ids, so the
    # importing store must not claim a newer revision than the exported data
    # actually has (None = unknown; the import keeps its own stamp).
    migration_revision: str | None = None
    notes: list[str] = Field(default_factory=list)
    # COGX 0.2: what the exporter could state for each OMP header field, so an
    # importer reads absence as "not populated" rather than "not true".
    coverage: dict[str, str] = Field(default_factory=dict)


def parse_record(data: dict[str, Any]) -> COGXRecord:
    return _record_adapter.validate_python(data)


def validate_cogx_version(version: str) -> None:
    """Reject archives written by a newer major COGX version than this reader."""
    try:
        archive_major = int(str(version).split(".")[0])
        current_major = int(COGX_VERSION.split(".")[0])
    except (ValueError, IndexError):
        raise ValueError(f"Unrecognized COGX version {version!r} (reader supports {COGX_VERSION}).")
    if archive_major > current_major:
        raise ValueError(
            f"COGX archive version {version} is newer than this reader "
            f"supports ({COGX_VERSION}). Upgrade cognee to import it."
        )


class COGXArchiveWriter:
    """Writes COGX records into an archive directory, one JSONL file per kind.

    The destination is treated as owned by the writer: any record files or
    manifest left by a previous export session are removed at init so a
    re-export never appends duplicates next to a stale manifest.
    """

    def __init__(self, directory: str | Path, source_system: str = "cognee"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for file_name in (*RECORD_FILES.values(), RAW_NODES_FILE, MANIFEST_FILE, PERMISSIONS_FILE):
            (self.directory / file_name).unlink(missing_ok=True)
        self.source_system = source_system
        self.counts: dict[str, int] = {}
        self.notes: list[str] = []
        self.embedding_model: str | None = None
        self.migration_revision: str | None = None
        self.coverage: dict[str, str] = {}
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(write_manifest=exc_type is None)

    def write(self, record: COGXRecord) -> None:
        file_name = RECORD_FILES[record.kind]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = open(self.directory / file_name, "a", encoding="utf-8")  # noqa: SIM115 - kept in self._handles, closed by __exit__
            self._handles[file_name] = handle
        record.checksum = checksum_of(record)
        handle.write(record.model_dump_json(exclude_none=True) + "\n")
        self.counts[record.kind] = self.counts.get(record.kind, 0) + 1

    def write_raw_node(self, node: dict[str, Any]) -> None:
        """Persist a graph node that has no typed COGX mapping (full fidelity)."""
        handle = self._handles.get(RAW_NODES_FILE)
        if handle is None:
            handle = open(self.directory / RAW_NODES_FILE, "a", encoding="utf-8")  # noqa: SIM115 - kept in self._handles, closed by __exit__
            self._handles[RAW_NODES_FILE] = handle
        handle.write(json.dumps(node, default=str) + "\n")
        self.counts["raw_node"] = self.counts.get("raw_node", 0) + 1

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def close(self, write_manifest: bool = True) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        if write_manifest:
            manifest = COGXManifest(
                source_system=self.source_system,
                exported_at=datetime.now(timezone.utc),
                counts=self.counts,
                embedding_model=self.embedding_model,
                migration_revision=self.migration_revision,
                notes=self.notes,
                coverage=self.coverage,
            )
            manifest_path = self.directory / MANIFEST_FILE
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_manifest(directory: str | Path) -> COGXManifest | None:
    manifest_path = Path(directory) / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    manifest = COGXManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    validate_cogx_version(manifest.cogx_version)
    return manifest


def write_social_layer(directory: str | Path, payload: dict[str, Any]) -> None:
    """Persist the dataset's social layer (owner + grants, WITH credentials).

    Written only on explicit request (``export(include_permissions=True)``):
    the payload carries user emails, password HASHES, and account flags, so an
    archive containing this file must be handled as a secret, not just data.
    """
    (Path(directory) / PERMISSIONS_FILE).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def read_social_layer(directory: str | Path) -> dict[str, Any] | None:
    permissions_path = Path(directory) / PERMISSIONS_FILE
    if not permissions_path.exists():
        return None
    return json.loads(permissions_path.read_text(encoding="utf-8"))


def _archive_file(base: Path, file_name: str) -> Path | None:
    """Resolve ``base / file_name`` and confirm it stays inside ``base``.

    Archive record file names are fixed constants, but the archive directory
    itself is caller-supplied; resolving and containment-checking the joined
    path keeps a crafted base from steering reads outside its own tree
    (CodeQL py/path-injection). Returns ``None`` when the file is absent or
    would resolve outside ``base``.
    """
    resolved_base = base.resolve()
    candidate = (resolved_base / file_name).resolve()
    if not candidate.is_relative_to(resolved_base):
        return None
    if not candidate.is_file():
        return None
    return candidate


def read_archive(directory: str | Path) -> Iterator[COGXRecord]:
    """Stream typed records from a COGX archive directory.

    Raw graph nodes (``nodes.jsonl``) are yielded as :class:`COGXRawNode`
    records after the typed record files, so importers can restore them with
    full fidelity.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"COGX archive directory not found: {directory}")
    for file_name in RECORD_FILES.values():
        file_path = _archive_file(directory, file_name)
        if file_path is None:
            continue
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield parse_record(json.loads(line))
    raw_nodes_path = _archive_file(directory, RAW_NODES_FILE)
    if raw_nodes_path is not None:
        with open(raw_nodes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield COGXRawNode(properties=json.loads(line))
