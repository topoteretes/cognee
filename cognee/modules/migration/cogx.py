"""COGX — the Cognee eXchange format for portable memory.

COGX is the hub format for memory migration: importers translate external
providers (Mem0, Zep/Graphiti, Letta, ...) into COGX records, a single loader
ingests COGX into Cognee, and a single dumper exports Cognee datasets into a
COGX archive that emitters can translate onward into other formats.

An archive is a directory containing ``manifest.json`` plus one JSONL file per
record kind. Records are Pydantic models discriminated by their ``kind`` field.

Since COGX 0.3 every record carries the metadata of an Open Memory Protocol
(OMP) memory object — editors' draft v0.6 (2026-11-09): the field table of
its section 1.3, the contributor list of 3.1.1, the evidence entries of 3.1.4
and the link edges of 3.2 — under the draft's own field names, plus the
Cognee extensions the draft lists in 5.1.2. :mod:`cognee.modules.migration.omp`
renders a record as an OMP memory and reads one back. Archives written by
COGX 0.1 and 0.2 read unchanged: their field names are mapped on read.
"""

import hashlib
import json
import math
import re
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    PlainSerializer,
    TypeAdapter,
    field_validator,
    model_validator,
)
from typing_extensions import Self

COGX_VERSION = "0.3"
# The OMP editors' draft the record metadata follows (its optional
# ``protocol_version`` field, and the manifest's required one).
OMP_PROTOCOL_VERSION = "0.6"
# Record checksums are computed over the canonical JSON form since this COGX
# version; earlier archives hashed a field order this reader no longer
# reproduces, so their checksums are not verified.
CANONICAL_CHECKSUM_SINCE = (0, 3)


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
        return _assume_utc(value)
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
        parsed = _read_timestamp(value)  # the one text parser models use too
        return _assume_utc(parsed) if isinstance(parsed, datetime) else None
    return None


def _read_timestamp(value: Any) -> Any:
    """Text read as YAML reads a timestamp, which takes spellings ISO 8601
    parsers do not (``2001-12-14 21:59:43.10 -5``); a bare date is its
    midnight, UTC. Anything YAML reads no timestamp in is left as given. ISO
    text, the common case, is parsed directly: YAML's parser is far slower."""
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        parsed = yaml.safe_load(value)
    except (yaml.YAMLError, ValueError):  # ValueError: a date that cannot exist (month 13)
        return value
    if isinstance(parsed, datetime):
        return parsed
    if isinstance(parsed, date):
        return datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
    return value


def _assume_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def format_timestamp(value: datetime) -> str:
    """The one spelling of a timestamp: UTC, six fractional digits, ``Z``."""
    return _assume_utc(value).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# A model timestamp: text is read the YAML way first; one with no offset is read
# as UTC, as parse_timestamp reads it, so aware and naive values never meet in
# one comparison; JSON writes it as format_timestamp spells it.
UTCDateTime = Annotated[
    datetime,
    BeforeValidator(_read_timestamp),
    AfterValidator(_assume_utc),
    PlainSerializer(format_timestamp, when_used="json"),
]


class COGXScope(BaseModel):
    """Ownership scope of a memory record in the source system."""

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None


# ── OMP memory-object metadata (COGX 0.3, OMP draft v0.6) ──────────────────
# Vocabularies. Where the draft gives examples rather than a closed list
# (``contribution``, link ``rel``), any string is accepted and the examples
# are listed for reference.

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
# OMP 3.1.4: how a memory relates to its evidence.
EvidenceRelation = Literal["observed", "stated", "inferred", "derived"]
ConfidenceOutcome = Literal["unverified", "confirmed", "corrected", "contradicted"]
Verifier = Literal["human", "agent", "automatic"]
# OMP 3.1.1 ``kind``: human, agent or system. ``unknown`` is the COGX 0.2
# value for a contributor whose kind was never stated; it is kept on read and
# is not an OMP value.
ContributorKind = Literal["human", "agent", "system", "unknown"]
# OMP 3.1.1 ``ai_usage`` (values to be aligned with EU AI Act terminology; the
# COGX 0.2 ``ai_involvement`` scale is kept until the group settles them).
AIUsage = Literal["none", "draft", "mixed", "full"]
# OMP 3.1.1 ``contribution`` examples.
CONTRIBUTIONS = (
    "authored",
    "instructed",
    "approved",
    "edited",
    "imported",
    "verified",
    "corrected",
    "contradicted",
)
# OMP 3.2 ``rel`` examples.
LINK_RELS = ("derived_from", "supersedes", "relates_to", "contradicts")
# OMP 5.3 manifest vocabularies.
SELECTIONS = ("snapshot", "selection", "history")
INTENTS = ("copy", "move", "federation")
PROFILES = ("minimal", "audit")

# COGX 0.2 contributor roles that are exact synonyms of OMP contributions.
# ``system`` (the pipeline or model that ran) has no synonym and is kept as
# written rather than guessed.
_LEGACY_ROLES = {"initiator": "instructed", "author": "authored", "approver": "approved"}

# OMP 1.3: ``media_type`` MUST be a valid MIME type (RFC 6838 type/subtype,
# optional parameters).
_MEDIA_TYPE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}"
    r"(?:\s*;\s*[A-Za-z0-9!#$&^_.+-]+=(?:\"[^\"]*\"|[^\s;]+))*$"
)


def is_media_type(value: Any) -> bool:
    """True when ``value`` is a syntactically valid MIME type."""
    return isinstance(value, str) and _MEDIA_TYPE.match(value) is not None


def content_hash_of(text: str) -> str:
    """Hash of the body alone (``sha256:<hex>``): "do I already have this?".

    Distinct from the record ``checksum``, which answers "did it arrive
    intact?". Identical bodies hash identically regardless of metadata
    differences, which is what dedupe across copies needs (OMP U-1).
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_checksum(payload: dict[str, Any]) -> str:
    """``sha256:<hex>`` over the canonical JSON of ``payload``.

    Canonical means sorted keys, no whitespace, non-ASCII kept as is; the
    ``checksum`` key itself is left out. Field order and newly added empty
    fields therefore never change a checksum.
    """
    body = {key: value for key, value in payload.items() if key != "checksum"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class COGXOrigin(BaseModel):
    """Where a record first came from, kept across copies and re-exports.

    ``external_system``/``external_id`` describe the system that wrote *this*
    archive; ``origin`` names the first system, so a memory imported from Mem0
    into Cognee and exported again still says it came from Mem0. (Cognee's
    origin qualifier, OMP 5.1.2; the OMP manifest carries the store-level
    ``origin``, OMP 5.3.)
    """

    system: str
    id: str
    version: str | None = None


class COGXVersion(BaseModel):
    """OMP 3.3.2: an opaque revision with a declared comparator.

    A bare integer is read as ``{"value": n, "comparator": "integer"}``; a
    bare string declares ``opaque`` — no comparator was stated, so two such
    versions are incomparable. ``semver`` compares dotted integers.
    """

    value: int | str
    comparator: str = "integer"

    @model_validator(mode="before")
    @classmethod
    def _from_scalar(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("a version cannot be a boolean")
        if isinstance(value, int):
            return {"value": value, "comparator": "integer"}
        if isinstance(value, str):
            return {"value": value, "comparator": "opaque"}
        return value

    def compare(self, other: "COGXVersion") -> int | None:
        """-1, 0 or 1 when both versions share a comparator that orders them;
        None when they are incomparable (OMP 3.3.2 allows that)."""
        if self.comparator != other.comparator:
            return None
        if self.comparator == "integer":
            try:
                left, right = int(self.value), int(other.value)
            except (TypeError, ValueError):
                return None
        elif self.comparator == "semver":
            try:
                left = tuple(int(part) for part in str(self.value).split("."))
                right = tuple(int(part) for part in str(other.value).split("."))
            except ValueError:
                return None
        else:
            return None
        return (left > right) - (left < right)


class COGXContributor(BaseModel):
    """One entry of the provenance list (OMP 3.1.1): who took part, and how.

    ``id`` identifies the contributor, ``kind`` says whether it is a human, an
    agent or a system, ``contribution`` what it did (``authored``,
    ``instructed``, ``approved``, ``imported``, ``verified``, ...), ``ai_usage``
    how much AI was involved, ``name`` a label (for an agent, the model), and
    ``at`` when it acted. COGX 0.2 wrote ``role``/``actor``; both are read.
    """

    id: str
    kind: ContributorKind = "unknown"
    contribution: str | None = None
    ai_usage: AIUsage | None = None
    name: str | None = None
    at: UTCDateTime | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_legacy(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"id": value}
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "id" not in value and "actor" in value:
            value["id"] = value.pop("actor")
        if "contribution" not in value and "role" in value:
            role = value.pop("role")
            value["contribution"] = _LEGACY_ROLES.get(role, role)
        if "ai_usage" not in value and "ai_involvement" in value:
            value["ai_usage"] = value.pop("ai_involvement")
        return value


class COGXEvidence(BaseModel):
    """What a record rests on (OMP 3.1.4): one reference.

    ``ref`` is another memory's identifier, a source locator (a file, a URL, a
    transcript turn) or a person; ``relation`` says how the record relates to
    it (``observed``, ``stated``, ``inferred``, ``derived``); ``dangling`` is
    set by the receiver when it cannot resolve the reference (OMP 3.5.5).
    ``confidence`` is the writer's belief at write time, a COGX extension (the
    OMP I-20 residual). A plain string is read as a bare ``ref``; COGX 0.2
    wrote ``role`` for ``relation``.
    """

    ref: str
    relation: EvidenceRelation | None = None
    dangling: bool | None = None
    confidence: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_legacy(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"ref": value}
        if isinstance(value, dict) and "relation" not in value and "role" in value:
            value = dict(value)
            value["relation"] = value.pop("role")
        return value


class COGXLink(BaseModel):
    """A typed edge to another memory (OMP 3.2): ``{target, rel, version}``.

    ``rel`` values such as ``supersedes``, ``derived_from``, ``relates_to`` or
    ``contradicts``; ``version`` pins the target revision when it matters.
    ``dangling`` is set by the receiver when the target is not in the export
    or the store (OMP 3.5.5).
    """

    target: str
    rel: str = "relates_to"
    version: int | str | None = None
    dangling: bool | None = None


class COGXConfidence(BaseModel):
    """How much a memory can be trusted, measured after it was written.

    A COGX extension: OMP defines no portable confidence (3.1.6); ``system``
    names the system that produced the score so a receiver does not mistake
    it for a portable value. ``outcome`` records whether a later check
    confirmed, corrected or contradicted the memory; ``verified_by`` whether a
    human, an agent or an automatic rule did so; ``score`` is a value in
    [0, 1] on an implementation-defined scale. A bare number given where a
    record expects this object (the COGX 0.1 fact ``confidence`` float)
    becomes ``score`` with the outcome left ``unverified``.
    """

    outcome: ConfidenceOutcome = "unverified"
    verified_at: UTCDateTime | None = None
    verified_by: Verifier | None = None
    score: float | None = None
    system: str | None = None


# COGX 0.2 record field names and their 0.3 (OMP v0.6) names.
LEGACY_FIELD_NAMES = {
    "content_type": "media_type",
    "valid_to": "valid_until",
    "contributors": "provenance",
}


def normalize_legacy_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Map COGX 0.1/0.2 field names onto their 0.3 names, in a copy.

    A 0.3 name already present wins over its legacy spelling. The COGX 0.1
    fact ``provenance`` was a list of source strings (the episodes a fact was
    extracted from): that is evidence, not authorship, so such strings move
    to ``evidence`` as bare references with no relation stated. They move
    first: a 0.2 fact carries them beside ``contributors``, which must then
    find ``provenance`` empty to land there.
    """
    data = dict(data)
    provenance = data.get("provenance")
    if isinstance(provenance, list) and any(isinstance(item, str) for item in provenance):
        refs = [item for item in provenance if isinstance(item, str)]
        data["provenance"] = [item for item in provenance if not isinstance(item, str)]
        evidence = list(data.get("evidence") or [])
        evidence.extend({"ref": ref} for ref in refs)
        data["evidence"] = evidence
    for legacy, current in LEGACY_FIELD_NAMES.items():
        if legacy in data:
            legacy_value = data.pop(legacy)
            if data.get(current) in (None, [], ""):
                data[current] = legacy_value
    return data


class COGXRecordBase(BaseModel):
    external_system: str = "unknown"
    external_id: str
    scope: COGXScope = Field(default_factory=COGXScope)
    created_at: UTCDateTime | None = None
    updated_at: UTCDateTime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # OMP memory-object fields (draft v0.6, 1.3). ``external_id`` is the
    # memory's ``identifier``; ``created_at``/``updated_at`` are shared with
    # the record. All optional on a record: a cross-provider import states
    # what its source knows, and nothing is guessed.
    protocol_version: str | None = None
    path: str | None = None
    title: str | None = None
    media_type: str | None = None  # the body's MIME type, e.g. text/plain, text/markdown
    stub: bool | None = None
    tags: list[str] = Field(default_factory=list)
    scope_tags: list[str] = Field(default_factory=list)  # OMP ``scope``: access-policy tags
    valid_from: UTCDateTime | None = None
    valid_until: UTCDateTime | None = None
    version: COGXVersion | None = None
    checksum: str | None = None  # stamped by the writer over the whole record
    provenance: list[COGXContributor] = Field(default_factory=list)
    evidence: list[COGXEvidence] = Field(default_factory=list)
    links: list[COGXLink] = Field(default_factory=list)
    weight: float | None = None
    # Fields the memory carried that this module does not define (OMP 1.3:
    # unknown fields MUST be preserved on export).
    extensions: dict[str, Any] = Field(default_factory=dict)

    # Cognee extensions (OMP 5.1.2, left for review in PR #16).
    origin: COGXOrigin | None = None
    content_hash: str | None = None
    accountable: str | None = None  # the one human an auditor needs
    ai_involvement: AIUsage | None = None  # record-level summary of the contributors' ai_usage
    subject: list[str] = Field(default_factory=list)
    semantic_kind: SemanticKind | None = None
    invalidated_by: str | None = None  # reverse pointer of a ``supersedes`` link
    confidence: COGXConfidence | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_names(cls, data: Any) -> Any:
        return normalize_legacy_fields(data) if isinstance(data, dict) else data

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
        if self.media_type is None and body:
            self.media_type = "text/plain"
        return self


class COGXDocument(COGXRecordBase):
    """Raw source content: a file, archival passage, or standalone text."""

    kind: Literal["document"] = "document"
    content: str
    mime_type: str | None = None

    def body_text(self) -> str | None:
        return self.content

    @model_validator(mode="after")
    def _media_type_from_mime(self) -> Self:
        if self.media_type in (None, "text/plain") and self.mime_type:
            self.media_type = self.mime_type
        return self


class COGXTurn(BaseModel):
    role: str
    content: str
    occurred_at: UTCDateTime | None = None


class COGXEpisode(COGXRecordBase):
    """A conversation episode: ordered turns with roles and timestamps."""

    kind: Literal["episode"] = "episode"
    turns: list[COGXTurn] = Field(default_factory=list)

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
    valid_at: UTCDateTime | None = None
    invalid_at: UTCDateTime | None = None

    def readable_text(self) -> str:
        """The fact for a reader: its text, else its triple, with the endpoint
        names ``subject`` carries in place of their ids where it has them."""
        if self.fact_text:
            return self.fact_text
        subject, obj = (list(self.subject) + [None, None])[:2]
        return f"{subject or self.subject_ref} {self.predicate} {obj or self.object_ref}"

    def body_text(self) -> str | None:
        return self.fact_text

    @model_validator(mode="after")
    def _sync_validity(self) -> Self:
        """``valid_at``/``invalid_at`` (0.1) and ``valid_from``/``valid_until``
        (OMP) are the same interval; whichever side is set fills the other."""
        if self.valid_from is None and self.valid_at is not None:
            self.valid_from = self.valid_at
        elif self.valid_at is None and self.valid_from is not None:
            self.valid_at = self.valid_from
        if self.valid_until is None and self.invalid_at is not None:
            self.valid_until = self.invalid_at
        elif self.invalid_at is None and self.valid_until is not None:
            self.invalid_at = self.valid_until
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

# The OMP metadata a record carries beyond its COGX identity and timestamps.
HEADER_FIELDS = (
    "protocol_version",
    "path",
    "title",
    "media_type",
    "stub",
    "tags",
    "scope_tags",
    "valid_from",
    "valid_until",
    "version",
    "checksum",
    "provenance",
    "evidence",
    "links",
    "weight",
    "extensions",
    "origin",
    "content_hash",
    "accountable",
    "ai_involvement",
    "subject",
    "semantic_kind",
    "invalidated_by",
    "confidence",
)

# Header fields that describe one exported record's bytes or export, not the
# memory: every export computes or declares them afresh, so an import never
# stores them. ``media_type`` is stored only when it is not the plain-text
# default a body gets anyway.
RECORD_FIELDS = ("content_hash", "checksum", "protocol_version")

_HEADER_ADAPTERS: dict[str, TypeAdapter] = {
    key: TypeAdapter(COGXRecordBase.model_fields[key].annotation)
    for key in HEADER_FIELDS
    if key not in RECORD_FIELDS
}


def _without_empty(value: Any) -> Any:
    """``value`` with empty lists and dicts left out of every mapping in it."""
    if isinstance(value, dict):
        # Clean first, then filter: a mapping that empties out is dropped too.
        cleaned = {key: _without_empty(item) for key, item in value.items()}
        return {key: item for key, item in cleaned.items() if item not in ([], {})}
    if isinstance(value, list):
        return [_without_empty(item) for item in value]
    return value


def checksum_of(record: BaseModel) -> str:
    """Integrity hash of the exported record, metadata and body, minus the
    checksum itself (OMP 1.3 ``checksum``): "did it arrive intact?". Not the
    body hash. Computed over the canonical JSON form (:func:`canonical_checksum`),
    empty fields left out so a field a later version adds never changes it.
    """
    return canonical_checksum(_without_empty(record.model_dump(mode="json", exclude_none=True)))


def verify_checksum(record: BaseModel) -> bool | None:
    """True/False when the record carries a checksum; None when it does not."""
    stored = getattr(record, "checksum", None)
    if not stored:
        return None
    return checksum_of(record) == stored


def omp_header_of(record: COGXRecordBase) -> dict[str, Any]:
    """The record's OMP metadata as one JSON-safe dict, for storage on import.

    Imports keep it as the ``omp_header`` of the element they create (a data
    item's external metadata, an Entity node, a fact edge), so a re-export
    states the metadata the memory arrived with. Empty fields are left out.

    A record naming no earlier origin originates where it says it comes from
    (``external_system``/``external_id``): a Mem0 memory stays Mem0's after
    cognee exports it, since the export restamps both with cognee's own.

    ``dangling`` marks are not stored: whether a reference resolves depends on
    the export or the reader, so each export marks its own.
    """
    # Empty fields are left out at the top level only: a field's own value (an
    # unknown field in ``extensions`` included) is kept exactly as stated.
    header = {
        key: value
        for key, value in record.model_dump(mode="json", exclude_none=True).items()
        if key in HEADER_FIELDS and key not in RECORD_FIELDS and value not in ([], {})
    }
    for key in ("links", "evidence"):
        for entry in header.get(key, []):
            entry.pop("dangling", None)
    if header.get("media_type") == "text/plain":
        del header["media_type"]  # the default every body gets; nothing to remember
    if "origin" not in header and record.external_system != "unknown":
        header["origin"] = {"system": record.external_system, "id": record.external_id}
    return header


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def parse_omp_header(value: Any) -> dict[str, Any]:
    """Read a stored ``omp_header`` (JSON string or dict) back into record fields.

    Headers stored by COGX 0.2 imports use the 0.2 names and are mapped. Each
    field is validated against the record model on its own; a malformed one
    is dropped, never guessed, so an absent field means "not stated".
    """
    stored = _loads(value)
    if not isinstance(stored, dict):
        return {}
    stored = normalize_legacy_fields(stored)
    header: dict[str, Any] = {}
    for key, adapter in _HEADER_ADAPTERS.items():
        if stored.get(key) in (None, [], {}):
            continue
        try:
            header[key] = adapter.validate_python(stored[key])
        except ValueError:
            continue
    return header


def header_from_graph_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """OMP metadata cognee's own node/edge properties state: version, tags,
    kind, validity, supersession, weight and the numeric edge ``confidence``.

    The header an import carried is stored separately (see
    :func:`parse_omp_header`). Malformed values are dropped, never guessed.
    """
    header: dict[str, Any] = {}
    version = properties.get("version")
    if isinstance(version, int) and not isinstance(version, bool):
        header["version"] = COGXVersion(value=version)
    tags = _loads(properties.get("tags"))
    if isinstance(tags, list):
        header["tags"] = [str(item) for item in tags]
    kind = properties.get("semantic_kind") or properties.get("lesson_kind")
    if kind in SEMANTIC_KINDS:
        header["semantic_kind"] = kind
    parsed = parse_timestamp(properties.get("valid_from"))
    if parsed is not None:
        header["valid_from"] = parsed
    parsed = parse_timestamp(properties.get("valid_until") or properties.get("valid_to"))
    if parsed is not None:
        header["valid_until"] = parsed
    invalidated_by = properties.get("invalidated_by") or properties.get("superseded_by")
    if invalidated_by:
        header["invalidated_by"] = str(invalidated_by)
    weight = properties.get("weight")
    if isinstance(weight, (int, float)) and not isinstance(weight, bool):
        header["weight"] = float(weight)
    score = properties.get("confidence")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        header["confidence"] = COGXConfidence(score=float(score))
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
    """What an export says about itself, once per archive.

    The OMP manifest fields (draft v0.6, 5.3 working text) sit beside COGX's
    own: ``protocol_version``, the ``profiles`` the export follows, the
    store-level ``origin`` (with a record's ``external_id`` the memory's
    global key), ``exported_at``, whether the export is a ``snapshot``, a
    ``selection`` or a full ``history``, the transfer ``intent`` and the body
    ``media_types`` present.
    """

    cogx_version: str = COGX_VERSION
    protocol_version: str | None = None
    source_system: str = "unknown"
    origin: str | None = None
    exported_at: UTCDateTime | None = None
    selection: str = "snapshot"
    intent: str = "copy"
    profiles: list[str] = Field(default_factory=list)
    media_types: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    embedding_model: str | None = None
    # Cognee-origin archives only: the source store's stamped data-migration
    # revision at export time. Raw nodes keep their source-store ids, so the
    # importing store must not claim a newer revision than the exported data
    # actually has (None = unknown; the import keeps its own stamp).
    migration_revision: str | None = None
    notes: list[str] = Field(default_factory=list)
    # What the exporter could state for each OMP field, so an importer reads
    # absence as "not populated" rather than "not true".
    coverage: dict[str, str] = Field(default_factory=dict)


def parse_record(data: dict[str, Any]) -> COGXRecord:
    return _record_adapter.validate_python(data)


def cogx_version_tuple(version: str) -> tuple[int, int]:
    """``"0.3"`` -> ``(0, 3)``; raises ValueError for anything else."""
    major, minor = (str(version).split(".") + ["0"])[:2]
    return int(major), int(minor)


def validate_cogx_version(version: str) -> None:
    """Reject archives written by a newer major COGX version than this reader."""
    try:
        archive_major = cogx_version_tuple(version)[0]
        current_major = cogx_version_tuple(COGX_VERSION)[0]
    except (ValueError, IndexError):
        raise ValueError(f"Unrecognized COGX version {version!r} (reader supports {COGX_VERSION}).")
    if archive_major > current_major:
        raise ValueError(
            f"COGX archive version {version} is newer than this reader "
            f"supports ({COGX_VERSION}). Upgrade cognee to import it."
        )


def record_meets_audit_profile(record: COGXRecordBase) -> bool:
    """OMP 3.7 audit profile, per record: provenance naming a human, and a
    version. The checksum is stamped by the writer and is not checked here."""
    return record.version is not None and any(
        contributor.kind == "human" for contributor in record.provenance
    )


class COGXArchiveWriter:
    """Writes COGX records into an archive directory, one JSONL file per kind.

    The destination is treated as owned by the writer: any record files or
    manifest left by a previous export session are removed at init so a
    re-export never appends duplicates next to a stale manifest.

    The manifest declares the OMP profiles the records meet: ``minimal``
    always (every typed record renders as a memory with the required fields,
    :func:`cognee.modules.migration.omp.memory_from_record` fills a missing
    ``created_at`` from ``exported_at``), ``audit`` when every record carries
    a human contributor and a version (:func:`record_meets_audit_profile`).
    """

    def __init__(
        self,
        directory: str | Path,
        source_system: str = "cognee",
        origin: str | None = None,
        intent: str = "copy",
        selection: str = "snapshot",
    ):
        if intent not in INTENTS:
            raise ValueError(f"Unknown transfer intent {intent!r}. Expected one of {INTENTS}.")
        if selection not in SELECTIONS:
            raise ValueError(f"Unknown selection {selection!r}. Expected one of {SELECTIONS}.")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for file_name in (*RECORD_FILES.values(), RAW_NODES_FILE, MANIFEST_FILE, PERMISSIONS_FILE):
            (self.directory / file_name).unlink(missing_ok=True)
        self.source_system = source_system
        self.origin = origin
        self.intent = intent
        self.selection = selection
        self.counts: dict[str, int] = {}
        self.notes: list[str] = []
        self.embedding_model: str | None = None
        self.migration_revision: str | None = None
        self.coverage: dict[str, str] = {}
        self._media_types: set[str] = set()
        self._audit = True
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
        if record.media_type:
            self._media_types.add(record.media_type)
        if self._audit and not record_meets_audit_profile(record):
            self._audit = False

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

    @property
    def profiles(self) -> list[str]:
        typed = sum(count for kind, count in self.counts.items() if kind != "raw_node")
        return ["minimal", "audit"] if typed and self._audit else ["minimal"]

    def close(self, write_manifest: bool = True) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        if write_manifest:
            manifest = COGXManifest(
                protocol_version=OMP_PROTOCOL_VERSION,
                source_system=self.source_system,
                origin=self.origin,
                exported_at=datetime.now(timezone.utc),
                selection=self.selection,
                intent=self.intent,
                profiles=self.profiles,
                media_types=sorted(self._media_types),
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
    It is a sidecar for cognee-to-cognee restores, never part of a memory
    (OMP 3.5.6: credentials are not memory).
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
