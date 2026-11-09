"""The Open Memory Protocol memory object (editors' draft v0.6, 2026-11-09).

An OMP memory is **metadata plus a body** (draft 1.1): the metadata is the
field table of 1.3, the body a string in the format ``media_type`` declares.
This module gives it a model (:class:`OMPMemory`), the two representations
of 3.4 (``interchange()`` for export and import, ``runtime_view()`` for what
an agent sees), the integrity ``checksum`` of 1.3, the conformance profiles
of 3.7, the markdown-with-front-matter carrier of 3.6.6 and Appendix B, and
the mapping between a memory and a COGX record in both directions.

Export and import files: ``memories.jsonl`` holds one interchange object per
line beside a ``manifest.json`` (draft 5.3), or a folder of ``.md`` files
carries the same memories as front matter plus body. A folder of markdown
memories with no other metadata meets the minimal profile (Appendix C).
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from cognee.modules.migration.cogx import (
    OMP_PROTOCOL_VERSION,
    COGXContributor,
    COGXDocument,
    COGXEntity,
    COGXEpisode,
    COGXEvidence,
    COGXFact,
    COGXLink,
    COGXMemory,
    COGXMemoryBlock,
    COGXOrigin,
    COGXRecord,
    COGXRecordBase,
    COGXScope,
    COGXTurn,
    COGXVersion,
    canonical_checksum,
    is_media_type,
)

OMP_MANIFEST_FILE = "manifest.json"
OMP_MEMORIES_FILE = "memories.jsonl"
# The root file of the markdown loading profile (draft 3.6.6).
OMP_INDEX_FILE = "MEMORY.md"
OMP_LAYOUTS = ("jsonl", "markdown")

# COGX ownership scope fields and the OMP ``scope`` tag prefix each becomes.
_SCOPE_PREFIXES = {"user_id": "user", "agent_id": "agent", "session_id": "session", "run_id": "run"}
_TIMESTAMP_FIELDS = ("created_at", "updated_at", "valid_from", "valid_until")
_FRONT_MATTER = re.compile(
    r"\A---[ \t]*\r?\n(.*?)(?:\r?\n)?^---[ \t]*(?:\r?\n|\Z)", re.DOTALL | re.MULTILINE
)
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class OMPManifest(BaseModel):
    """What an export says about itself (draft 5.3 working text).

    ``origin`` is the store the memories come from; with a memory's
    ``identifier`` it is the memory's global key. ``exported_at`` is the time
    the export is as of. ``selection`` says whether the export is a complete
    ``snapshot``, a ``selection`` or a full ``history``; absence from a
    partial export is not deletion (3.5.1). ``intent`` is the transfer intent
    of 3.5.2. Unknown fields are kept.
    """

    model_config = ConfigDict(extra="allow")

    protocol_version: str = OMP_PROTOCOL_VERSION
    profiles: list[str] = Field(default_factory=list)
    origin: str | None = None
    exported_at: datetime | None = None
    selection: str = "snapshot"
    intent: str = "copy"
    media_types: list[str] = Field(default_factory=list)


@dataclass
class OMPConformance:
    """What a memory can claim (draft 3.7).

    ``problems`` say why a profile is not met; ``notes`` are observations the
    profiles ask to record without failing them (dangling references, an
    unknown author, an expired validity).
    """

    minimal: bool
    audit: bool
    problems: list[str]
    notes: list[str]

    @property
    def profiles(self) -> list[str]:
        profiles = []
        if self.minimal:
            profiles.append("minimal")
        if self.audit:
            profiles.append("audit")
        return profiles


class OMPMemory(BaseModel):
    """One memory in the interchange representation (draft 1.3, Appendix A).

    ``identifier``, ``media_type``, ``created_at`` and ``body`` are required;
    everything else is optional. Fields this model does not define are
    preserved (``model_extra``), as 1.3 requires; cognee's own extensions
    travel under one ``cognee`` key (see :func:`memory_from_record`).
    """

    model_config = ConfigDict(extra="allow")

    identifier: str
    media_type: str
    body: str
    created_at: datetime
    protocol_version: str | None = None
    path: str | None = None
    title: str | None = None
    stub: bool | None = None
    tags: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    version: COGXVersion | None = None
    updated_at: datetime | None = None
    checksum: str | None = None
    provenance: list[COGXContributor] = Field(default_factory=list)
    evidence: list[COGXEvidence] = Field(default_factory=list)
    links: list[COGXLink] = Field(default_factory=list)
    weight: float | None = None

    def extension(self, name: str) -> Any:
        """A field outside the standard table, or None."""
        return (self.model_extra or {}).get(name)

    def interchange(self) -> dict[str, Any]:
        """The memory as the JSON object of Appendix A: every stated field,
        unknown fields included, empty lists and null fields left out."""
        data = self.model_dump(mode="json", exclude_none=True)
        return {key: value for key, value in data.items() if value not in ([], {})}

    def compute_checksum(self) -> str:
        """``sha256:<hex>`` over the canonical interchange form minus ``checksum``."""
        return canonical_checksum(self.interchange())

    def stamp_checksum(self) -> Self:
        self.checksum = self.compute_checksum()
        return self

    def verify_checksum(self) -> bool | None:
        """True/False when the memory carries a checksum; None when it does not.

        The draft fixes no canonical form (U-1), so a memory another system
        hashed differently reports False here; receivers report that, they
        do not reject on it (3.5.7).
        """
        if not self.checksum:
            return None
        return self.compute_checksum() == self.checksum

    def conformance(self) -> OMPConformance:
        """Check the memory against the minimal and audit profiles (3.7)."""
        problems: list[str] = []
        notes: list[str] = []
        if not self.identifier.strip():
            problems.append("identifier is empty (1.3)")
        if not is_media_type(self.media_type):
            problems.append(f"media_type {self.media_type!r} is not a valid MIME type (1.3)")
        minimal = not problems

        audit_problems: list[str] = []
        if not any(contributor.kind == "human" for contributor in self.provenance):
            audit_problems.append("no provenance entry with kind human (3.1.3)")
        if self.version is None:
            audit_problems.append("no version (1.3)")
        verified = self.verify_checksum()
        if verified is None:
            audit_problems.append("no checksum (1.3)")
        elif not verified:
            audit_problems.append("checksum does not match the canonical form (3.5.3)")
        problems.extend(f"audit: {problem}" for problem in audit_problems)

        if not self.provenance:
            notes.append("no provenance: author unknown (3.1.5)")
        dangling = [entry.ref for entry in self.evidence if entry.dangling] + [
            link.target for link in self.links if link.dangling
        ]
        if dangling:
            notes.append(f"dangling references kept and marked (3.5.5): {', '.join(dangling)}")
        if self.valid_until is not None and self.valid_until <= datetime.now(timezone.utc):
            notes.append("past valid_until: excluded from current reads (3.3.4)")
        return OMPConformance(
            minimal=minimal, audit=minimal and not audit_problems, problems=problems, notes=notes
        )

    def runtime_view(self) -> dict[str, Any]:
        """The projection an agent sees (3.4 placement table, Appendix B).

        Keeps identifier, media type, body, stub when true, title, tags,
        created_at, path, the version value, the validity interval, a
        provenance summary (who, what kind, which contribution) and the
        links; drops checksum, scope, evidence, weight, update time,
        protocol version and every extension field.
        """
        view: dict[str, Any] = {
            "identifier": self.identifier,
            "media_type": self.media_type,
            "body": self.body,
        }
        if self.stub:
            view["stub"] = True
        if self.title:
            view["title"] = self.title
        if self.tags:
            view["tags"] = list(self.tags)
        view["created_at"] = self.created_at.isoformat()
        if self.path:
            view["path"] = self.path
        if self.version is not None:
            view["version"] = self.version.value
        for key in ("valid_from", "valid_until"):
            value = getattr(self, key)
            if value is not None:
                view[key] = value.isoformat()
        if self.provenance:
            view["provenance"] = [
                {
                    key: value
                    for key, value in (
                        ("id", contributor.id),
                        ("kind", contributor.kind),
                        ("contribution", contributor.contribution),
                    )
                    if value is not None
                }
                for contributor in self.provenance
            ]
        if self.links:
            view["links"] = [
                link.model_dump(exclude_none=True, exclude={"dangling"}) for link in self.links
            ]
        return view

    def to_markdown(self) -> str:
        """The memory as markdown with YAML front matter (3.6.6, Appendix B).

        The front matter carries the full interchange metadata, so nothing
        is lost in this carrier; the body follows one blank line after the
        closing fence, exactly as written.
        """
        metadata = {key: value for key, value in self.interchange().items() if key != "body"}
        front_matter = yaml.safe_dump(
            metadata, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
        )
        return f"---\n{front_matter}---\n\n{self.body}"

    @classmethod
    def from_markdown(
        cls,
        text: str,
        *,
        identifier: str | None = None,
        path: str | None = None,
        created_at: datetime | None = None,
    ) -> Self:
        """Read a markdown memory: YAML front matter (optional) plus body.

        A file with no front matter is the smallest conforming memory
        (Appendix C) once its carrier supplies what the file cannot:
        ``identifier`` (its path), ``created_at`` (when the store recorded
        it) and ``media_type`` (markdown). Values the front matter states win.
        """
        metadata: dict[str, Any] = {}
        body = text
        match = _FRONT_MATTER.match(text)
        if match:
            loaded = yaml.safe_load(match.group(1)) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Front matter must be a YAML mapping.")
            metadata = loaded
            body = text[match.end() :]
            if body.startswith("\r\n"):
                body = body[2:]
            elif body.startswith("\n"):
                body = body[1:]
        if identifier is not None:
            metadata.setdefault("identifier", identifier)
        if path is not None:
            metadata.setdefault("path", path)
        if created_at is not None:
            metadata.setdefault("created_at", created_at)
        metadata.setdefault("media_type", "text/markdown")
        for key in _TIMESTAMP_FIELDS:
            value = metadata.get(key)
            if isinstance(value, date) and not isinstance(value, datetime):
                metadata[key] = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        if isinstance(metadata.get("protocol_version"), (int, float)):
            metadata["protocol_version"] = str(metadata["protocol_version"])
        return cls.model_validate({**metadata, "body": body})


# ── COGX record <-> OMP memory ──────────────────────────────────────────────


def _cognee_extension(record: COGXRecordBase) -> dict[str, Any]:
    """Cognee's fields outside the standard table (draft 5.1.2) and the
    record-kind payload a memory needs to become the same COGX record again."""
    extension: dict[str, Any] = {"kind": record.kind}
    if record.origin is not None:
        extension["origin"] = record.origin.model_dump(exclude_none=True)
    for key in ("content_hash", "accountable", "ai_involvement", "semantic_kind", "invalidated_by"):
        value = getattr(record, key)
        if value is not None:
            extension[key] = value
    if record.subject:
        extension["subject"] = list(record.subject)
    if record.confidence is not None:
        extension["confidence"] = record.confidence.model_dump(mode="json", exclude_none=True)
    if record.metadata:
        extension["metadata"] = dict(record.metadata)
    if isinstance(record, COGXEntity):
        extension["name"] = record.name
        for key in ("entity_type", "description", "aliases", "attributes"):
            value = getattr(record, key)
            if value not in (None, [], {}):
                extension[key] = value
    elif isinstance(record, COGXFact):
        extension["subject_ref"] = record.subject_ref
        extension["predicate"] = record.predicate
        extension["object_ref"] = record.object_ref
        if record.fact_text:
            extension["fact_text"] = record.fact_text
    elif isinstance(record, COGXDocument):
        if record.mime_type:
            extension["mime_type"] = record.mime_type
    elif isinstance(record, COGXEpisode):
        extension["turns"] = [
            turn.model_dump(mode="json", exclude_none=True) for turn in record.turns
        ]
    elif isinstance(record, COGXMemory):
        if record.categories:
            extension["categories"] = list(record.categories)
    elif isinstance(record, COGXMemoryBlock):
        extension["label"] = record.label
        if record.limit is not None:
            extension["limit"] = record.limit
    return extension


def _fact_body(record: COGXFact) -> str:
    """A fact without text renders as its triple, with the endpoint names the
    exporter put in ``subject`` where it had them."""
    subject, obj = (list(record.subject) + [None, None])[:2]
    return f"{subject or record.subject_ref} {record.predicate} {obj or record.object_ref}"


def memory_from_record(
    record: COGXRecord,
    *,
    exported_at: datetime | None = None,
    protocol_version: str | None = OMP_PROTOCOL_VERSION,
) -> OMPMemory | None:
    """Render a COGX record as an OMP memory; None for raw graph nodes.

    ``identifier`` is the record's ``external_id``. ``created_at`` is
    required by the draft: a record whose source recorded none gets the
    export time (``exported_at``, else now), the earliest moment this
    system can vouch for. The ownership scope becomes ``user:``/``agent:``/
    ``session:``/``run:`` scope tags beside the record's own; a semantic kind
    and Mem0-style categories join the tags (1.3: tags are where a system
    records the kind of a memory). Cognee's extensions and the record kind's
    own payload travel under ``cognee`` so :func:`record_from_memory` is
    lossless; ``invalidated_by`` becomes a ``superseded_by`` link. The
    checksum is stamped last.
    """
    if not isinstance(record, COGXRecordBase):
        return None
    body = record.body_text()
    if body is None and isinstance(record, COGXFact):
        body = _fact_body(record)
    if body is None:
        body = ""

    scope = list(record.scope_tags)
    for field_name, prefix in _SCOPE_PREFIXES.items():
        value = getattr(record.scope, field_name)
        if value:
            scope.append(f"{prefix}:{value}")

    tags = list(record.tags)
    if record.semantic_kind and record.semantic_kind not in tags:
        tags.append(record.semantic_kind)
    if isinstance(record, COGXMemory):
        tags.extend(category for category in record.categories if category not in tags)

    title = record.title
    if title is None and isinstance(record, COGXEntity):
        title = record.name
    elif title is None and isinstance(record, COGXMemoryBlock):
        title = record.label

    links = list(record.links)
    if record.invalidated_by and not any(
        link.rel == "superseded_by" and link.target == record.invalidated_by for link in links
    ):
        links.append(COGXLink(target=record.invalidated_by, rel="superseded_by"))

    payload: dict[str, Any] = {
        **record.extensions,
        "identifier": record.external_id,
        "protocol_version": record.protocol_version or protocol_version,
        "path": record.path,
        "title": title,
        "media_type": record.media_type or "text/plain",
        "stub": record.stub,
        "tags": tags,
        "scope": scope,
        "created_at": record.created_at or exported_at or datetime.now(timezone.utc),
        "valid_from": record.valid_from,
        "valid_until": record.valid_until,
        "version": record.version,
        "updated_at": record.updated_at,
        "provenance": record.provenance,
        "evidence": record.evidence,
        "links": links,
        "weight": record.weight,
        "body": body,
        "cognee": _cognee_extension(record),
    }
    return OMPMemory.model_validate(payload).stamp_checksum()


def _split_scope(scope: list[str]) -> tuple[COGXScope, list[str]]:
    """``user:``/``agent:``/``session:``/``run:`` tags back into the ownership
    scope; every other tag stays an access-scope tag."""
    ownership: dict[str, str] = {}
    tags: list[str] = []
    prefixes = {prefix: field_name for field_name, prefix in _SCOPE_PREFIXES.items()}
    for tag in scope:
        prefix, separator, value = tag.partition(":")
        field_name = prefixes.get(prefix)
        if separator and field_name and value and field_name not in ownership:
            ownership[field_name] = value
        else:
            tags.append(tag)
    return COGXScope(**ownership), tags


def record_from_memory(memory: OMPMemory, *, external_system: str = "omp") -> COGXRecord:
    """Read an OMP memory as a COGX record.

    A memory that cognee exported (its ``cognee`` extension names the record
    kind) becomes that kind again, payload included. Any other memory becomes
    a :class:`COGXMemory`: body as content, standard fields carried, unknown
    fields kept in ``extensions`` so they are preserved on re-export (1.3).
    ``external_system`` is the store the memory comes from (the manifest's
    ``origin``), which with the identifier makes the import idempotent.
    """
    extras = dict(memory.model_extra or {})
    extension = extras.pop("cognee", None)
    extension = extension if isinstance(extension, dict) else {}
    scope, scope_tags = _split_scope(memory.scope)

    links = list(memory.links)
    invalidated_by = extension.get("invalidated_by")
    if invalidated_by is None:
        invalidated_by = next((link.target for link in links if link.rel == "superseded_by"), None)
    links = [link for link in links if link.rel != "superseded_by"]

    semantic_kind = extension.get("semantic_kind")
    categories = list(extension.get("categories") or [])
    tags = [tag for tag in memory.tags if tag != semantic_kind and tag not in categories]

    kind = extension.get("kind")
    title = memory.title
    # memory_from_record titles an entity by its name and a block by its
    # label when the record had no title of its own; undo that here.
    if (kind == "entity" and title == extension.get("name")) or (
        kind == "memory_block" and title == extension.get("label")
    ):
        title = None

    origin = extension.get("origin")
    confidence = extension.get("confidence")
    common: dict[str, Any] = {
        "external_system": external_system,
        "external_id": memory.identifier,
        "scope": scope,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "metadata": dict(extension.get("metadata") or {}),
        "protocol_version": memory.protocol_version,
        "path": memory.path,
        "title": title,
        "media_type": memory.media_type,
        "stub": memory.stub,
        "tags": tags,
        "scope_tags": scope_tags,
        "valid_from": memory.valid_from,
        "valid_until": memory.valid_until,
        "version": memory.version,
        "checksum": memory.checksum,
        "provenance": list(memory.provenance),
        "evidence": list(memory.evidence),
        "links": links,
        "weight": memory.weight,
        "extensions": extras,
        "origin": COGXOrigin.model_validate(origin) if isinstance(origin, dict) else None,
        "accountable": extension.get("accountable"),
        "ai_involvement": extension.get("ai_involvement"),
        "subject": list(extension.get("subject") or []),
        "semantic_kind": semantic_kind,
        "invalidated_by": invalidated_by,
        "confidence": confidence if isinstance(confidence, dict) else None,
    }

    if kind == "entity" and extension.get("name"):
        return COGXEntity(
            name=str(extension["name"]),
            entity_type=extension.get("entity_type"),
            description=extension.get("description"),
            aliases=list(extension.get("aliases") or []),
            attributes=dict(extension.get("attributes") or {}),
            **common,
        )
    if kind == "fact" and all(
        extension.get(key) for key in ("subject_ref", "predicate", "object_ref")
    ):
        return COGXFact(
            subject_ref=str(extension["subject_ref"]),
            predicate=str(extension["predicate"]),
            object_ref=str(extension["object_ref"]),
            fact_text=extension.get("fact_text"),
            **common,
        )
    if kind == "document":
        mime_type = extension.get("mime_type")
        if mime_type is None and memory.media_type != "text/plain":
            mime_type = memory.media_type
        return COGXDocument(content=memory.body, mime_type=mime_type, **common)
    if kind == "episode" and isinstance(extension.get("turns"), list):
        return COGXEpisode(
            turns=[COGXTurn.model_validate(turn) for turn in extension["turns"]], **common
        )
    if kind == "memory_block" and extension.get("label"):
        return COGXMemoryBlock(
            label=str(extension["label"]), value=memory.body, limit=extension.get("limit"), **common
        )
    return COGXMemory(content=memory.body, categories=categories, **common)


def mark_dangling(memory: OMPMemory, identifiers: set[str]) -> OMPMemory:
    """Mark the links and evidence a receiver cannot resolve (3.5.5).

    A reference resolves when it names a memory of the same export; anything
    else — a transcript turn, a URL, a memory left out of a selection — is
    dangling from the receiver's side. A mark the exporter already set is
    kept. References are never dropped.
    """
    for link in memory.links:
        if link.dangling is None:
            link.dangling = link.target not in identifiers
    for entry in memory.evidence:
        if entry.dangling is None:
            entry.dangling = entry.ref not in identifiers
    return memory


# ── Export and import files ────────────────────────────────────────────────


def _slug(value: str) -> str:
    slug = _SLUG_UNSAFE.sub("-", value).strip("-.")
    return slug or "memory"


def _relative_markdown_path(memory: OMPMemory) -> Path:
    """Where a memory sits in a markdown export: its ``path`` when it is a
    safe relative path (given a ``.md`` suffix if it has none), else a file
    named by its identifier."""
    if memory.path:
        candidate = Path(memory.path)
        parts = candidate.parts
        if (
            parts
            and not candidate.is_absolute()
            and all(part not in ("", ".", "..") and "\\" not in part for part in parts)
        ):
            if candidate.suffix.lower() != ".md":
                candidate = candidate.with_name(candidate.name + ".md")
            return candidate
    return Path(_slug(memory.identifier) + ".md")


def _index_markdown(entries: list[tuple[str, str]], exported_at: datetime) -> str:
    """The root ``MEMORY.md`` of the markdown loading profile (3.6.6): a stub
    memory whose body lists every deferred file one level down."""
    index = OMPMemory(
        identifier=OMP_INDEX_FILE,
        media_type="text/markdown",
        created_at=exported_at,
        title="Memory index",
        tags=["index"],
        stub=True,
        body="\n".join(f"- [{title}]({path})" for title, path in entries) + "\n",
    )
    return index.to_markdown()


def write_omp_export(
    memories: Iterable[OMPMemory],
    destination: str | Path,
    *,
    manifest: OMPManifest | None = None,
    layout: str = "jsonl",
) -> dict[str, Any]:
    """Write memories as an OMP export directory and return its counts.

    ``jsonl`` writes ``memories.jsonl`` (one Appendix A object per line);
    ``markdown`` writes one ``.md`` file per memory at its ``path`` plus a
    root ``MEMORY.md`` index (unless a memory claims that path). Both write
    ``manifest.json`` last, with the profiles every memory meets and the
    media types present filled in from the memories themselves.
    """
    if layout not in OMP_LAYOUTS:
        raise ValueError(f"Unknown OMP layout {layout!r}. Expected one of {OMP_LAYOUTS}.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = manifest.model_copy() if manifest else OMPManifest()
    if manifest.exported_at is None:
        manifest.exported_at = datetime.now(timezone.utc)

    count = 0
    minimal = audit = True
    media_types: set[str] = set()
    written: set[Path] = set()
    index_entries: list[tuple[str, str]] = []
    resolved_destination = destination.resolve()
    jsonl_handle = None
    if layout == "jsonl":
        jsonl_handle = open(destination / OMP_MEMORIES_FILE, "w", encoding="utf-8")  # noqa: SIM115 - closed below
    try:
        for memory in memories:
            conformance = memory.conformance()
            minimal = minimal and conformance.minimal
            audit = audit and conformance.audit
            media_types.add(memory.media_type)
            count += 1
            if jsonl_handle is not None:
                jsonl_handle.write(_dumps(memory.interchange()) + "\n")
                continue
            relative = _relative_markdown_path(memory)
            stem, suffix, bump = relative.with_suffix(""), relative.suffix, 1
            while relative in written:
                relative = stem.with_name(f"{stem.name}-{bump}{suffix}")
                bump += 1
            target = (destination / relative).resolve()
            if not target.is_relative_to(resolved_destination):
                relative = Path(_slug(memory.identifier) + ".md")
                target = (destination / relative).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(memory.to_markdown(), encoding="utf-8")
            written.add(relative)
            index_entries.append((memory.title or memory.identifier, relative.as_posix()))
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    if layout == "markdown" and Path(OMP_INDEX_FILE) not in written:
        (destination / OMP_INDEX_FILE).write_text(
            _index_markdown(index_entries, manifest.exported_at), encoding="utf-8"
        )

    manifest.media_types = sorted(media_types)
    manifest.profiles = (["minimal"] if minimal else []) + (["audit"] if count and audit else [])
    payload = manifest.model_dump(mode="json", exclude_none=True)
    payload["memories"] = count
    payload["layout"] = layout
    (destination / OMP_MANIFEST_FILE).write_text(_dumps(payload, indent=2), encoding="utf-8")
    return {"memories": count, "profiles": manifest.profiles, "layout": layout}


def _dumps(payload: Any, indent: int | None = None) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=indent)


def _loads(text: str) -> Any:
    import json

    return json.loads(text)


def read_omp_manifest(source: Any) -> OMPManifest | None:
    """The manifest of an export directory, or of a dict carrying one; None
    when there is none (a bare file, a folder of markdown)."""
    if isinstance(source, dict) and isinstance(source.get("manifest"), dict):
        return OMPManifest.model_validate(source["manifest"])
    if isinstance(source, (str, Path)):
        manifest_path = Path(source) / OMP_MANIFEST_FILE
        if Path(source).is_dir() and manifest_path.is_file():
            return OMPManifest.model_validate(_loads(manifest_path.read_text(encoding="utf-8")))
    return None


def _file_created_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _memory_from_file(path: Path, root: Path | None = None) -> OMPMemory:
    relative = path.relative_to(root).as_posix() if root else path.name
    return OMPMemory.from_markdown(
        path.read_text(encoding="utf-8"),
        identifier=relative,
        path=relative,
        created_at=_file_created_at(path),
    )


def iter_omp_memories(source: Any) -> Iterator[OMPMemory]:
    """Stream the memories of an OMP export.

    ``source`` is an export directory (``memories.jsonl`` and/or ``.md``
    files, read in path order), a ``.jsonl`` file, a ``.json`` file holding
    one memory, a list of them or ``{"memories": [...]}``, a single ``.md``
    file, or the same shapes already parsed (a list or dict).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            jsonl = path / OMP_MEMORIES_FILE
            if jsonl.is_file():
                yield from _iter_jsonl(jsonl)
            for markdown in sorted(path.rglob("*.md")):
                if markdown.is_file():
                    yield _memory_from_file(markdown, path)
            return
        if not path.is_file():
            raise FileNotFoundError(f"OMP export not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from _iter_jsonl(path)
        elif suffix == ".md":
            yield _memory_from_file(path)
        else:
            yield from _iter_parsed(_loads(path.read_text(encoding="utf-8")))
        return
    yield from _iter_parsed(source)


def _iter_jsonl(path: Path) -> Iterator[OMPMemory]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield OMPMemory.model_validate(_loads(line))


def _iter_parsed(data: Any) -> Iterator[OMPMemory]:
    if isinstance(data, dict):
        for key in ("memories", "items", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            yield OMPMemory.model_validate(data)
            return
    if not isinstance(data, list):
        raise ValueError(
            "Unrecognized OMP export shape: expected a memory, a list, or a directory."
        )
    for item in data:
        if isinstance(item, dict):
            yield OMPMemory.model_validate(item)
