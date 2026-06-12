"""COGX — the Cognee eXchange format for portable memory.

COGX is the hub format for memory migration: importers translate external
providers (Mem0, Zep/Graphiti, Letta, ...) into COGX records, a single loader
ingests COGX into Cognee, and a single dumper exports Cognee datasets into a
COGX archive that emitters can translate onward into other formats.

An archive is a directory containing ``manifest.json`` plus one JSONL file per
record kind. Records are Pydantic models discriminated by their ``kind`` field.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

COGX_VERSION = "0.1"


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse a timestamp from ISO strings, epoch seconds, or epoch milliseconds."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Heuristic: values past the year ~2603 in seconds are milliseconds.
        seconds = value / 1000 if value > 2e10 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class COGXScope(BaseModel):
    """Ownership scope of a memory record in the source system."""

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None


class COGXRecordBase(BaseModel):
    external_system: str = "unknown"
    external_id: str
    scope: COGXScope = Field(default_factory=COGXScope)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class COGXDocument(COGXRecordBase):
    """Raw source content: a file, archival passage, or standalone text."""

    kind: Literal["document"] = "document"
    content: str
    title: Optional[str] = None
    mime_type: Optional[str] = None


class COGXTurn(BaseModel):
    role: str
    content: str
    occurred_at: Optional[datetime] = None


class COGXEpisode(COGXRecordBase):
    """A conversation episode: ordered turns with roles and timestamps."""

    kind: Literal["episode"] = "episode"
    turns: List[COGXTurn] = Field(default_factory=list)
    title: Optional[str] = None


class COGXEntity(COGXRecordBase):
    """An extracted entity with optional type, aliases, and description."""

    kind: Literal["entity"] = "entity"
    name: str
    entity_type: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


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
    fact_text: Optional[str] = None
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    confidence: Optional[float] = None
    provenance: List[str] = Field(default_factory=list)


class COGXMemory(COGXRecordBase):
    """An atomic derived memory (Mem0-style short fact text)."""

    kind: Literal["memory"] = "memory"
    content: str
    categories: List[str] = Field(default_factory=list)


class COGXMemoryBlock(COGXRecordBase):
    """A named, bounded core-memory block (Letta-style)."""

    kind: Literal["memory_block"] = "memory_block"
    label: str
    value: str
    limit: Optional[int] = None


COGXRecord = Union[COGXDocument, COGXEpisode, COGXEntity, COGXFact, COGXMemory, COGXMemoryBlock]

_record_adapter: TypeAdapter = TypeAdapter(COGXRecord)

RECORD_FILES: Dict[str, str] = {
    "document": "documents.jsonl",
    "episode": "episodes.jsonl",
    "entity": "entities.jsonl",
    "fact": "facts.jsonl",
    "memory": "memories.jsonl",
    "memory_block": "memory_blocks.jsonl",
}

MANIFEST_FILE = "manifest.json"
RAW_NODES_FILE = "nodes.jsonl"


class COGXManifest(BaseModel):
    cogx_version: str = COGX_VERSION
    source_system: str = "unknown"
    exported_at: Optional[datetime] = None
    counts: Dict[str, int] = Field(default_factory=dict)
    embedding_model: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


def parse_record(data: Dict[str, Any]) -> COGXRecord:
    return _record_adapter.validate_python(data)


class COGXArchiveWriter:
    """Writes COGX records into an archive directory, one JSONL file per kind."""

    def __init__(self, directory: Union[str, Path], source_system: str = "cognee"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source_system = source_system
        self.counts: Dict[str, int] = {}
        self.notes: List[str] = []
        self._handles: Dict[str, Any] = {}

    def __enter__(self) -> "COGXArchiveWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(write_manifest=exc_type is None)

    def write(self, record: COGXRecord) -> None:
        file_name = RECORD_FILES[record.kind]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = open(self.directory / file_name, "a", encoding="utf-8")
            self._handles[file_name] = handle
        handle.write(record.model_dump_json(exclude_none=True) + "\n")
        self.counts[record.kind] = self.counts.get(record.kind, 0) + 1

    def write_raw_node(self, node: Dict[str, Any]) -> None:
        """Persist a graph node that has no typed COGX mapping (full fidelity)."""
        handle = self._handles.get(RAW_NODES_FILE)
        if handle is None:
            handle = open(self.directory / RAW_NODES_FILE, "a", encoding="utf-8")
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
                notes=self.notes,
            )
            manifest_path = self.directory / MANIFEST_FILE
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_manifest(directory: Union[str, Path]) -> Optional[COGXManifest]:
    manifest_path = Path(directory) / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    return COGXManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def read_archive(directory: Union[str, Path]) -> Iterator[COGXRecord]:
    """Stream typed records from a COGX archive directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"COGX archive directory not found: {directory}")
    for file_name in RECORD_FILES.values():
        file_path = directory / file_name
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield parse_record(json.loads(line))
