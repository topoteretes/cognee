"""
Pydantic models for the presort report.

A ``PresortReport`` is the bridge between the two presort phases: the analyze
phase (``remember(path, dry_run="presort")``) produces it, and the apply phase
(``remember(report)``) consumes it. It is JSON-serializable and carries
everything apply needs — proposed groups, per-file verdicts, and the apply
options — so a report can be saved, reviewed/edited, and applied later.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# Marker key used to recognize a report dict handed to remember() as data.
PRESORT_REPORT_MARKER = "presort_report"

CogneeStatus = Literal["new", "staged", "cognified", "unknown"]
PiiSeverity = Literal["low", "medium", "high"]


class _ReportBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileRecord(_ReportBase):
    path: str
    name: str
    extension: str = ""
    mime_type: Optional[str] = None
    size_bytes: int = 0
    is_text: bool = False
    # Whether a registered cognee loader claims this path extension.
    loader_claimed: bool = False
    is_code: bool = False
    # Coarse deterministic bucket (documents/images/audio/video/archives/code/
    # data/other); refined by the opt-in LLM classification.
    family: str = "other"
    content_class: Optional[str] = None  # LLM content classification, if run
    content_hash: Optional[str] = None
    cognee_status: CogneeStatus = "unknown"
    known_in_datasets: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JunkFile(_ReportBase):
    path: str
    reason: str


class DuplicateCluster(_ReportBase):
    content_hash: str
    # Ordered: the first path is the one apply keeps when skip_duplicates=True.
    paths: list[str]
    size_bytes: int = 0

    @property
    def wasted_bytes(self) -> int:
        return self.size_bytes * (len(self.paths) - 1)


class VersionCandidate(_ReportBase):
    normalized_stem: str
    extension: str
    directory: str
    # Ordered oldest -> newest by modification time; the last is the latest.
    paths: list[str]


class PiiFinding(_ReportBase):
    path: str
    category: str
    severity: PiiSeverity = "medium"
    source: Literal["filename", "content", "llm"] = "content"
    # Samples are always redacted (e.g. "j***@example.com"); raw matches are
    # never stored in the report.
    redacted_sample: Optional[str] = None
    detail: Optional[str] = None


class RelationInstance(_ReportBase):
    """One computed instance of a spec-declared relation.

    The generic view of the report's findings, keyed by the relation names on
    the relationship spec's root entity. ``source`` is a file path (a root
    entity instance); ``target`` is a file path for self-relations or the
    target entity's name otherwise.
    """

    source: str
    relation: str
    target: str
    target_entity: str
    origin: Literal["detector", "custom", "llm"] = "detector"
    confidence: Optional[float] = None
    detail: Optional[str] = None


class ProposedGroup(_ReportBase):
    name: str
    dataset_name: str
    kind: Literal["code_project", "folder", "extension_family", "semantic"] = "folder"
    reason: str = ""
    file_paths: list[str] = Field(default_factory=list)


class PresortReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Marker so remember() can recognize a report dict passed as `data`.
    presort_report: Literal[True] = True
    scan_id: str = ""
    root_path: str = ""
    created_at: str = ""
    used_llm: bool = False
    spec_used: dict = Field(default_factory=dict)

    files: list[FileRecord] = Field(default_factory=list)
    junk: list[JunkFile] = Field(default_factory=list)
    duplicates: list[DuplicateCluster] = Field(default_factory=list)
    versions: list[VersionCandidate] = Field(default_factory=list)
    pii: list[PiiFinding] = Field(default_factory=list)
    groups: list[ProposedGroup] = Field(default_factory=list)
    # Generic view: one entry per relation declared on the spec's root entity
    # (built-ins derived from the typed sections above; custom/LLM detectors
    # fill the rest). This is what apply_graph writes into the graph.
    relationships: dict[str, list[RelationInstance]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # Apply options — settable on the report (or overridden via kwargs) before
    # handing it back to remember(report).
    skip_duplicates: bool = True
    exclude_pii: bool = False
    apply_groups: Optional[list[str]] = None

    report_path: Optional[str] = None  # where the report was persisted, if it was

    # Populated only by auto_apply (remember(..., dry_run="presort", auto_apply=True)):
    # {dataset_name: RememberResult}. Live objects — excluded from serialization.
    apply_results: Optional[dict] = Field(default=None, exclude=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def summary(self) -> dict:
        status_counts: dict = {"new": 0, "staged": 0, "cognified": 0, "unknown": 0}
        for file in self.files:
            status_counts[file.cognee_status] += 1
        new_bytes = sum(f.size_bytes for f in self.files if f.cognee_status == "new")
        return {
            "root_path": self.root_path,
            "files": len(self.files),
            "junk": len(self.junk),
            "duplicate_clusters": len(self.duplicates),
            "wasted_bytes": sum(cluster.wasted_bytes for cluster in self.duplicates),
            "version_candidates": len(self.versions),
            "pii_findings": len(self.pii),
            "files_with_pii": len({finding.path for finding in self.pii}),
            "groups": len(self.groups),
            "cognee_status": status_counts,
            "bytes_needing_processing": new_bytes,
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, destination: Union[str, Path]) -> str:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.report_path = str(destination)
        destination.write_text(self.to_json(), encoding="utf-8")
        return self.report_path

    @classmethod
    def from_json(cls, source: Union[str, Path, dict]) -> "PresortReport":
        """Load a report from a dict, a JSON string, or a path to a report file."""
        if isinstance(source, dict):
            return cls.model_validate(source)
        text = str(source)
        candidate = Path(text)
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8")
        except OSError:
            pass  # not a usable path (e.g. too long) — treat as JSON text
        return cls.model_validate(json.loads(text))


def looks_like_presort_report(data) -> bool:
    """Whether a remember() `data` argument is a presort report (object or dict)."""
    if isinstance(data, PresortReport):
        return True
    return isinstance(data, dict) and data.get(PRESORT_REPORT_MARKER) is True
