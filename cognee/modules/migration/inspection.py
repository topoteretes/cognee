"""Inspect portable COGX archives without importing their contents."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, computed_field

from .cogx import (
    COGX_VERSION,
    MANIFEST_FILE,
    PERMISSIONS_FILE,
    RAW_NODES_FILE,
    RECORD_FILES,
    COGXManifest,
    COGXRawNode,
    parse_record,
    validate_cogx_version,
    verify_checksum,
)


class COGXInspectionDiagnostic(BaseModel):
    """A content-free diagnostic identifying a file and optional line."""

    file: str
    line: int | None = None
    code: str
    severity: Literal["error", "warning"]
    message: str


class COGXArchiveInspection(BaseModel):
    """Archive metadata, checksum coverage, and validation findings."""

    directory: str
    cogx_version: str | None = None
    source_system: str | None = None
    record_counts: dict[str, int] = Field(default_factory=dict)
    declared_counts: dict[str, int] = Field(default_factory=dict)
    checksum_counts: dict[str, int] = Field(
        default_factory=lambda: {"valid": 0, "invalid": 0, "missing": 0}
    )
    contains_permissions: bool = False
    diagnostics: list[COGXInspectionDiagnostic] = Field(default_factory=list)

    @computed_field
    @property
    def valid(self) -> bool:
        """Whether inspection found no error-level diagnostics."""
        return not any(item.severity == "error" for item in self.diagnostics)

    @computed_field
    @property
    def total_records(self) -> int:
        """Number of schema-valid records in their expected files."""
        return sum(self.record_counts.values())


def inspect_cogx_archive(directory: str | Path) -> COGXArchiveInspection:
    """Check schemas, inventory, and available checksums without modifying stores.

    Missing manifests warn for legacy compatibility. Diagnostics omit record
    contents; permissions are detected but never read. Fact references and
    database-specific import behavior are not validated.
    """
    report = COGXArchiveInspection(directory=str(directory))

    def diagnostic(
        file: str,
        code: str,
        message: str,
        *,
        line: int | None = None,
        severity: Literal["error", "warning"] = "error",
    ) -> None:
        report.diagnostics.append(
            COGXInspectionDiagnostic(
                file=file, line=line, code=code, severity=severity, message=message
            )
        )

    try:
        base = Path(directory).resolve()
        if not base.is_dir():
            diagnostic(".", "directory_not_found", "COGX archive directory does not exist.")
            return report
    except (OSError, RuntimeError):
        diagnostic(".", "directory_unreadable", "COGX archive directory cannot be inspected.")
        return report

    files: dict[str, Path] = {}
    present: set[str] = set()
    for name in (MANIFEST_FILE, *RECORD_FILES.values(), RAW_NODES_FILE, PERMISSIONS_FILE):
        candidate = base / name
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            diagnostic(name, "unreadable_file", "Archive file cannot be inspected.")
            continue
        present.add(name)
        if name == PERMISSIONS_FILE:
            report.contains_permissions = True
        try:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(base):
                diagnostic(name, "unsafe_file", "Archive file resolves outside its directory.")
            elif not resolved.is_file():
                diagnostic(name, "invalid_file", "Expected a regular archive file.")
            else:
                files[name] = resolved
        except (OSError, RuntimeError):
            diagnostic(name, "unreadable_file", "Archive file cannot be inspected.")

    if not present.difference({PERMISSIONS_FILE}):
        diagnostic(".", "not_archive", "No COGX manifest or recognized record files found.")
        return report

    counts_declared = False
    manifest_path = files.get(MANIFEST_FILE)
    if MANIFEST_FILE not in present:
        diagnostic(
            MANIFEST_FILE,
            "missing_manifest",
            "No manifest found; version and declared counts cannot be checked.",
            severity="warning",
        )
    elif manifest_path is not None:
        try:
            manifest = COGXManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            diagnostic(MANIFEST_FILE, "invalid_manifest", "Manifest is not valid COGX JSON.")
        except OSError:
            diagnostic(MANIFEST_FILE, "unreadable_file", "Manifest cannot be read.")
        else:
            report.cogx_version = manifest.cogx_version
            report.source_system = manifest.source_system
            report.declared_counts = dict(manifest.counts)
            counts_declared = "counts" in manifest.model_fields_set
            if any(count < 0 for count in manifest.counts.values()):
                diagnostic(
                    MANIFEST_FILE,
                    "invalid_manifest",
                    "Declared record counts must be non-negative.",
                )
            try:
                validate_cogx_version(manifest.cogx_version)
            except ValueError:
                diagnostic(
                    MANIFEST_FILE,
                    "unsupported_version",
                    f"Manifest version is not supported by the COGX {COGX_VERSION} reader.",
                )

    for kind, name in (*RECORD_FILES.items(), ("raw_node", RAW_NODES_FILE)):
        path = files.get(name)
        if path is None:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        diagnostic(
                            name, "invalid_json", "Record is not valid JSON.", line=line_number
                        )
                        continue
                    try:
                        record = (
                            COGXRawNode(properties=payload)
                            if kind == "raw_node"
                            else parse_record(payload)
                        )
                    except (ValidationError, ValueError, TypeError):
                        diagnostic(
                            name,
                            "invalid_record",
                            "Record does not match the COGX schema.",
                            line=line_number,
                        )
                        continue
                    if record.kind != kind:
                        diagnostic(
                            name,
                            "wrong_record_kind",
                            "Record kind does not match its archive file.",
                            line=line_number,
                        )
                        continue
                    report.record_counts[kind] = report.record_counts.get(kind, 0) + 1
                    try:
                        checksum = verify_checksum(record)
                    except ValueError:
                        # JSON may contain surrogates that cannot be encoded for hashing.
                        report.checksum_counts["invalid"] += 1
                        diagnostic(
                            name,
                            "checksum_error",
                            "Record cannot be serialized for checksum verification.",
                            line=line_number,
                        )
                        continue
                    if checksum is None:
                        report.checksum_counts["missing"] += 1
                    elif checksum:
                        report.checksum_counts["valid"] += 1
                    else:
                        report.checksum_counts["invalid"] += 1
                        diagnostic(
                            name,
                            "checksum_mismatch",
                            "Record checksum does not match its contents.",
                            line=line_number,
                        )
        except (OSError, UnicodeError):
            diagnostic(name, "unreadable_file", "Record file cannot be read as UTF-8 text.")

    if counts_declared:
        for kind in sorted(report.declared_counts.keys() | report.record_counts.keys()):
            declared = report.declared_counts.get(kind, 0)
            observed = report.record_counts.get(kind, 0)
            if declared != observed:
                diagnostic(
                    MANIFEST_FILE,
                    "count_mismatch",
                    f"Declared count for {kind!r} is {declared}; "
                    f"observed {observed} schema-valid records.",
                )
    return report
