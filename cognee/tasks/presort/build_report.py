"""
Report assembly for presort: the relationship spec (a JSON graph-model DSL
document, default ``DEFAULT_PRESORT_SPEC``) drives which relationship sections
the report carries — a spec whose root entity omits e.g. ``contains_pii``
drops the PII section.
"""

from pathlib import Path
from typing import List, Optional
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.graph_models import GraphSchemaSpec

from .models import (
    DuplicateCluster,
    FileRecord,
    JunkFile,
    PiiFinding,
    PresortReport,
    ProposedGroup,
    RelationInstance,
    VersionCandidate,
)

# Relation-field names on the spec's root entity -> report sections they gate.
RELATION_SECTIONS = {
    "duplicate_of": "duplicates",
    "version_of": "versions",
    "belongs_to_group": "groups",
    "contains_pii": "pii",
}


def enabled_sections(spec: GraphSchemaSpec) -> set:
    root = spec.root_entity()
    relation_names = {field.name for field in root.fields if field.kind == "relation"}
    return {
        section
        for relation_name, section in RELATION_SECTIONS.items()
        if relation_name in relation_names
    }


def scan_id_for(root_path: str, owner_id: Optional[str] = None) -> str:
    return str(uuid5(NAMESPACE_OID, f"presort:{owner_id or ''}:{root_path}"))


def build_report(
    root: Path,
    files: List[FileRecord],
    junk: List[JunkFile],
    duplicates: List[DuplicateCluster],
    versions: List[VersionCandidate],
    pii: List[PiiFinding],
    groups: List[ProposedGroup],
    *,
    spec: GraphSchemaSpec,
    relationships: Optional[dict[str, List[RelationInstance]]] = None,
    used_llm: bool = False,
    warnings: Optional[List[str]] = None,
    owner_id: Optional[str] = None,
) -> PresortReport:
    sections = enabled_sections(spec)
    warnings = list(warnings or [])

    for section_name in RELATION_SECTIONS.values():
        if section_name not in sections:
            warnings.append(
                f"section {section_name!r} disabled: the relationship spec's root entity "
                "declares no relation for it"
            )

    return PresortReport(
        scan_id=scan_id_for(str(root), owner_id),
        root_path=str(root),
        created_at=PresortReport._now(),
        used_llm=used_llm,
        spec_used=spec.model_dump(),
        files=files,
        junk=junk,
        duplicates=duplicates if "duplicates" in sections else [],
        versions=versions if "versions" in sections else [],
        pii=pii if "pii" in sections else [],
        groups=groups if "groups" in sections else [],
        relationships=relationships or {},
        warnings=warnings,
    )
