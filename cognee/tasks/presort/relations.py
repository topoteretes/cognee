"""
Generic relation computation for presort: the spec's root-entity relation
fields drive what gets computed, resolved in priority order:

1. built-in derivers (``duplicate_of``, ``version_of``, ``belongs_to_group``,
   ``contains_pii``) — derived from the deterministic typed sections;
2. custom detectors registered via ``register_relation_detector(name, fn)``;
3. the LLM fallback (``use_llm=True``): a structured-output extraction over
   each text file's sample, targeting the relation's target entity;
4. otherwise: an empty list plus a report warning naming the gap.

This makes the report's relationship content follow the spec instead of a
fixed menu: a spec relation is only silent if nothing can compute it.
"""

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.graph_models import GraphSchemaSpec, RelationFieldSpec
from cognee.shared.logging_utils import get_logger

from .models import (
    DuplicateCluster,
    FileRecord,
    PiiFinding,
    ProposedGroup,
    RelationInstance,
    VersionCandidate,
)

logger = get_logger("presort")

_LLM_CONCURRENCY = 8
_LLM_MIN_CONFIDENCE = 0.5


@dataclass
class RelationContext:
    """Everything a relation detector may need."""

    root: Path
    spec: GraphSchemaSpec
    relation: RelationFieldSpec
    files: List[FileRecord]
    duplicates: List[DuplicateCluster]
    versions: List[VersionCandidate]
    pii: List[PiiFinding]
    groups: List[ProposedGroup]
    max_sample_bytes: int = 65536


RelationDetector = Callable[
    [RelationContext], Union[List[RelationInstance], Awaitable[List[RelationInstance]]]
]

_CUSTOM_DETECTORS: Dict[str, RelationDetector] = {}


def register_relation_detector(relation_name: str, detector: RelationDetector) -> None:
    """Register a detector for a spec relation presort has no built-in for.

    The detector receives a :class:`RelationContext` and returns (or awaits to)
    a list of :class:`RelationInstance`. It takes priority over the LLM
    fallback but never over a built-in deriver.
    """
    _CUSTOM_DETECTORS[relation_name] = detector


def _derive_duplicate_of(ctx: RelationContext) -> List[RelationInstance]:
    return [
        RelationInstance(
            source=path,
            relation=ctx.relation.name,
            target=cluster.paths[0],
            target_entity=ctx.relation.relation.target_entity_name,
            detail="identical content hash",
        )
        for cluster in ctx.duplicates
        for path in cluster.paths[1:]
    ]


def _derive_version_of(ctx: RelationContext) -> List[RelationInstance]:
    return [
        RelationInstance(
            source=path,
            relation=ctx.relation.name,
            target=candidate.paths[-1],
            target_entity=ctx.relation.relation.target_entity_name,
            detail=f"revision of {candidate.normalized_stem!r}",
        )
        for candidate in ctx.versions
        for path in candidate.paths[:-1]
    ]


def _derive_belongs_to_group(ctx: RelationContext) -> List[RelationInstance]:
    return [
        RelationInstance(
            source=path,
            relation=ctx.relation.name,
            target=group.name,
            target_entity=ctx.relation.relation.target_entity_name,
            detail=group.reason,
        )
        for group in ctx.groups
        for path in group.file_paths
    ]


def _derive_contains_pii(ctx: RelationContext) -> List[RelationInstance]:
    return [
        RelationInstance(
            source=finding.path,
            relation=ctx.relation.name,
            target=finding.category,
            target_entity=ctx.relation.relation.target_entity_name,
            detail=f"{finding.severity} severity ({finding.source})",
        )
        for finding in ctx.pii
    ]


_BUILTIN_DERIVERS: Dict[str, Callable[[RelationContext], List[RelationInstance]]] = {
    "duplicate_of": _derive_duplicate_of,
    "version_of": _derive_version_of,
    "belongs_to_group": _derive_belongs_to_group,
    "contains_pii": _derive_contains_pii,
}


class ExtractedRelationTarget(BaseModel):
    target_name: str = Field(description="Name of the target entity this file relates to")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One short sentence; no verbatim personal data")


class RelationExtraction(BaseModel):
    instances: List[ExtractedRelationTarget] = Field(default_factory=list)


async def _llm_detect(ctx: RelationContext) -> List[RelationInstance]:
    """LLM fallback for relations with no detector: structured extraction per text file."""
    target_name = ctx.relation.relation.target_entity_name
    target_entity = next(
        (entity for entity in ctx.spec.entities if entity.name == target_name), None
    )
    system_prompt = render_prompt(
        "extract_presort_relation.txt",
        {
            "relation_name": ctx.relation.name,
            "relation_description": ctx.relation.description or "",
            "target_entity": target_name,
            "target_description": (target_entity.description if target_entity else "") or "",
        },
    )

    candidates = [record for record in ctx.files if record.is_text and not record.is_code]
    semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

    async def extract(record: FileRecord) -> List[RelationInstance]:
        async with semaphore:
            try:
                with open(record.path, "rb") as file:
                    sample = file.read(ctx.max_sample_bytes).decode("utf-8", errors="replace")
                extraction = await LLMGateway.acreate_structured_output(
                    f"File name: {record.name}\n\n{sample}", system_prompt, RelationExtraction
                )
            except Exception as error:  # LLM failures must not abort presort
                record.warnings.append(
                    f"LLM extraction for relation {ctx.relation.name!r} failed: {error}"
                )
                logger.debug(f"Presort LLM relation failed for {record.path}: {error}")
                return []
            return [
                RelationInstance(
                    source=record.path,
                    relation=ctx.relation.name,
                    target=instance.target_name,
                    target_entity=target_name,
                    origin="llm",
                    confidence=instance.confidence,
                    detail=instance.rationale,
                )
                for instance in extraction.instances
                if instance.confidence >= _LLM_MIN_CONFIDENCE
            ]

    results = await asyncio.gather(*(extract(record) for record in candidates))
    return [instance for batch in results for instance in batch]


async def compute_relationships(
    root: Path,
    spec: GraphSchemaSpec,
    files: List[FileRecord],
    duplicates: List[DuplicateCluster],
    versions: List[VersionCandidate],
    pii: List[PiiFinding],
    groups: List[ProposedGroup],
    *,
    use_llm: bool = False,
    max_sample_bytes: int = 65536,
) -> tuple[Dict[str, List[RelationInstance]], List[str]]:
    """Compute instances for every relation on the spec's root entity.

    Returns ``(relationships, warnings)``: one dict entry per declared
    relation, and a warning per relation nothing could compute.
    """
    relationships: Dict[str, List[RelationInstance]] = {}
    warnings: List[str] = []

    root_entity = spec.root_entity()
    for field in root_entity.fields:
        if field.kind != "relation":
            continue
        ctx = RelationContext(
            root=root,
            spec=spec,
            relation=field,
            files=files,
            duplicates=duplicates,
            versions=versions,
            pii=pii,
            groups=groups,
            max_sample_bytes=max_sample_bytes,
        )

        if field.name in _BUILTIN_DERIVERS:
            relationships[field.name] = _BUILTIN_DERIVERS[field.name](ctx)
        elif field.name in _CUSTOM_DETECTORS:
            result = _CUSTOM_DETECTORS[field.name](ctx)
            if inspect.isawaitable(result):
                result = await result
            relationships[field.name] = [
                instance.model_copy(update={"origin": "custom"}) for instance in result
            ]
        elif use_llm:
            relationships[field.name] = await _llm_detect(ctx)
        else:
            relationships[field.name] = []
            warnings.append(
                f"relation {field.name!r}: no built-in or registered detector and "
                "use_llm=False — no instances computed"
            )

    return relationships, warnings


def builtin_relation_names() -> frozenset:
    return frozenset(_BUILTIN_DERIVERS)


def registered_relation_names() -> frozenset:
    return frozenset(_CUSTOM_DETECTORS)


def unregister_relation_detector(relation_name: str) -> Optional[RelationDetector]:
    return _CUSTOM_DETECTORS.pop(relation_name, None)
