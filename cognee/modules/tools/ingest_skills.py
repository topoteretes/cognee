"""Explicit SKILL.md ingestion for the agentic-skills v1 scope."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.modules.engine.models import NodeSet, Skill
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.tools.path_safety import trusted_is_dir, trusted_is_file, trusted_rglob
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points


logger = get_logger("cognee.tools.ingest_skills")

SKILL_SOURCE_ROOTS_ENV = "COGNEE_SKILL_SOURCE_ROOTS"


def _configured_skill_source_roots() -> Tuple[Path, ...]:
    roots = [Path.cwd()]
    for raw_root in os.environ.get(SKILL_SOURCE_ROOTS_ENV, "").split(os.pathsep):
        raw_root = raw_root.strip()
        if raw_root:
            roots.append(Path(raw_root).expanduser())
    return tuple(roots)


def _normalize_skill_path(path: Union[str, Path]) -> str:
    return os.path.normpath(os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path)))))


def _has_allowed_skill_root(path_str: str, root_str: str) -> bool:
    root_prefix = root_str if root_str.endswith(os.sep) else f"{root_str}{os.sep}"
    return path_str == root_str or path_str.startswith(root_prefix)


def _resolve_candidate_under_root(raw_path: str, root: Path) -> Optional[str]:
    root_str = _normalize_skill_path(root)
    candidate = raw_path if os.path.isabs(raw_path) else os.path.join(root_str, raw_path)
    candidate = _normalize_skill_path(candidate)
    if _has_allowed_skill_root(candidate, root_str):
        return candidate
    return None


def _resolve_skill_source_path(source: Union[str, Path]) -> Optional[Path]:
    if isinstance(source, Path):
        raw_path = os.fspath(source)
    elif isinstance(source, str):
        raw_path = source.strip()
        if not raw_path or "\x00" in raw_path or "://" in raw_path:
            return None
    else:
        return None

    for root in _configured_skill_source_roots():
        try:
            candidate = _resolve_candidate_under_root(raw_path, root)
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate is not None:
            return Path(candidate)
    return None


def looks_like_skill_source(data) -> bool:
    """Return True for an explicit skill source candidate.

    This helper is intentionally not used by ``remember()`` for auto-dispatch.
    Callers must opt in with ``content_type="skills"``.
    """
    path = _resolve_skill_source_path(data)
    if path is None:
        return False
    try:
        if trusted_is_file(path):
            return path.name.lower() == "skill.md"
        if trusted_is_dir(path):
            return any(
                trusted_is_file(candidate) and candidate.name.lower() == "skill.md"
                for candidate in trusted_rglob(path, "*")
            )
    except OSError:
        return False
    return False


def _skill_source_data_id(dataset_id: UUID, source: Path) -> UUID:
    return uuid5(NAMESPACE_URL, f"cognee:skills:{dataset_id}:{_normalize_skill_path(source)}")


def _scoped_skill_id(dataset_id: UUID, skill: Skill) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"cognee:skill:{dataset_id}:{skill.source_dir}:{skill.name}",
    )


def _make_storage_context(user, dataset, source: Path) -> Optional[PipelineContext]:
    if user is None or dataset is None:
        return None
    return PipelineContext(
        user=user,
        dataset=dataset,
        data_item=SimpleNamespace(id=_skill_source_data_id(dataset.id, source)),
        pipeline_name="skills_ingest_pipeline",
    )


async def add_skills(
    source: Union[str, Path],
    *,
    source_repo: str = "",
    node_set: str = "skills",
    user=None,
    dataset=None,
) -> List[Skill]:
    """Parse and persist SKILL.md files as dataset-scoped Skill nodes."""
    if dataset is None or getattr(dataset, "id", None) is None:
        raise ValueError("Skill ingestion requires one explicit dataset.")

    from cognee.modules.tools.skill_parser import parse_skill_file, parse_skills_folder

    path = _resolve_skill_source_path(source)
    if path is None:
        raise PermissionError(
            f"Skill source must be under the current working directory or a root "
            f"listed in {SKILL_SOURCE_ROOTS_ENV}: {source}"
        )

    if trusted_is_dir(path):
        parsed = parse_skills_folder(path, source_repo=source_repo, base_dir=path)
    elif trusted_is_file(path):
        skill = parse_skill_file(path, source_repo=source_repo, base_dir=path.parent)
        parsed = [skill] if skill is not None else []
    else:
        raise FileNotFoundError(f"Skill source not found: {source}")

    if not parsed:
        logger.warning("No SKILL.md files discovered under %s", source)
        return []

    dataset_id = dataset.id
    node_set_point = NodeSet(id=generate_node_id(f"NodeSet:{node_set}"), name=node_set)
    scoped: List[Skill] = []
    for skill in parsed:
        skill.id = _scoped_skill_id(dataset_id, skill)
        skill.dataset_scope = [str(dataset_id)]
        skill.belongs_to_set = [node_set_point]
        if not skill.skill_text:
            skill.skill_text = "\n\n".join(
                part for part in (skill.name, skill.description, skill.procedure) if part
            )
        if not skill.search_text:
            skill.search_text = skill.skill_text
        scoped.append(skill)

    await add_data_points(scoped, ctx=_make_storage_context(user, dataset, path))
    logger.info("Ingested %d skill(s) from %s into dataset %s", len(scoped), source, dataset.name)
    return scoped
