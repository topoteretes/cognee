"""Ingest SKILL.md files as Skill DataPoints into the knowledge graph.

Single entry point for all skill ingestion. Always uses the rich parser
(content hash, resources, extra metadata). With ``enrich=True`` also runs
the LLM enrichment pipeline (triggers, tags, instruction_summary,
complexity, task_pattern_candidates) and materializes TaskPattern nodes.

Re-ingesting an already-ingested skills folder is safe: each skill's
``content_hash`` is compared against the graph, and only new or
changed skills go through the persist path. Removed skills are
deleted from both graph and vector stores, and every change gets a
``SkillChangeEvent`` node for audit.

Called directly from ``cognee.remember(path, enrich=...)``.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.modules.engine.models import Skill
from cognee.modules.pipelines.models import PipelineContext
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points


logger = get_logger("cognee.tools.ingest_skills")


def _is_skill_entry(path: Path) -> bool:
    return path.is_file() and path.name.lower() == "skill.md"


def looks_like_skill_source(data) -> bool:
    """Return True when ``data`` is a SKILL.md file, a directory containing
    SKILL.md files at any depth, or a directory of skill folders.
    """
    if not isinstance(data, str):
        return False
    path = Path(data)
    try:
        if _is_skill_entry(path):
            return True
        if path.is_dir():
            return any(p.name.lower() == "skill.md" for p in path.rglob("*") if p.is_file())
    except OSError:
        return False
    return False


def _scope_matches(props: dict, dataset_id: Optional[UUID]) -> bool:
    scope = props.get("dataset_scope")
    if not scope:
        return True
    if dataset_id is None:
        return False
    return str(dataset_id) in scope


def _skill_source_data_id(dataset_id: UUID, source: Path) -> UUID:
    """Stable pseudo data id used to attach direct Skill writes to dataset ACL tables."""
    return uuid5(NAMESPACE_URL, f"cognee:skills:{dataset_id}:{source}")


def _make_storage_context(user, dataset, source: Path) -> Optional[PipelineContext]:
    """Build a minimal PipelineContext so add_data_points writes dataset ACL rows."""
    if user is None or dataset is None:
        return None
    return PipelineContext(
        user=user,
        dataset=dataset,
        data_item=SimpleNamespace(id=_skill_source_data_id(dataset.id, source)),
        pipeline_name="skills_ingest_pipeline",
    )


async def _diff_against_graph(
    parsed: List[Skill],
    node_set: str,
    dataset_id: Optional[UUID] = None,
) -> Tuple[List[Skill], List[str], List[Tuple[str, str, str, str]]]:
    """Diff parsed skills against what's already in the graph.

    Returns ``(to_persist, nids_to_delete, change_event_args)`` where:

    * ``to_persist`` — Skills whose ``content_hash`` is new or changed.
    * ``nids_to_delete`` — graph node IDs for skills that changed or
      that existed before but are no longer on disk. Both must be
      cleared from graph and vector before re-adding the new version.
    * ``change_event_args`` — 4-tuples of
      ``(name, change_type, old_hash, new_hash)`` to turn into
      ``SkillChangeEvent`` nodes.
    """
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.engine.models.node_set import NodeSet

    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    existing: Dict[str, Tuple[str, dict]] = {
        props.get("name"): (str(nid), props)
        for nid, props in raw_nodes
        if props.get("type") == "Skill" and props.get("name") and _scope_matches(props, dataset_id)
    }

    parsed_by_name = {s.name: s for s in parsed}
    to_persist: List[Skill] = []
    nids_to_delete: List[str] = []
    events: List[Tuple[str, str, str, str]] = []  # (name, type, old_hash, new_hash)

    for name, skill in parsed_by_name.items():
        if name in existing:
            old_nid, old_props = existing[name]
            old_hash = old_props.get("content_hash", "")
            if old_hash == skill.content_hash:
                continue  # unchanged
            nids_to_delete.append(old_nid)
            to_persist.append(skill)
            events.append((name, "updated", old_hash, skill.content_hash))
        else:
            to_persist.append(skill)
            events.append((name, "added", "", skill.content_hash))

    for name in existing:
        if name not in parsed_by_name:
            old_nid, old_props = existing[name]
            nids_to_delete.append(old_nid)
            events.append((name, "removed", old_props.get("content_hash", ""), ""))

    return to_persist, nids_to_delete, events


async def _delete_stale(nids_to_delete: List[str]) -> None:
    """Delete graph nodes and their vector-index rows for stale Skills."""
    if not nids_to_delete:
        return
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine

    engine = await get_graph_engine()
    await engine.delete_nodes(nids_to_delete)
    vector_engine = get_vector_engine()
    for field in ("name", "instruction_summary", "description"):
        try:
            await vector_engine.delete_data_points(f"Skill_{field}", nids_to_delete)
        except Exception:
            pass
    logger.info("Deleted %d stale skill node(s)", len(nids_to_delete))


async def add_skills(
    source: Union[str, Path],
    enrich: bool = True,
    source_repo: str = "",
    node_set: str = "skills",
    user=None,
    dataset=None,
) -> List[Skill]:
    """Parse SKILL.md file(s) and persist them as Skill DataPoints.

    Args:
        source: Either a SKILL.md file or a directory of skill folders /
            flat .md files.
        enrich: When True (default), runs LLM enrichment (triggers, tags,
            instruction_summary, complexity, task_pattern_candidates) and
            materializes TaskPattern nodes before persisting — which
            makes skill routing by ``cognee.search(..., skills_auto_retrieve=True)``
            and ``cognee.skills.run(...)`` meaningfully effective. Set
            to False for fast / offline ingest; the Skill will still be
            ingested and directly-named ``search(..., skills=["..."])``
            calls work, but auto-routing quality drops.
        source_repo: Provenance label attached to each Skill.
        node_set: Tag applied via ``belongs_to_set`` for vector-search
            scoping.
        user: Optional resolved user. When provided with ``dataset``, direct
            ``add_data_points`` calls write the same dataset ACL rows as normal
            pipelines.
        dataset: Optional resolved dataset. Its id is copied to
            ``Skill.dataset_scope`` so skills cannot leak across datasets.

    Returns:
        Only the Skills that were *persisted* this call. Unchanged
        skills (same ``content_hash`` as the graph copy) are silently
        skipped.
    """
    # Late import keeps parser dependencies off the hot import path.
    from cognee.modules.tools.skill_parser import (
        parse_skill_file,
        parse_skills_folder,
    )

    path = Path(source).expanduser().resolve()
    if path.is_dir():
        parsed = parse_skills_folder(path, source_repo=source_repo, base_dir=path)
    elif path.is_file():
        one = parse_skill_file(path, source_repo=source_repo, base_dir=path.parent)
        parsed = [one] if one is not None else []
    else:
        raise FileNotFoundError(f"Skill source not found: {source}")

    if not parsed:
        logger.warning("No SKILL.md files discovered under %s", source)
        return []

    dataset_id = getattr(dataset, "id", None)
    if dataset_id is not None:
        scope = str(dataset_id)
        for skill in parsed:
            if not skill.dataset_scope:
                skill.dataset_scope = [scope]
            elif scope not in skill.dataset_scope:
                skill.dataset_scope.append(scope)

    ctx = _make_storage_context(user, dataset, path)

    # Diff against what's already in the graph. Skip unchanged, delete
    # old copies of changed skills + skills that disappeared from disk,
    # and emit SkillChangeEvent nodes for the audit trail.
    to_persist, nids_to_delete, events = await _diff_against_graph(parsed, node_set, dataset_id)
    await _delete_stale(nids_to_delete)

    if events:
        from cognee.modules.tools.skill_change_events import _make_change_event

        change_nodes = [
            _make_change_event(name, name, change_type, old_hash=oh, new_hash=nh)
            for name, change_type, oh, nh in events
        ]
        await add_data_points(change_nodes, ctx=ctx)

    if not to_persist:
        logger.info("No changes to persist for %s (all %d skill(s) unchanged)", source, len(parsed))
        return []

    if enrich:
        from cognee.pipelines import Task, run_tasks
        from cognee.modules.users.methods import get_default_user
        from cognee.modules.data.methods import load_or_create_datasets
        from cognee.tasks.storage.index_graph_edges import index_graph_edges

        from cognee.modules.tools.skill_enrichment_tasks import enrich_skills
        from cognee.modules.tools.skill_pattern_tasks import (
            materialize_task_patterns,
        )
        from cognee.modules.tools.skill_node_set_task import apply_node_set

        if user is None:
            user = await get_default_user()
        if dataset is None:
            datasets = await load_or_create_datasets([node_set], [], user)
            dataset = datasets[0]
        dataset_id = dataset.id

        tasks = [
            Task(enrich_skills),
            Task(materialize_task_patterns),
            Task(apply_node_set, node_set=node_set),
            Task(add_data_points),
        ]
        async for status in run_tasks(
            tasks, dataset_id, to_persist, user, "skills_enrich_pipeline"
        ):
            logger.info("Enrich pipeline status: %s", status)
        await index_graph_edges()
    else:
        await add_data_points(to_persist, ctx=ctx)

    logger.info(
        "Ingested %d skill(s) from %s (enrich=%s); %d unchanged, %d removed",
        len(to_persist),
        source,
        enrich,
        len(parsed) - len(to_persist),
        sum(1 for _, ct, _, _ in events if ct == "removed"),
    )
    return to_persist
