"""Skills ingestion pipeline: parse -> enrich -> materialize patterns -> add to graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from cognee.low_level import setup
from cognee.pipelines import Task, run_tasks
from cognee.tasks.storage import add_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import load_or_create_datasets
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.models.node_set import NodeSet

from cognee.cognee_skills.tasks.parse_skills import parse_skills_task
from cognee.cognee_skills.tasks.enrich_skills import enrich_skills
from cognee.cognee_skills.tasks.materialize_task_patterns import materialize_task_patterns
from cognee.cognee_skills.tasks.apply_node_set import apply_node_set
from cognee.cognee_skills.models.skill_change_event import SkillChangeEvent
from cognee.cognee_skills.utils import _make_change_event

logger = logging.getLogger(__name__)


async def ingest_skills(
    skills_folder: str | Path,
    dataset_name: str = "skills",
    source_repo: str = "",
    skip_enrichment: bool = False,
    node_set: str = "skills",
) -> None:
    """Ingest all skills from a folder into Cognee.

    Pipeline: parse -> enrich -> materialize -> apply_node_set -> add_data_points -> index

    Args:
        skills_folder: Path to the directory containing skill subdirectories.
        dataset_name: Cognee dataset name to store skills under.
        source_repo: Provenance label (e.g. "anthropics/skills").
        skip_enrichment: If True, skip the LLM enrichment step (parser output only).
        node_set: Tag for belongs_to_set (used for vector search scoping).
    """
    skills_folder = str(Path(skills_folder).resolve())

    await setup()

    user = await get_default_user()
    datasets = await load_or_create_datasets([dataset_name], [], user)
    dataset_id = datasets[0].id

    tasks = [
        Task(parse_skills_task, source_repo=source_repo),
    ]

    if not skip_enrichment:
        tasks.append(Task(enrich_skills))
        tasks.append(Task(materialize_task_patterns))

    tasks.append(Task(apply_node_set, node_set=node_set))
    tasks.append(Task(add_data_points))

    logger.info("Ingesting skills from %s into dataset '%s'", skills_folder, dataset_name)

    async for status in run_tasks(tasks, dataset_id, skills_folder, user, "skills_pipeline"):
        logger.info("Pipeline status: %s", status)

    await index_graph_edges()

    logger.info("Skills ingestion complete.")


async def upsert_skills(
    skills_folder: str | Path,
    dataset_name: str = "skills",
    source_repo: str = "",
    node_set: str = "skills",
) -> dict:
    """Re-ingest skills, deleting changed/removed ones and adding new versions.

    1. Parse the folder to get current skills.
    2. Load existing Skill nodes from graph.
    3. Compare content_hash: unchanged -> skip, changed -> delete old + re-add.
    4. Skills removed from folder -> delete from graph AND vector.
    5. Emit SkillChangeEvent nodes for temporal tracking.

    Returns a summary dict with counts.
    """
    from cognee.cognee_skills.parser.skill_parser import parse_skills_folder

    skills_folder = str(Path(skills_folder).resolve())
    await setup()

    new_skills = parse_skills_folder(skills_folder, source_repo=source_repo)
    new_by_id = {s.skill_id: s for s in new_skills}

    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=[node_set])
    existing_skills: Dict[str, tuple] = {
        props.get("skill_id"): (nid, props)
        for nid, props in raw_nodes
        if props.get("type") == "Skill"
    }

    unchanged, updated, added, removed = [], [], [], []

    for skill_id, skill in new_by_id.items():
        if skill_id in existing_skills:
            _, old_props = existing_skills[skill_id]
            if old_props.get("content_hash") == skill.content_hash:
                unchanged.append(skill_id)
            else:
                updated.append(skill_id)
        else:
            added.append(skill_id)

    for skill_id in existing_skills:
        if skill_id not in new_by_id:
            removed.append(skill_id)

    # --- Delete old nodes from graph AND vector ---
    to_delete_ids: List[str] = []
    for skill_id in updated + removed:
        if skill_id in existing_skills:
            nid, _ = existing_skills[skill_id]
            to_delete_ids.append(str(nid))

    if to_delete_ids:
        await engine.delete_nodes(to_delete_ids)

        vector_engine = get_vector_engine()
        for field in ["name", "instruction_summary", "description"]:
            collection = f"Skill_{field}"
            try:
                await vector_engine.delete_data_points(collection, to_delete_ids)
            except Exception:
                pass
        logger.info("Deleted %d old skill nodes (graph + vector)", len(to_delete_ids))

    # --- Emit temporal change events ---
    change_events: List[SkillChangeEvent] = []

    for skill_id in added:
        s = new_by_id[skill_id]
        change_events.append(
            _make_change_event(
                skill_id,
                s.name,
                "added",
                new_hash=s.content_hash,
            )
        )

    for skill_id in updated:
        s = new_by_id[skill_id]
        _, old_props = existing_skills[skill_id]
        change_events.append(
            _make_change_event(
                skill_id,
                s.name,
                "updated",
                old_hash=old_props.get("content_hash", ""),
                new_hash=s.content_hash,
            )
        )

    for skill_id in removed:
        _, old_props = existing_skills[skill_id]
        change_events.append(
            _make_change_event(
                skill_id,
                old_props.get("name", skill_id),
                "removed",
                old_hash=old_props.get("content_hash", ""),
            )
        )

    if change_events:
        await add_data_points(change_events)
        logger.info("Recorded %d SkillChangeEvents", len(change_events))

    # --- Re-ingest changed/new skills ---
    skills_to_ingest = [new_by_id[sid] for sid in updated + added]

    if skills_to_ingest:
        user = await get_default_user()
        datasets = await load_or_create_datasets([dataset_name], [], user)
        dataset_id = datasets[0].id

        tasks = [
            Task(enrich_skills),
            Task(materialize_task_patterns),
            Task(apply_node_set, node_set=node_set),
            Task(add_data_points),
        ]

        async for status in run_tasks(
            tasks, dataset_id, skills_to_ingest, user, "skills_upsert_pipeline"
        ):
            logger.info("Upsert pipeline status: %s", status)

        await index_graph_edges()

    summary = {
        "unchanged": len(unchanged),
        "updated": len(updated),
        "added": len(added),
        "removed": len(removed),
        "events_emitted": len(change_events),
    }
    logger.info("Upsert complete: %s", summary)
    return summary


async def remove_skill(skill_id: str) -> bool:
    """Remove a single skill by skill_id from graph and vector stores.

    Also emits a SkillChangeEvent for temporal tracking.

    Returns True if the skill was found and deleted, False otherwise.
    """
    await setup()

    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_nodeset_subgraph(node_type=NodeSet, node_name=["skills"])

    skill_nid = None
    skill_props = None
    for nid, props in raw_nodes:
        if props.get("type") == "Skill" and props.get("skill_id") == skill_id:
            skill_nid = str(nid)
            skill_props = props
            break

    if skill_nid is None:
        logger.info("Skill '%s' not found in graph", skill_id)
        return False

    await engine.delete_nodes([skill_nid])

    vector_engine = get_vector_engine()
    for field in ["name", "instruction_summary", "description"]:
        collection = f"Skill_{field}"
        try:
            await vector_engine.delete_data_points(collection, [skill_nid])
        except Exception:
            pass

    event = _make_change_event(
        skill_id,
        skill_props.get("name", skill_id),
        "removed",
        old_hash=skill_props.get("content_hash", ""),
    )
    await add_data_points([event])

    logger.info("Removed skill '%s' (node %s)", skill_id, skill_nid)
    return True
