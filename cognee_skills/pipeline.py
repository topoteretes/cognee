"""Skills ingestion pipeline: parse -> enrich -> materialize patterns -> add to graph."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid5, UUID

from cognee.low_level import setup
from cognee.pipelines import Task, run_tasks
from cognee.tasks.storage import add_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import load_or_create_datasets
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.utils.generate_timestamp_datapoint import generate_timestamp_datapoint
from cognee.modules.engine.models.Timestamp import Timestamp

from cognee_skills.tasks.parse_skills import parse_skills_task
from cognee_skills.tasks.enrich_skills import enrich_skills
from cognee_skills.tasks.materialize_task_patterns import materialize_task_patterns
from cognee_skills.tasks.apply_node_set import apply_node_set
from cognee_skills.models.skill_change_event import SkillChangeEvent

logger = logging.getLogger(__name__)

EVENT_NAMESPACE = UUID("d4e5f6a7-b8c9-0123-def0-123456789abc")


def _now_timestamp() -> Timestamp:
    """Create a Cognee Timestamp DataPoint for the current UTC time."""
    now = datetime.now(timezone.utc)
    raw = Timestamp(
        id=UUID(int=0),
        time_at=0,
        year=now.year,
        month=now.month,
        day=now.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
        timestamp_str="",
    )
    return generate_timestamp_datapoint(raw)


def _make_change_event(
    skill_id: str,
    skill_name: str,
    change_type: str,
    old_hash: str = "",
    new_hash: str = "",
) -> SkillChangeEvent:
    ts = _now_timestamp()
    return SkillChangeEvent(
        id=uuid5(EVENT_NAMESPACE, f"{skill_id}:{change_type}:{ts.time_at}"),
        name=f"skill_{change_type}: {skill_name}",
        description=f"Skill '{skill_name}' ({skill_id}) was {change_type}.",
        skill_id=skill_id,
        change_type=change_type,
        old_content_hash=old_hash,
        new_content_hash=new_hash,
        skill_name=skill_name,
        at=ts,
    )


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
    from cognee_skills.parser.skill_parser import parse_skills_folder

    skills_folder = str(Path(skills_folder).resolve())
    await setup()

    new_skills = parse_skills_folder(skills_folder, source_repo=source_repo)
    new_by_id = {s.skill_id: s for s in new_skills}

    engine = await get_graph_engine()
    raw_nodes, _ = await engine.get_graph_data()
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
