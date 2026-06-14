"""List dataset-scoped Skill nodes for UI consumption.

Reads the knowledge graph scoped to one dataset and returns the Skill nodes
with their publisher/provenance metadata. Mirrors the scoping approach used by
``get_schema_inventory`` (``set_database_global_context_variables``) so the graph
databases are resolved for the dataset's owner before the read.
"""

from typing import Any
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

# Node ``type`` property that identifies a Skill node in the graph.
SKILL_NODE_TYPE = "Skill"


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [_as_str(value)] if _as_str(value) else []


def _skill_from_props(node_id: str, props: dict[str, Any]) -> dict[str, Any]:
    """Project a graph node's properties onto the UI skill shape."""
    return {
        "id": _as_str(props.get("id")) or _as_str(node_id),
        "name": _as_str(props.get("name")),
        "description": _as_str(props.get("description")),
        "maintainer": _as_str(props.get("maintainer")),
        "maintainer_url": _as_str(props.get("maintainer_url")),
        "version": _as_str(props.get("skill_version")),
        "tags": _as_str_list(props.get("tags")),
        "license": _as_str(props.get("license")),
        "declared_tools": _as_str_list(props.get("declared_tools")),
        "dataset_scope": _as_str_list(props.get("dataset_scope")),
        "is_active": bool(props.get("is_active", True)),
        "source_repo_url": _as_str(props.get("source_repo_url")),
        "source_dir": _as_str(props.get("source_dir")),
    }


async def list_skills(
    dataset: str | UUID | None = None,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    """Return the Skill nodes for an authorized dataset.

    Parameters:
        dataset: dataset id/name used to scope the graph databases.
        include_inactive: when False (default) only ``is_active`` skills are
            returned.

    Returns:
        A list of skill dicts ordered by name, each carrying publisher metadata
        (``maintainer``, ``maintainer_url``, ``version``, ``tags``, ``license``)
        alongside ``declared_tools`` and ``is_active``.
    """
    if dataset is not None:
        owner_id = await _resolve_dataset_owner(dataset)
        if owner_id is not None:
            async with set_database_global_context_variables(dataset, owner_id):
                return await _collect_skills(dataset, include_inactive)
    return await _collect_skills(dataset, include_inactive)


async def _resolve_dataset_owner(dataset: str | UUID) -> UUID | None:
    """Return the owner id for a dataset, or None when it cannot be resolved."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset

    if not isinstance(dataset, UUID):
        return None

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        record = await session.get(Dataset, dataset)
        return record.owner_id if record else None


async def _collect_skills(
    dataset: str | UUID | None,
    include_inactive: bool,
) -> list[dict[str, Any]]:
    graph_engine = await get_graph_engine()
    nodes, _ = await graph_engine.get_graph_data()

    dataset_id = str(dataset) if dataset is not None else None

    skills: list[dict[str, Any]] = []
    for node_id, props in nodes:
        if not isinstance(props, dict) or props.get("type") != SKILL_NODE_TYPE:
            continue
        skill = _skill_from_props(node_id, props)
        if not include_inactive and not skill["is_active"]:
            continue
        # When scoping resolved a real dataset id, keep only skills declaring it.
        # Skills predating dataset_scope (empty list) are kept to avoid hiding data.
        scope = skill["dataset_scope"]
        if dataset_id is not None and scope and dataset_id not in scope:
            continue
        skills.append(skill)

    skills.sort(key=lambda s: (s["name"] or "").lower())
    return skills
