"""Resolve a caller-supplied skill list into concrete Skill objects.

Accepts either Skill instances (used directly) or skill names (looked up in the
graph). Missing names are logged and dropped rather than raised, so an agentic
search with a partially-unknown skill list still runs.

Vector auto-retrieve:
    resolve_skills(..., auto_retrieve_query="user's question", top_k=3)
    runs a vector similarity search over the Skill_description collection and
    merges the hits with any explicit skills the caller supplied (deduplicating
    by skill.id).
"""

from typing import List, Optional, Sequence, Union
from uuid import UUID

from cognee.modules.engine.models import Skill
from cognee.shared.logging_utils import get_logger


logger = get_logger("cognee.tools.resolve_skills")


SKILL_VECTOR_COLLECTION = "Skill_description"


async def resolve_skills(
    skills: Optional[Sequence[Union[str, Skill]]] = None,
    dataset_id: Optional[UUID] = None,
    auto_retrieve_query: Optional[str] = None,
    top_k: int = 3,
) -> List[Skill]:
    """Resolve explicit skills plus optional top-k skills vector-matched to a query."""
    resolved: List[Skill] = []
    seen_ids = set()

    for item in skills or []:
        skill = None
        if isinstance(item, Skill):
            skill = item
        elif isinstance(item, str):
            skill = await _find_skill_by_name(item, dataset_id=dataset_id)
            if skill is None:
                logger.warning("Skill %r not found; skipping", item)
        else:
            logger.warning(
                "Skill entries must be Skill or str; got %s, skipping",
                type(item).__name__,
            )
        if skill is not None and skill.id not in seen_ids:
            resolved.append(skill)
            seen_ids.add(skill.id)

    if auto_retrieve_query:
        retrieved = await _auto_retrieve_skills(auto_retrieve_query, top_k=top_k)
        for skill in retrieved:
            if skill.id in seen_ids:
                continue
            if (
                dataset_id is not None
                and skill.dataset_scope
                and str(dataset_id) not in skill.dataset_scope
            ):
                continue
            resolved.append(skill)
            seen_ids.add(skill.id)

    return resolved


async def _auto_retrieve_skills(query: str, top_k: int) -> List[Skill]:
    """Vector search the Skill_description collection for skills relevant to query."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine
    except Exception:
        return []

    try:
        vector_engine = get_vector_engine()
    except Exception:
        return []

    try:
        results = await vector_engine.search(
            collection_name=SKILL_VECTOR_COLLECTION,
            query_text=query,
            query_vector=None,
            limit=top_k,
            include_payload=True,
        )
    except Exception as exc:
        logger.warning("Skill vector auto-retrieve failed: %s", exc)
        return []

    skills: List[Skill] = []
    for result in results or []:
        payload = getattr(result, "payload", None)
        skill = _coerce_skill(payload) if payload else None
        if skill is None:
            skill = await _find_skill_by_id(getattr(result, "id", None))
        if skill is not None:
            skills.append(skill)
    return skills


async def _find_skill_by_id(skill_id) -> Optional[Skill]:
    """Graph fallback when the vector payload isn't complete enough to rehydrate."""
    if skill_id is None:
        return None
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
    except Exception:
        return None
    try:
        graph_engine = await get_graph_engine()
    except Exception:
        return None
    get_node = getattr(graph_engine, "get_node", None)
    if get_node is None:
        return None
    try:
        node = await get_node(str(skill_id))
    except Exception:
        return None
    return _coerce_skill(node)


async def _find_skill_by_name(name: str, dataset_id: Optional[UUID]) -> Optional[Skill]:
    """Graph lookup for a Skill by name. Uses the nodeset subgraph projection.

    Kuzu / most graph adapters expose `get_nodeset_subgraph(node_type, node_name,
    ...)` which returns `(nodes, edges)` with nodes as `List[Tuple[id, props_dict]]`.
    We iterate props to find the name match, then return a coerced Skill.
    """
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
    except Exception:
        return None

    try:
        graph_engine = await get_graph_engine()
    except Exception:
        return None

    get_nodeset = getattr(graph_engine, "get_nodeset_subgraph", None)
    if get_nodeset is None:
        return None

    try:
        nodes_and_edges = await get_nodeset(node_type=Skill, node_name=[name])
    except Exception:
        return None

    nodes = (nodes_and_edges or [[]])[0]
    for entry in nodes or []:
        props = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else entry
        skill = _coerce_skill(props)
        if skill is None or skill.name != name:
            continue
        if (
            dataset_id is not None
            and skill.dataset_scope
            and str(dataset_id) not in skill.dataset_scope
        ):
            continue
        return skill
    return None


def _coerce_skill(raw) -> Optional[Skill]:
    """Best-effort conversion of a graph-node or vector-payload dict into Skill.

    The graph stores DataPoint.metadata without the "type" key required by the
    MetaData TypedDict (auto-derivation from Annotated markers doesn't populate
    it), so we strip it before validating and let Pydantic re-derive defaults.
    """
    if isinstance(raw, Skill):
        return raw
    data = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(data, dict):
        return None
    data = {k: v for k, v in data.items() if k != "metadata"}
    try:
        return Skill.model_validate(data)
    except Exception:
        return None
