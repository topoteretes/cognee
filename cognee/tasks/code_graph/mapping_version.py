"""Report code graph mappings that completed items would otherwise skip."""

from cognee.shared.logging_utils import get_logger

logger = get_logger("code_graph")


async def warn_outdated_code_mapping(items, dataset, user):
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.cognify.routing import CognifyRoute, cognify_route_for
    from cognee.tasks.code_graph.extract_code_graph import CODE_GRAPH_MAPPING_VERSION

    if not any(
        cognify_route_for(item) in {CognifyRoute.CODE, CognifyRoute.CODE_REPO} for item in items
    ):
        return
    async with set_database_global_context_variables(dataset.id, user.id):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_filtered_graph_data([{"type": ["CodeRepository"]}])
    outdated = [
        props.get("name")
        for _, props in nodes
        if props.get("last_mapping_version") is None
        or int(props["last_mapping_version"]) < CODE_GRAPH_MAPPING_VERSION
    ]
    if outdated:
        logger.warning(
            "Dataset %s has code repositories %s on an older graph mapping. Reindex them with "
            "cognify(datasets=[%r], data_ids=[<repository data id>], incremental_loading=False, "
            "data_cache=False); a remote repository must be added again first if its cached "
            "checkout is gone. Unmarked edges of the old mapping that the snapshot still derives "
            "are marked by that run.",
            dataset.id,
            outdated,
            str(dataset.id),
        )
