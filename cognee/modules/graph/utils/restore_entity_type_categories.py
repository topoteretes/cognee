from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import EntityType


async def restore_entity_type_categories(data_points_by_id: dict[str, DataPoint]) -> None:
    """Copy the category already stored for each unclassified EntityType onto its new instance.

    Extraction rebuilds an EntityType for every chunk that mentions it, and a graph
    adapter replaces a node's whole property set on write, so a category assigned
    earlier would be wiped by the next document. This reads the stored category before
    the write and fills only the instances that have none, so a category set in this
    run is never replaced.

    Not atomic with the later write: another document in this same run, or another
    worker, can store a category in between and have it replaced. Closing that needs a
    conditional write in every graph adapter, and a later backfill restores it. The
    Ladybug adapter also returns no nodes when its read fails, which loses the
    categories of the batch the same way.
    """
    unclassified = [
        data_point
        for data_point in data_points_by_id.values()
        if isinstance(data_point, EntityType) and data_point.category is None
    ]
    if not unclassified:
        return

    graph_engine = await get_graph_engine()
    stored_nodes = await graph_engine.get_nodes(
        [str(entity_type.id) for entity_type in unclassified]
    )
    stored_categories = {str(node["id"]): node.get("category") for node in stored_nodes}

    for entity_type in unclassified:
        stored_category = stored_categories.get(str(entity_type.id))
        if stored_category:
            entity_type.category = stored_category
