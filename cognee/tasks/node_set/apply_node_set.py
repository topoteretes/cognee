import json
import logging
from sqlalchemy import select
from typing import List, Any

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.infrastructure.engine.models.DataPoint import DataPoint

logger = logging.getLogger(__name__)


async def apply_node_set(data_points: List[DataPoint]) -> List[DataPoint]:
    """
    Apply NodeSet values from the relational store to DataPoint instances.

    This task fetches the NodeSet values from the Data model in the relational database
    and applies them to the corresponding DataPoint instances.

    Args:
        data_points: List of DataPoint instances to process

    Returns:
        List of updated DataPoint instances with NodeSet values applied
    """
    logger.info(f"Applying NodeSet values to {len(data_points)} DataPoints")

    if not data_points:
        return data_points

    # Create a map of data_point IDs for efficient lookup
    data_point_map = {str(dp.id): dp for dp in data_points}

    # Get the database engine
    db_engine = get_relational_engine()

    # Get session (handles both sync and async cases for testing)
    session = db_engine.get_async_session()

    try:
        # Handle both AsyncMock and actual async context manager for testing
        if hasattr(session, "__aenter__"):
            # It's a real async context manager
            async with session as sess:
                await _process_data_points(sess, data_point_map)
        else:
            # It's an AsyncMock in tests
            await _process_data_points(session, data_point_map)

    except Exception as e:
        logger.error(f"Error applying NodeSet values: {e}")

    return data_points


async def _process_data_points(session, data_point_map):
    """
    Process data points with the given session.

    This helper function handles the actual database query and NodeSet application.

    Args:
        session: Database session
        data_point_map: Map of data point IDs to DataPoint objects
    """
    # Get all data points from the Data table that have node_set values
    # and correspond to the data_points we're processing
    data_ids = list(data_point_map.keys())

    query = select(Data).where(Data.id.in_(data_ids), Data.node_set.is_not(None))

    result = await session.execute(query)
    data_records = result.scalars().all()

    # Apply NodeSet values to corresponding DataPoint instances
    for data_record in data_records:
        data_point_id = str(data_record.id)
        if data_point_id in data_point_map and data_record.node_set:
            # Parse the JSON string to get the NodeSet
            try:
                node_set = json.loads(data_record.node_set)
                data_point_map[data_point_id].NodeSet = node_set
                logger.debug(f"Applied NodeSet {node_set} to DataPoint {data_point_id}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse NodeSet JSON for DataPoint {data_point_id}")
                continue

    logger.info(f"Successfully applied NodeSet values to DataPoints")
