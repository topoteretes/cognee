from cognee.tasks.storage import add_data_points
from cognee.shared.logging_utils import get_logger
from cognee.modules.retrieval.utils.models import CogneeSearchSentiment
from typing import Optional, List
logger = get_logger()
async def enrichment_task(sentiment_data_points: List[CogneeSearchSentiment]):
    """
    This task takes the sentiment data points and adds them to the graph.
    """
    if sentiment_data_points:
        # Save the sentiment data points to the database
        await add_data_points(data_points=sentiment_data_points, update_edge_collection=False)
        logger.info(f"Enriched the graph with {len(sentiment_data_points)} sentiment data points.")
    else:
        logger.info("No sentiment data points to store.")
