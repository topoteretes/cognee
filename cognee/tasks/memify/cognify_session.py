import cognee

from cognee.exceptions import CogneeValidationError, CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_session")


async def cognify_session(data, dataset_id=None):
    """
    Process and cognify session data into the knowledge graph.

    Adds session content to cognee with a dedicated "user_sessions" node set,
    then triggers the cognify pipeline to extract entities and relationships
    from the session data.

    Args:
        data: Session string containing Question, Context, and Answer information.
        dataset_name: Name of dataset.

    Raises:
        CogneeValidationError: If data is None or empty.
        CogneeSystemError: If cognee operations fail.
    """
    try:
        if not data or (isinstance(data, str) and not data.strip()):
            logger.warning("Empty session data provided to cognify_session task, skipping")
            raise CogneeValidationError(message="Session data cannot be empty", log=False)

        logger.info("Processing session data for cognification")

        await cognee.add(data, dataset_id=dataset_id, node_set=["user_sessions_from_cache"])
        logger.debug("Session data added to cognee with node_set: user_sessions")
        await cognee.cognify(datasets=[dataset_id])
        logger.info("Session data successfully cognified")

    except CogneeValidationError:
        raise
    except Exception as e:
        logger.error(f"Error cognifying session data: {str(e)}")
        raise CogneeSystemError(message=f"Failed to cognify session data: {str(e)}", log=False)
