from typing import Optional
from uuid import UUID

import cognee

from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")


async def cognify_agent_trace_feedback(
    data: str,
    dataset_id: Optional[UUID | str] = None,
    node_set_name: str = "agent_trace_feedbacks",
) -> None:
    """
    Process and cognify agent trace session text into the knowledge graph.

    Args:
        data: Agent trace text for a single session. Depending on the extractor
            configuration, this may contain either session feedback summaries or
            raw method return values.
        dataset_id: Dataset identifier to write to.
        node_set_name: Node-set name used when adding the trace text.

    Raises:
        CogneeValidationError: If data is None or empty.
        CogneeSystemError: If cognee operations fail.
    """
    try:
        if not data or (isinstance(data, str) and not data.strip()):
            logger.warning(
                "Empty agent trace content provided to cognify_agent_trace_feedback task, skipping"
            )
            raise CogneeValidationError(
                message="Agent trace content cannot be empty",
                log=False,
            )

        logger.info("Processing agent trace content for cognification")

        await cognee.add(data, dataset_id=dataset_id, node_set=[node_set_name])
        logger.debug(
            "Agent trace content added to cognee with node_set: %s",
            node_set_name,
        )
        await cognee.cognify(datasets=[dataset_id])
        logger.info("Agent trace content successfully cognified")

    except CogneeValidationError:
        raise
    except Exception as error:
        logger.error("Error cognifying agent trace content: %s", error)
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )
