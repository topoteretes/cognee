from typing import Optional, Union
from uuid import UUID

import cognee

from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.session.session_persist_watermark import (
    TracePersistWindow,
    save_persisted_trace_count,
)
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")


async def cognify_agent_trace_feedback(
    data: Union[str, TracePersistWindow],
    dataset_id: Optional[UUID | str] = None,
    node_set_name: str = "agent_trace_feedbacks",
    user: Optional[User] = None,
) -> None:
    """
    Process and cognify agent trace session text into the knowledge graph.

    Args:
        data: Agent trace text for a single session (or a ``TracePersistWindow``
            when the extractor ran in watermark mode). Depending on the extractor
            configuration, the text contains either session feedback summaries or
            raw method return values.
        dataset_id: Dataset identifier to write to.
        node_set_name: Node-set name used when adding the trace text.
        user: User the add/cognify calls run as. Without it they fall back to
            the default user, which has no write ACL on multi-tenant deployments.

    Raises:
        CogneeValidationError: If data is None or empty.
        CogneeSystemError: If cognee operations fail.
    """
    window = data if isinstance(data, TracePersistWindow) else None
    text = window.text if window is not None else data

    try:
        if not text or (isinstance(text, str) and not text.strip()):
            logger.warning(
                "Empty agent trace content provided to cognify_agent_trace_feedback task, skipping"
            )
            raise CogneeValidationError(
                message="Agent trace content cannot be empty",
                log=False,
            )

        logger.info("Processing agent trace content for cognification")

        await cognee.add(text, dataset_id=dataset_id, node_set=[node_set_name], user=user)
        logger.debug(
            "Agent trace content added to cognee with node_set: %s",
            node_set_name,
        )
        await cognee.cognify(datasets=[dataset_id], user=user)
        logger.info("Agent trace content successfully cognified")

    except CogneeValidationError:
        raise
    except Exception as error:
        logger.error("Error cognifying agent trace content: %s", error)
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )

    if window is not None:
        # Advance the persist watermark only now: a failed cognify above leaves it
        # untouched, so the same window retries next run (add-level content-hash
        # dedup makes the retry safe).
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        await save_persisted_trace_count(
            get_session_manager(),
            window.user_id,
            window.session_id,
            persisted_trace_count=window.persisted_trace_count,
        )
