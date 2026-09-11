from uuid import UUID

import cognee
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.session.agent_trace_persist_watermark import (
    AgentTracePersistWindow,
    save_persisted_trace_count,
)
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")


async def cognify_agent_trace_feedback(
    data: str | AgentTracePersistWindow | list[str | AgentTracePersistWindow],
    dataset_id: UUID | str | None = None,
    node_set_name: str = "agent_trace_feedbacks",
    user: User | None = None,
) -> None:
    """
    Process and cognify agent trace session text into the knowledge graph.

    Args:
        data: Agent trace text or incremental trace window(s). Depending on the
            extractor configuration, content may contain either session feedback
            summaries or raw method return values.
        dataset_id: Dataset identifier to write to.
        node_set_name: Node-set name used when adding the trace text.
        user: User the add/cognify calls run as. Without it they fall back to
            the default user, which has no write ACL on multi-tenant deployments.

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

        items = data if isinstance(data, list) else [data]
        if not all(isinstance(item, (str, AgentTracePersistWindow)) for item in items):
            raise CogneeValidationError(
                message="Agent trace content must be text or a persistence window",
                log=False,
            )

        for item in items:
            if isinstance(item, AgentTracePersistWindow):
                window = item
                trace_content = item.text
            else:
                window = None
                trace_content = item

            if not trace_content.strip():
                if window is None:
                    raise CogneeValidationError(
                        message="Agent trace content cannot be empty",
                        log=False,
                    )
                await save_persisted_trace_count(
                    get_session_manager(),
                    user_id=window.user_id,
                    session_id=window.session_id,
                    persisted_trace_count=window.persisted_trace_count,
                )
                continue

            logger.info("Processing agent trace content for cognification")
            await cognee.add(
                trace_content,
                dataset_id=dataset_id,
                node_set=[node_set_name],
                user=user,
            )
            logger.debug(
                "Agent trace content added to cognee with node_set: %s",
                node_set_name,
            )
            # Keep a failed window pending so a later improve() can retry it.
            cognify_result = await cognee.cognify(
                datasets=[dataset_id], user=user, raise_on_error=False
            )
            errored_run = get_errored_run_info(cognify_result)
            if errored_run is not None:
                logger.error(
                    "Cognify failed for agent trace content (%s: %s); continuing with the run",
                    errored_run.error_class,
                    errored_run.error_message,
                )
                continue
            logger.info("Agent trace content successfully cognified")

            if window is not None:
                await save_persisted_trace_count(
                    get_session_manager(),
                    user_id=window.user_id,
                    session_id=window.session_id,
                    persisted_trace_count=window.persisted_trace_count,
                )

    except CogneeValidationError:
        raise
    except Exception as error:
        logger.exception("Error cognifying agent trace content")
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )
