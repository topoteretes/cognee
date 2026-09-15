from uuid import UUID

import cognee
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_trace_persist_watermark import (
    TracePersistWindow,
    save_persisted_trace_count,
)
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")


def _as_items(data) -> list:
    return data if isinstance(data, list) else [data]


async def _cognify_text(
    text: str, dataset_id: UUID | str | None, node_set_name: str, user: User | None
) -> bool:
    """Add + cognify one trace text; True iff the cognify run did not error."""
    await cognee.add(text, dataset_id=dataset_id, node_set=[node_set_name], user=user)
    logger.debug("Agent trace content added to cognee with node_set: %s", node_set_name)
    # raise_on_error=False: one trace session's failed build must not kill
    # the whole memify run — log the cause and let the remaining sessions
    # proceed (the pre-loud-cognify behavior, now with the error visible).
    cognify_result = await cognee.cognify(datasets=[dataset_id], user=user, raise_on_error=False)
    errored_run = get_errored_run_info(cognify_result)
    if errored_run is not None:
        logger.error(
            "Cognify failed for agent trace content (%s: %s); continuing with the run",
            errored_run.error_class,
            errored_run.error_message,
        )
        return False
    logger.info("Agent trace content successfully cognified")
    return True


async def cognify_agent_trace_feedback(
    data: str | TracePersistWindow | list[str | TracePersistWindow],
    dataset_id: UUID | str | None = None,
    node_set_name: str = "agent_trace_feedbacks",
    user: User | None = None,
) -> None:
    """
    Process and cognify agent trace session text into the knowledge graph.

    Receives plain strings (legacy ``last_n_steps`` extraction) or
    ``TracePersistWindow`` items (watermarked extraction) from
    ``extract_agent_trace_feedbacks`` — the pipeline runner may deliver them in
    batches. For a window: add + cognify its text and, only after both succeed,
    advance that session's trace persist watermark to the step count captured at
    extraction time. A window with empty text (every new step had no content)
    only advances the watermark. On failure the watermark stays put, so the same
    window is re-extracted on the next improve().

    Args:
        data: Agent trace text or window(s) for single sessions. Depending on the
            extractor configuration, the text contains either session feedback
            summaries or raw method return values.
        dataset_id: Dataset identifier to write to.
        node_set_name: Node-set name used when adding the trace text.
        user: User the add/cognify calls run as. Without it they fall back to
            the default user, which has no write ACL on multi-tenant deployments.

    Raises:
        CogneeValidationError: If no valid, non-empty item was provided.
        CogneeSystemError: If cognee operations fail.
    """
    items = [
        item
        for item in _as_items(data)
        if isinstance(item, TracePersistWindow) or (isinstance(item, str) and item.strip())
    ]
    if not items:
        logger.warning(
            "Empty agent trace content provided to cognify_agent_trace_feedback task, skipping"
        )
        raise CogneeValidationError(message="Agent trace content cannot be empty", log=False)

    try:
        for item in items:
            if isinstance(item, str):
                logger.info("Processing agent trace content for cognification")
                await _cognify_text(item, dataset_id, node_set_name, user)
                continue

            window = item
            if window.text.strip():
                logger.info(
                    "Processing session %s trace window (%d steps persisted after this)",
                    window.session_id,
                    window.persisted_trace_count,
                )
                if not await _cognify_text(window.text, dataset_id, node_set_name, user):
                    logger.error(
                        "Session %s trace watermark not advanced; window will be retried "
                        "on the next improve()",
                        window.session_id,
                    )
                    continue

            await save_persisted_trace_count(
                get_session_manager(),
                user_id=window.user_id,
                session_id=window.session_id,
                persisted_trace_count=window.persisted_trace_count,
            )
            logger.info(
                "Session %s trace persist watermark advanced to %d",
                window.session_id,
                window.persisted_trace_count,
            )

    except CogneeValidationError:
        raise
    except Exception as error:
        logger.exception("Error cognifying agent trace content")
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )
