from typing import Optional, Union
from uuid import UUID

import cognee

from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.modules.improve.constants import AGENT_TRACE_FEEDBACKS_NODE_SET
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import (
    TracePersistWindow,
    save_persisted_trace_count,
)
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")

TraceContent = Union[TracePersistWindow, str]


def _coerce_trace_items(data) -> list[TraceContent]:
    """Accept one window, one string, or a batch of either; drop empty items."""
    items = data if isinstance(data, list) else [data]
    valid: list[TraceContent] = []
    for item in items:
        text = item.text if isinstance(item, TracePersistWindow) else item
        if isinstance(text, str) and text.strip():
            valid.append(item)
    return valid


async def cognify_agent_trace_feedback(
    data: Union[TraceContent, list[TraceContent]],
    dataset_id: Optional[UUID | str] = None,
    node_set_name: str = AGENT_TRACE_FEEDBACKS_NODE_SET,
    user: Optional[User] = None,
) -> None:
    """
    Cognify agent trace windows into the knowledge graph and advance their watermarks.

    Receives one ``TracePersistWindow`` (or a batch of them — the pipeline
    runner delivers generator output in batches) from
    ``extract_agent_trace_feedbacks``. For each window: adds its text to cognee
    under ``node_set_name``, triggers cognify, and — only after both succeed —
    advances that session's trace persist watermark to the step count captured
    at extraction time. On failure the watermark stays put, so the same window
    is re-extracted and retried on the next improve() (add-level content-hash
    dedup makes the retry safe).

    Plain strings are still accepted for callers that feed their own trace text
    through the public ``memify`` task registry; they carry no watermark.

    Args:
        data: Window(s) yielded by ``extract_agent_trace_feedbacks`` (or raw text).
        dataset_id: Dataset identifier to write to.
        node_set_name: Node-set name used when adding the trace text.
        user: User the add/cognify calls run as. Without it they fall back to
            the default user, which has no write ACL on multi-tenant deployments.

    Raises:
        CogneeValidationError: If no non-empty window or text was provided.
        CogneeSystemError: If cognee operations fail.
    """
    items = _coerce_trace_items(data)
    if not items:
        logger.warning(
            "Empty agent trace content provided to cognify_agent_trace_feedback task, skipping"
        )
        raise CogneeValidationError(
            message="Agent trace content cannot be empty",
            log=False,
        )

    try:
        for item in items:
            window = item if isinstance(item, TracePersistWindow) else None
            text = window.text if window is not None else item

            logger.info("Processing agent trace content for cognification")
            await cognee.add(text, dataset_id=dataset_id, node_set=[node_set_name], user=user)
            logger.debug(
                "Agent trace content added to cognee with node_set: %s",
                node_set_name,
            )
            # raise_on_error=False: one trace session's failed build must not kill
            # the whole memify run — log the cause, keep this window's watermark
            # put (so it is re-extracted and retried on the next improve()), and
            # let the remaining sessions proceed.
            cognify_result = await cognee.cognify(
                datasets=[dataset_id], user=user, raise_on_error=False
            )
            errored_run = get_errored_run_info(cognify_result)
            if errored_run is not None:
                logger.error(
                    "Cognify failed for agent trace content (%s: %s); watermark not advanced, "
                    "continuing with the run",
                    errored_run.error_class,
                    errored_run.error_message,
                )
                continue
            logger.info("Agent trace content successfully cognified")

            if window is None:
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
        logger.error("Error cognifying agent trace content: %s", error)
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )
