from uuid import UUID

import cognee
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.llm.exceptions import raise_if_budget_exhausted
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_node_set import bridge_node_set
from cognee.infrastructure.session.session_persist_watermark import (
    TRACE_PERSIST_WATERMARK,
    TracePersistWindow,
)
from cognee.modules.improve.constants import AGENT_TRACE_FEEDBACKS_NODE_SET
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")

TraceContent = TracePersistWindow | str


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
    data: TraceContent | list[TraceContent],
    dataset_id: UUID | str | None = None,
    node_set_name: str = AGENT_TRACE_FEEDBACKS_NODE_SET,
    user: User | None = None,
) -> None:
    """
    Cognify agent trace windows into the knowledge graph and advance their watermarks.

    Receives one ``TracePersistWindow`` (or a batch of them — the pipeline
    runner delivers every window of the run in one call)
    from ``extract_agent_trace_feedbacks``. Adds every window's text to cognee
    under ``node_set_name``, then runs ONE cognify for the dataset: the run
    builds the windows as independent documents, concurrently, so a batch
    costs about one window of LLM time instead of one per window. Only after
    that run completed does each window's session advance its trace persist
    watermark to the step count captured at extraction time. A failed run is
    rolled back as a whole and raises ``CognifyFailedError`` with the root
    cause, which fails the memify run: no watermark moves, every window is
    re-extracted and retried on the next improve() (add-level content-hash
    dedup makes the retry safe), and improve reports the cause.

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
        LLMPaymentRequiredError: If the build hit an exhausted LLM budget.
        CognifyFailedError: If the build failed, with the root cause.
        CogneeSystemError: If another cognee operation fails.
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
        windows: list[TracePersistWindow] = []
        for item in items:
            window = item if isinstance(item, TracePersistWindow) else None
            text = window.text if window is not None else item

            logger.info("Staging agent trace content for cognification")
            # Plain-text callers carry no session, so no pinned node set.
            node_set = bridge_node_set([node_set_name], window.node_set if window else ())
            await cognee.add(text, dataset_id=dataset_id, node_set=node_set, user=user)
            logger.debug("Agent trace content added to cognee with node_set: %s", node_set)
            if window is not None:
                windows.append(window)

        # One cognify for the whole batch: the pipeline builds the staged
        # documents concurrently. A failed build raises with its root cause and
        # fails the memify run; the run was rolled back, so no watermark moves.
        # A budget rejection is raised as LLMPaymentRequiredError: every later
        # window would fail the same way, so improve stops the run.
        try:
            await cognee.cognify(datasets=[dataset_id], user=user)
        except Exception as error:
            raise_if_budget_exhausted(error)
            raise
        logger.info("Agent trace content successfully cognified (%d window(s))", len(items))

        for window in windows:
            await TRACE_PERSIST_WATERMARK.write_count(
                get_session_manager(),
                user_id=window.user_id,
                session_id=window.session_id,
                count=window.persisted_trace_count,
            )
            logger.info(
                "Session %s trace persist watermark advanced to %d",
                window.session_id,
                window.persisted_trace_count,
            )

    except (CogneeValidationError, CogneeSystemError):
        # cognee's own typed errors (a 402, a CognifyFailedError with its cause)
        # pass through unwrapped.
        raise
    except Exception as error:
        logger.exception("Error cognifying agent trace content")
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )
