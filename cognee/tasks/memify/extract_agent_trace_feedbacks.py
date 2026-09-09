import json

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_trace_persist_watermark import (
    TracePersistWindow,
    get_persisted_trace_count,
)
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("extract_agent_trace_feedbacks")


def _normalize_trace_content(value) -> str | None:
    """Convert raw trace content into a non-empty string suitable for memify payloads."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, (bool, int, float)):
        return str(value)

    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    normalized = serialized.strip()
    return normalized or None


async def extract_agent_trace_feedbacks(
    data,
    session_ids: list[str] | None = None,
    raw_trace_content: bool = False,
    last_n_steps: int | None = None,
):
    """
    Extract step-level agent trace content for the current user.

    Retrieves either stored ``session_feedback`` values or raw ``method_return_value``
    values from agent trace sessions and yields one formatted text window per session.
    Only non-empty entries are included.

    With ``last_n_steps=None`` (the ``improve()`` path) the extractor reads the
    session's trace persist watermark (see ``session_trace_persist_watermark``) and
    yields a ``TracePersistWindow`` covering only the steps above it, so re-running
    improve() on an unchanged session does zero ingestion work. The watermark is
    advanced by ``cognify_agent_trace_feedback`` only after the window is cognified.
    An explicit ``last_n_steps`` keeps the legacy "last N steps" behaviour and yields
    a plain string (no watermark involved).

    Args:
        data: Data passed from memify. If empty dict ({}), no external data is provided.
        session_ids: Optional list of specific session IDs to extract.
        raw_trace_content: When True, persist raw ``method_return_value`` values instead
            of ``session_feedback`` summaries.
        last_n_steps: Optional number of most recent trace steps to extract per
            session. When None, all not-yet-persisted steps are used.

    Yields:
        ``TracePersistWindow`` (watermarked path) or a string (``last_n_steps`` path)
        containing the session ID and all non-empty extracted entries.

    Raises:
        CogneeSystemError: If SessionManager is unavailable or extraction fails.
    """
    try:
        if not data or data == [{}]:
            logger.info("Fetching agent trace feedback for current user")

        user: User = session_user.get()
        if not user:
            raise CogneeSystemError(message="No authenticated user found in context", log=False)

        user_id = str(user.id)

        session_manager = get_session_manager()
        if not session_manager.is_available:
            raise CogneeSystemError(
                message=(
                    "SessionManager not available for agent trace feedback extraction, "
                    "please enable caching in order to have sessions to save"
                ),
                log=False,
            )

        if not isinstance(raw_trace_content, bool):
            raise CogneeSystemError(
                message="raw_trace_content must be a boolean",
                log=False,
            )

        if session_ids:
            for session_id in session_ids:
                try:
                    content_label = (
                        "method_return_value" if raw_trace_content else "session_feedback"
                    )
                    if not raw_trace_content:
                        trace_values = await session_manager.get_agent_trace_feedback(
                            user_id=user_id,
                            session_id=session_id,
                            last_n=last_n_steps,
                        )
                    else:
                        trace_session = await session_manager.get_agent_trace_session(
                            user_id=user_id,
                            session_id=session_id,
                            last_n=last_n_steps,
                        )
                        trace_values = [entry.method_return_value for entry in trace_session]
                    trace_values = list(trace_values or [])
                    total_steps = len(trace_values)

                    persisted_count = 0
                    if last_n_steps is None:
                        persisted_count = await get_persisted_trace_count(
                            session_manager, user_id, session_id
                        )
                        if persisted_count > total_steps:
                            # The session shrank below the watermark (cleared and
                            # rebuilt): the watermark is stale, persist everything
                            # currently in the session again.
                            logger.warning(
                                "Session %s has %d trace steps but watermark is %d; "
                                "treating watermark as stale and persisting from the start",
                                session_id,
                                total_steps,
                                persisted_count,
                            )
                            persisted_count = 0
                        trace_values = trace_values[persisted_count:]
                        if not trace_values:
                            logger.info(
                                "Session %s trace steps already persisted up to %d, nothing new",
                                session_id,
                                persisted_count,
                            )
                            continue

                    normalized_trace_values = [
                        normalized
                        for value in trace_values
                        if (normalized := _normalize_trace_content(value)) is not None
                    ]
                    text = f"Session ID: {session_id}\n\n" + "\n".join(normalized_trace_values)
                    if last_n_steps is not None:
                        if normalized_trace_values:
                            logger.info(
                                "Extracted session %s via SessionManager with %d %s entries",
                                session_id,
                                len(normalized_trace_values),
                                content_label,
                            )
                            yield text
                        continue

                    logger.info(
                        "Extracted session %s via SessionManager: %d new of %d trace steps "
                        "(%d non-empty %s entries)",
                        session_id,
                        len(trace_values),
                        total_steps,
                        len(normalized_trace_values),
                        content_label,
                    )
                    # Yield the window even when every new step has empty content:
                    # cognify_agent_trace_feedback then only advances the watermark,
                    # so those steps are not re-read on every later improve().
                    yield TracePersistWindow(
                        user_id=user_id,
                        session_id=session_id,
                        text=text if normalized_trace_values else "",
                        persisted_trace_count=total_steps,
                    )
                except Exception as error:
                    logger.warning(
                        "Failed to extract agent trace %s for session %s: %s",
                        content_label,
                        session_id,
                        error,
                        exc_info=True,
                    )
                    continue
        else:
            logger.info(
                "No specific session_ids provided. Please specify which sessions to extract."
            )

    except CogneeSystemError:
        raise
    except Exception as error:
        logger.exception("Error extracting agent trace feedbacks")
        raise CogneeSystemError(
            message=f"Failed to extract agent trace feedbacks: {error}",
            log=False,
        )
