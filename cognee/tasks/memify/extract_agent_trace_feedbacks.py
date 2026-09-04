import json
from typing import Optional

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import (
    TRACE_PERSIST_WATERMARK,
    TracePersistWindow,
    get_persisted_trace_count,
    save_persisted_trace_count,
)
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("extract_agent_trace_feedbacks")


def _normalize_trace_content(value) -> Optional[str]:
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


def resolve_trace_window_size(
    total_trace_count: int,
    persisted_trace_count: int,
    last_n_steps: Optional[int],
    *,
    session_id: str = "",
) -> int:
    """How many of the most recent trace steps are not yet persisted.

    The pending window is every step above the watermark. A stale watermark
    (above the current step count: the trace session was cleared and rebuilt)
    restarts from the beginning. An explicit ``last_n_steps`` caps the window
    at the most recent N pending steps — the caller asked for a bounded
    persist — and the watermark still advances to the total, so steps below
    that cap are deliberately left behind, exactly as ``last_n`` did before.
    """
    effective = TRACE_PERSIST_WATERMARK.resolve_effective(
        persisted_trace_count, total_trace_count, session_id=session_id
    )
    pending = max(0, total_trace_count - effective)
    if last_n_steps is not None:
        pending = min(pending, max(0, int(last_n_steps)))
    return pending


async def extract_agent_trace_feedbacks(
    data,
    session_ids: Optional[list[str]] = None,
    raw_trace_content: bool = False,
    last_n_steps: Optional[int] = None,
):
    """
    Extract not-yet-persisted agent trace steps for the current user.

    For each session, reads the trace persist watermark (see
    ``session_persist_watermark.TRACE_PERSIST_WATERMARK``) and yields ONE
    ``TracePersistWindow`` holding the formatted content of the trace steps
    above it — either stored ``session_feedback`` values or raw
    ``method_return_value`` values. A session with no new steps yields nothing,
    so re-running improve() on an unchanged session does zero ingestion work.
    The watermark itself is advanced by ``cognify_agent_trace_feedback`` only
    after the window is successfully cognified.

    Args:
        data: Data passed from memify. If empty dict ({}), no external data is provided.
        session_ids: Optional list of specific session IDs to extract.
        raw_trace_content: When True, persist raw ``method_return_value`` values instead
            of ``session_feedback`` summaries.
        last_n_steps: Optional cap on the number of most recent pending trace
            steps to extract per session. ``None`` means every step above the
            watermark — never "everything stored".

    Yields:
        TracePersistWindow covering the session's unpersisted steps.

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
                content_label = "method_return_value" if raw_trace_content else "session_feedback"
                try:
                    total_trace_count = await session_manager.get_agent_trace_count(
                        user_id=user_id, session_id=session_id
                    )
                    if not total_trace_count:
                        continue

                    persisted_count = await get_persisted_trace_count(
                        session_manager, user_id, session_id
                    )
                    window_size = resolve_trace_window_size(
                        total_trace_count,
                        persisted_count,
                        last_n_steps,
                        session_id=session_id,
                    )
                    if window_size <= 0:
                        logger.info(
                            "Session %s trace steps already persisted up to %d, nothing new",
                            session_id,
                            persisted_count,
                        )
                        continue

                    if not raw_trace_content:
                        trace_values = await session_manager.get_agent_trace_feedback(
                            user_id=user_id,
                            session_id=session_id,
                            last_n=window_size,
                        )
                    else:
                        trace_session = await session_manager.get_agent_trace_session(
                            user_id=user_id,
                            session_id=session_id,
                            last_n=window_size,
                        )
                        trace_values = [entry.method_return_value for entry in trace_session]

                    normalized_trace_values = [
                        normalized
                        for value in trace_values
                        if (normalized := _normalize_trace_content(value)) is not None
                    ]
                    if not normalized_trace_values:
                        # Nothing worth cognifying in this window (steps without
                        # feedback text). Mark it done so it is not re-read forever;
                        # there is no cognify whose success the advance could wait on.
                        await save_persisted_trace_count(
                            session_manager, user_id, session_id, total_trace_count
                        )
                        logger.info(
                            "Session %s: %d pending trace steps carry no %s; watermark "
                            "advanced to %d without ingestion",
                            session_id,
                            window_size,
                            content_label,
                            total_trace_count,
                        )
                        continue

                    logger.info(
                        "Extracted session %s via SessionManager: %d %s entries "
                        "(%d new of %d trace steps)",
                        session_id,
                        len(normalized_trace_values),
                        content_label,
                        window_size,
                        total_trace_count,
                    )
                    yield TracePersistWindow(
                        user_id=user_id,
                        session_id=session_id,
                        text=f"Session ID: {session_id}\n\n" + "\n".join(normalized_trace_values),
                        persisted_trace_count=total_trace_count,
                    )
                except Exception as error:
                    logger.warning(
                        "Failed to extract agent trace %s for session %s: %s",
                        content_label,
                        session_id,
                        error,
                    )
                    continue
        else:
            logger.info(
                "No specific session_ids provided. Please specify which sessions to extract."
            )

    except CogneeSystemError:
        raise
    except Exception as error:
        logger.error("Error extracting agent trace feedbacks: %s", error)
        raise CogneeSystemError(
            message=f"Failed to extract agent trace feedbacks: {error}",
            log=False,
        )
