from typing import List, Optional
from uuid import UUID

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.data.methods import get_authorized_dataset
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError, UserNotFoundError
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_api_sdk")


async def _resolve_user(user: Optional[User]) -> User:
    if user is not None:
        if getattr(user, "id", None) is None:
            raise CogneeValidationError(
                message="Session user must have an id.",
                name="SessionPreconditionError",
            )
        return user
    ctx_user = session_user.get()
    if ctx_user is not None and getattr(ctx_user, "id", None) is not None:
        return ctx_user
    try:
        return await get_default_user()
    except (DatabaseNotCreatedError, UserNotFoundError) as error:
        raise CogneeValidationError(
            message=(
                "Session prerequisites not met: no default user found. "
                "Initialize Cognee before using session APIs by running "
                "`await cognee.add(...)` followed by `await cognee.cognify()`."
            ),
            name="SessionPreconditionError",
        ) from error


async def _resolve_session_dataset_id(
    *,
    user: User,
    session_id: str,
    dataset_id: str | UUID | None,
    permission_type: str,
) -> UUID | None:
    """Resolve one canonical dataset scope and enforce its current ACL.

    ``None`` intentionally preserves the legacy owner-only cache namespace.
    Dataset-scoped access is always re-authorized, including inferred scopes,
    so a stale lifecycle row cannot bypass a revoked permission.
    """
    resolved_dataset_id = dataset_id
    if resolved_dataset_id is None:
        try:
            owner_id = UUID(str(user.id))
        except (TypeError, ValueError):
            return None

        from cognee.modules.session_lifecycle.metrics import get_owned_session_dataset_id

        try:
            resolved_dataset_id = await get_owned_session_dataset_id(
                session_id=session_id, user_id=owner_id
            )
        except ValueError as error:
            raise CogneeValidationError(
                message=str(error),
                name="SessionDatasetScopeError",
            ) from error
        except Exception as error:
            logger.debug("Session dataset-scope inference unavailable: %s", error)
            return None

    if resolved_dataset_id is None:
        return None

    try:
        canonical_dataset_id = UUID(str(resolved_dataset_id))
    except (TypeError, ValueError) as error:
        raise CogneeValidationError(
            message="dataset_id must be a valid UUID.",
            name="SessionDatasetScopeError",
        ) from error

    authorized_dataset = await get_authorized_dataset(user, canonical_dataset_id, permission_type)
    if authorized_dataset is None or UUID(str(authorized_dataset.id)) != canonical_dataset_id:
        raise PermissionDeniedError(
            f"Session user does not have [{permission_type}] permission on the requested dataset."
        )
    return canonical_dataset_id


async def get_session(
    session_id: str = "default_session",
    last_n: Optional[int] = None,
    user: Optional[User] = None,
    dataset_id: str | UUID | None = None,
) -> List[SessionQAEntry]:
    resolved_user = await _resolve_user(user)
    user_id = str(resolved_user.id)
    dataset_id = await _resolve_session_dataset_id(
        user=resolved_user,
        session_id=session_id,
        dataset_id=dataset_id,
        permission_type="read",
    )

    try:
        sm = get_session_manager()
        raw = await sm.get_session(
            user_id=user_id,
            session_id=session_id,
            last_n=last_n,
            formatted=False,
            dataset_id=dataset_id,
        )
    except Exception as e:
        logger.warning("get_session: error from SessionManager: %s", e)
        return []

    if not raw:
        return []

    result: List[SessionQAEntry] = []
    for entry in raw:
        if isinstance(entry, dict):
            try:
                result.append(SessionQAEntry.model_validate(entry))
            except Exception as e:
                logger.warning("get_session: skip invalid entry: %s", e)
        elif isinstance(entry, SessionQAEntry):
            result.append(entry)
        else:
            logger.warning("get_session: skip non-dict non-SessionQAEntry entry: %s", type(entry))
    return result


async def add_feedback(
    session_id: str,
    qa_id: str,
    feedback_text: Optional[str] = None,
    feedback_score: Optional[int] = None,
    user: Optional[User] = None,
    dataset_id: str | UUID | None = None,
) -> bool:
    resolved_user = await _resolve_user(user)
    user_id = str(resolved_user.id)
    dataset_id = await _resolve_session_dataset_id(
        user=resolved_user,
        session_id=session_id,
        dataset_id=dataset_id,
        permission_type="write",
    )

    try:
        sm = get_session_manager()
        return await sm.add_feedback(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            feedback_text=feedback_text,
            feedback_score=feedback_score,
            dataset_id=dataset_id,
        )
    except Exception as e:
        logger.warning("add_feedback: error from SessionManager: %s", e)
        return False


async def add_frequency_weights(
    session_id: str,
    qa_id: str,
    node_ids: Optional[list[str]] = None,
    edge_ids: Optional[list[str]] = None,
    user: Optional[User] = None,
    dataset_id: str | UUID | None = None,
) -> bool:
    """Add or update frequency weight data for a QA entry.

    This function stores the graph elements (node_ids and edge_ids) that were used
    in generating the answer for a QA entry. This data is later processed by
    apply_frequency_weights to increment the frequency weights of those elements.

    The frequency_weights_applied flag is reset to False so the entry will be
    reprocessed by the apply_frequency_weights pipeline.

    Args:
        session_id: Session identifier.
        qa_id: QA entry identifier.
        node_ids: List of node IDs used in generating the answer.
        edge_ids: List of edge IDs used in generating the answer.
        user: User that owns the session. If None, uses session/context user or default user.

    Returns:
        True if updated, False if QA not found or cache unavailable.
    """
    from cognee.tasks.memify.frequency_weights_constants import (
        MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY,
    )

    resolved_user = await _resolve_user(user)
    user_id = str(resolved_user.id)
    dataset_id = await _resolve_session_dataset_id(
        user=resolved_user,
        session_id=session_id,
        dataset_id=dataset_id,
        permission_type="write",
    )

    used_graph_element_ids: dict[str, list[str]] = {}
    if node_ids:
        used_graph_element_ids["node_ids"] = node_ids
    if edge_ids:
        used_graph_element_ids["edge_ids"] = edge_ids

    try:
        sm = get_session_manager()
        return await sm.update_qa(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            used_graph_element_ids=used_graph_element_ids if used_graph_element_ids else None,
            memify_metadata={MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY: False},
            dataset_id=dataset_id,
        )
    except Exception as e:
        logger.warning("add_frequency_weights: error from SessionManager: %s", e)
        return False


async def delete_feedback(
    session_id: str,
    qa_id: str,
    user: Optional[User] = None,
    dataset_id: str | UUID | None = None,
) -> bool:
    """
    Clear feedback for a QA entry (sets feedback_text and feedback_score to None).

    When user is None, uses session context or default user.

    Args:
        session_id: Session identifier.
        qa_id: QA entry identifier to clear feedback for.
        user: User that owns the session. If None, uses session/context user or default user.

    Returns:
        True if feedback was cleared, False if QA not found or cache unavailable.
    """
    resolved_user = await _resolve_user(user)
    user_id = str(resolved_user.id)
    dataset_id = await _resolve_session_dataset_id(
        user=resolved_user,
        session_id=session_id,
        dataset_id=dataset_id,
        permission_type="write",
    )

    try:
        sm = get_session_manager()
        return await sm.delete_feedback(
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            dataset_id=dataset_id,
        )
    except Exception as e:
        logger.warning("delete_feedback: error from SessionManager: %s", e)
        return False
