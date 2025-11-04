from typing import Optional, List

import cognee
from cognee import memify
from cognee.context_global_variables import (
    session_user,
    set_database_global_context_variables,
    set_session_user_context_variable,
)
from cognee.exceptions import CogneeValidationError, CogneeSystemError
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User

logger = get_logger("persist_sessions_in_knowledge_graph")


async def extract_user_sessions(
    data,
    session_ids: Optional[List[str]] = None,
):
    """
    Extract Q&A sessions for the current user from cache.

    Retrieves all Q&A triplets from specified session IDs and yields them
    as formatted strings combining question, context, and answer.

    Args:
        data: Data passed from memify. If empty dict ({}), no external data is provided.
        session_ids: Optional list of specific session IDs to extract.

    Yields:
        String containing session ID and all Q&A pairs formatted.

    Raises:
        CogneeSystemError: If cache engine is unavailable or extraction fails.
    """
    try:
        if not data or data == [{}]:
            logger.info("Fetching session metadata for current user")

        user: User = session_user.get()
        if not user:
            raise CogneeSystemError(message="No authenticated user found in context", log=False)

        user_id = str(user.id)

        cache_engine = get_cache_engine()
        if cache_engine is None:
            raise CogneeSystemError(
                message="Cache engine not available for session extraction, please enable caching in order to have sessions to save",
                log=False,
            )

        if session_ids:
            for session_id in session_ids:
                try:
                    qa_data = await cache_engine.get_all_qas(user_id, session_id)
                    if qa_data:
                        logger.info(f"Extracted session {session_id} with {len(qa_data)} Q&A pairs")
                        session_string = f"Session ID: {session_id}\n\n"
                        for qa_pair in qa_data:
                            question = qa_pair.get("question", "")
                            answer = qa_pair.get("answer", "")
                            session_string += f"Question: {question}\n\nAnswer: {answer}\n\n"
                        yield session_string
                except Exception as e:
                    logger.warning(f"Failed to extract session {session_id}: {str(e)}")
                    continue
        else:
            logger.info(
                "No specific session_ids provided. Please specify which sessions to extract."
            )

    except CogneeSystemError:
        raise
    except Exception as e:
        logger.error(f"Error extracting user sessions: {str(e)}")
        raise CogneeSystemError(message=f"Failed to extract user sessions: {str(e)}", log=False)


async def cognify_session(data):
    """
    Process and cognify session data into the knowledge graph.

    Adds session content to cognee with a dedicated "user_sessions" node set,
    then triggers the cognify pipeline to extract entities and relationships
    from the session data.

    Args:
        data: Session string containing Question, Context, and Answer information.

    Raises:
        CogneeValidationError: If data is None or empty.
        CogneeSystemError: If cognee operations fail.
    """
    try:
        if not data or (isinstance(data, str) and not data.strip()):
            logger.warning("Empty session data provided to cognify_session task, skipping")
            raise CogneeValidationError(message="Session data cannot be empty", log=False)

        logger.info("Processing session data for cognification")

        await cognee.add(data, node_set=["user_sessions"])
        logger.debug("Session data added to cognee with node_set: user_sessions")
        await cognee.cognify()
        logger.info("Session data successfully cognified")

    except CogneeValidationError:
        raise
    except Exception as e:
        logger.error(f"Error cognifying session data: {str(e)}")
        raise CogneeSystemError(message=f"Failed to cognify session data: {str(e)}", log=False)


async def persist_sessions_in_knowledge_graph_pipeline(
    user: User,
    session_ids: Optional[List[str]] = None,
    dataset: str = "main_dataset",
    run_in_background: bool = False,
):
    await set_session_user_context_variable(user)
    dataset_to_write = await get_authorized_existing_datasets(
        user=user, datasets=[dataset], permission_type="write"
    )

    if not dataset_to_write:
        raise CogneeValidationError(
            message=f"User does not have write access to dataset: {dataset}", log=False
        )

    await set_database_global_context_variables(
        dataset_to_write[0].id, dataset_to_write[0].owner_id
    )

    extraction_tasks = [Task(extract_user_sessions, session_ids=session_ids)]

    enrichment_tasks = [
        Task(cognify_session),
    ]

    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset_to_write[0].id,
        data=[{}],
        run_in_background=run_in_background,
    )

    logger.info("Session persistence pipeline completed")
    return result
