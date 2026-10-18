import asyncio
import os

from cognee.context_global_variables import (
    graph_db_config as context_graph_db_config,
)
from cognee.context_global_variables import (
    vector_db_config as context_vector_db_config,
)
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Each probe is cached on its own: a run that needed no LLM must not mark the
# LLM "checked" for the runs that do, so mixed LLM-free/LLM use in one process
# stays order-independent.
_llm_checked = False
_embeddings_checked = False
_first_run_lock = asyncio.Lock()


async def setup_and_check_environment(
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    skip_connection_test: bool = False,
    needs_llm: bool = True,
):
    """Create database tables and run the first-use connection probes.

    ``needs_llm`` says whether the pipeline ahead contains an LLM task — the
    caller derives it from the tasks themselves (the ``Task.needs_llm`` union,
    ``pipeline_needs_llm``) and this layer never reads cognify config itself.
    A skipped-because-not-needed probe is never recorded as done.
    """
    if vector_db_config:
        context_vector_db_config.set(vector_db_config)
    if graph_db_config:
        context_graph_db_config.set(graph_db_config)

    # Create tables for databases
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    global _llm_checked, _embeddings_checked
    async with _first_run_lock:
        llm_pending = needs_llm and not _llm_checked
        embeddings_pending = not _embeddings_checked
        if not llm_pending and not embeddings_pending:
            return

        skip_test = os.getenv("COGNEE_SKIP_CONNECTION_TEST", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        if skip_test:
            # Process-wide opt-out: never probe in this process.
            logger.info(
                "Skipping LLM/embedding connection tests (COGNEE_SKIP_CONNECTION_TEST is set)."
            )
            _llm_checked = True
            _embeddings_checked = True
            return
        if skip_connection_test:
            # Caller-scoped skip (e.g. the LLM-free code graph pipeline).
            # Marks nothing done: a later pipeline that does need
            # LLM/embeddings must still run the checks.
            logger.info(
                "Skipping LLM/embedding connection tests for this pipeline "
                "(requested by the caller)."
            )
            return

        from cognee.infrastructure.llm.utils import (
            determine_embedding_dimensions,
            test_embedding_connection,
            test_llm_connection,
        )

        if llm_pending:
            await test_llm_connection()
            _llm_checked = True
        elif not needs_llm and not _llm_checked:
            logger.info("Skipping LLM connection test: this pipeline has no LLM task.")
        if embeddings_pending:
            detected_embedding_dimensions = await test_embedding_connection()
            await determine_embedding_dimensions(detected_embedding_dimensions)
            _embeddings_checked = True
