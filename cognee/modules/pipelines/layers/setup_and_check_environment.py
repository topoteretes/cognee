import asyncio
from cognee.context_global_variables import (
    graph_db_config as context_graph_db_config,
    vector_db_config as context_vector_db_config,
)

from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)

_first_run_done = False
_first_run_lock = asyncio.Lock()


async def setup_and_check_environment(
    vector_db_config: dict = None,
    graph_db_config: dict = None,
):
    if vector_db_config:
        context_vector_db_config.set(vector_db_config)
    if graph_db_config:
        context_graph_db_config.set(graph_db_config)

    # Create tables for databases
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    global _first_run_done
    async with _first_run_lock:
        if not _first_run_done:
            from cognee.infrastructure.llm.utils import (
                test_llm_connection,
                test_embedding_connection,
            )

            await test_llm_connection()
            await test_embedding_connection()
            _first_run_done = True
