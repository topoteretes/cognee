from typing import Union, BinaryIO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines import run_tasks, Task
from cognee.tasks.ingestion import ingest_data, resolve_data_directories
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from uuid import uuid5, NAMESPACE_OID


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
):
    # Create tables for databases
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    # Initialize first_run attribute if it doesn't exist
    if not hasattr(add, "first_run"):
        add.first_run = True

    if add.first_run:
        from cognee.infrastructure.llm.utils import test_llm_connection, test_embedding_connection

        # Test LLM and Embedding configuration once before running Cognee
        await test_llm_connection()
        await test_embedding_connection()
        add.first_run = False  # Update flag after first run

    if user is None:
        user = await get_default_user()

    tasks = [Task(resolve_data_directories), Task(ingest_data, dataset_name, user)]

    dataset_id = uuid5(NAMESPACE_OID, dataset_name)
    pipeline = run_tasks(
        tasks=tasks, dataset_id=dataset_id, data=data, pipeline_name="add_pipeline"
    )

    async for result in pipeline:
        print(result)
