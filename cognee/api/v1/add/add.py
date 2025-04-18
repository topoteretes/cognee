from typing import Union, BinaryIO, List, Optional
from cognee.modules.users.models import User
from cognee.modules.pipelines import Task
from cognee.tasks.ingestion import ingest_data, resolve_data_directories
from cognee.modules.pipelines import cognee_pipeline


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
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

    tasks = [Task(resolve_data_directories), Task(ingest_data, dataset_name, user, node_set)]


    await cognee_pipeline(
        tasks=tasks, datasets=dataset_name, data=data, user=user, pipeline_name="add_pipeline"
    )
