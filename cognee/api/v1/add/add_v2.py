from typing import Union, BinaryIO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines import run_tasks, Task
from cognee.tasks.ingestion import save_data_to_storage, ingest_data
from cognee.infrastructure.databases.relational import create_db_and_tables as create_relational_db_and_tables
from cognee.infrastructure.databases.vector.pgvector import create_db_and_tables as create_pgvector_db_and_tables

async def add(data: Union[BinaryIO, list[BinaryIO], str, list[str]], dataset_name: str = "main_dataset", user: User = None):
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    if user is None:
        user = await get_default_user()

    tasks = [
        Task(save_data_to_storage, dataset_name),
        Task(ingest_data, dataset_name, user)
    ]

    pipeline = run_tasks(tasks, data, "add_pipeline")

    async for result in pipeline:
        print(result)
