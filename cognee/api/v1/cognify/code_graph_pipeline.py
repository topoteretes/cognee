import asyncio
import logging
from pathlib import Path

from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, MonitoringTool
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.ingestion import ingest_data_with_metadata
from cognee.tasks.repo_processor import (
    enrich_dependency_graph,
    expand_dependency_graph,
    get_data_list_for_user,
    get_non_py_files,
    get_repo_file_dependencies,
)
from cognee.tasks.repo_processor.get_source_code_chunks import get_source_code_chunks
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_code, summarize_text

monitoring = get_base_config().monitoring_tool
if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


logger = logging.getLogger("code_graph_pipeline")
update_status_lock = asyncio.Lock()


@observe
async def run_code_graph_pipeline(repo_path, include_docs=True):
    import os
    import pathlib

    import cognee
    from cognee.infrastructure.databases.relational import create_db_and_tables

    file_path = Path(__file__).parent
    data_directory_path = str(
        pathlib.Path(os.path.join(file_path, ".data_storage/code_graph")).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(os.path.join(file_path, ".cognee_system/code_graph")).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_db_and_tables()

    cognee_config = get_cognify_config()
    user = await get_default_user()

    tasks = [
        Task(get_repo_file_dependencies),
        Task(enrich_dependency_graph),
        Task(expand_dependency_graph, task_config={"batch_size": 50}),
        Task(get_source_code_chunks, task_config={"batch_size": 50}),
        Task(summarize_code, task_config={"batch_size": 50}),
        Task(add_data_points, task_config={"batch_size": 50}),
    ]

    if include_docs:
        non_code_tasks = [
            Task(get_non_py_files, task_config={"batch_size": 50}),
            Task(ingest_data_with_metadata, dataset_name="repo_docs", user=user),
            Task(get_data_list_for_user, dataset_name="repo_docs", user=user),
            Task(classify_documents),
            Task(extract_chunks_from_documents, max_tokens=cognee_config.max_tokens),
            Task(
                extract_graph_from_data, graph_model=KnowledgeGraph, task_config={"batch_size": 50}
            ),
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config={"batch_size": 50},
            ),
        ]

    if include_docs:
        async for result in run_tasks(non_code_tasks, repo_path):
            yield result

    async for result in run_tasks(tasks, repo_path, "cognify_code_pipeline"):
        yield result
