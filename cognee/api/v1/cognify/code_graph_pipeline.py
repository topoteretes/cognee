import asyncio
import logging
from uuid import NAMESPACE_OID, uuid5

from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, MonitoringTool
from cognee.shared.utils import render_graph
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.ingestion import ingest_data
from cognee.tasks.repo_processor import (
    get_data_list_for_user,
    get_non_py_files,
    get_repo_file_dependencies,
)

from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.infrastructure.llm import get_max_chunk_tokens

monitoring = get_base_config().monitoring_tool
if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


logger = logging.getLogger("code_graph_pipeline")
update_status_lock = asyncio.Lock()


@observe
async def run_code_graph_pipeline(repo_path, include_docs=False):
    import cognee
    from cognee.low_level import setup

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    cognee_config = get_cognify_config()
    user = await get_default_user()
    detailed_extraction = False

    tasks = [
        Task(get_repo_file_dependencies, detailed_extraction=detailed_extraction),
        # Task(enrich_dependency_graph, task_config={"batch_size": 50}),
        # Task(expand_dependency_graph, task_config={"batch_size": 50}),
        # Task(get_source_code_chunks, task_config={"batch_size": 50}),
        # Task(summarize_code, task_config={"batch_size": 50}),
        Task(add_data_points, task_config={"batch_size": 100 if detailed_extraction else 500}),
    ]

    if include_docs:
        non_code_tasks = [
            Task(get_non_py_files, task_config={"batch_size": 50}),
            Task(ingest_data, dataset_name="repo_docs", user=user),
            Task(get_data_list_for_user, dataset_name="repo_docs", user=user),
            Task(classify_documents),
            Task(extract_chunks_from_documents, max_chunk_tokens=get_max_chunk_tokens()),
            Task(
                extract_graph_from_data, graph_model=KnowledgeGraph, task_config={"batch_size": 50}
            ),
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config={"batch_size": 50},
            ),
        ]

    dataset_id = uuid5(NAMESPACE_OID, "codebase")

    if include_docs:
        non_code_pipeline_run = run_tasks(non_code_tasks, dataset_id, repo_path, "cognify_pipeline")
        async for run_status in non_code_pipeline_run:
            yield run_status

    async for run_status in run_tasks(tasks, dataset_id, repo_path, "cognify_code_pipeline"):
        yield run_status


if __name__ == "__main__":

    async def main():
        async for data_points in run_code_graph_pipeline("REPO_PATH"):
            print(data_points)

        await render_graph()

    asyncio.run(main())
