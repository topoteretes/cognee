import os
import pathlib
import asyncio
from uuid import NAMESPACE_OID, uuid5

from cognee.shared.logging_utils import get_logger
from cognee.api.v1.search import SearchType, search
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.Task import Task, TaskConfig
from cognee.modules.pipelines.operations.needs import merge_needs
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, MonitoringTool

from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.ingestion import ingest_data
from cognee.tasks.repo_processor import get_non_py_files, get_repo_file_dependencies

from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.infrastructure.llm import get_max_chunk_tokens

monitoring = get_base_config().monitoring_tool

if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


logger = get_logger("code_graph_pipeline")


@observe
async def run_code_graph_pipeline(repo_path, include_docs=False):
    import cognee
    from cognee.low_level import setup

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    cognee_config = get_cognify_config()
    user = await get_default_user()
    detailed_extraction = True

    tasks = [
        Task(
            get_repo_file_dependencies,
            detailed_extraction=detailed_extraction,
            task_config=TaskConfig(output_batch_size=500),
        ),
        # Task(summarize_code, task_config=TaskConfig(output_batch_size=500)), # This task takes a long time to complete
        Task(add_data_points, task_config=TaskConfig(needs=[get_repo_file_dependencies])),
    ]

    if include_docs:
        # This tasks take a long time to complete
        non_code_tasks = [
            Task(get_non_py_files),
            Task(
                ingest_data,
                dataset_name="repo_docs",
                user=user,
                task_config=TaskConfig(needs=[get_non_py_files]),
            ),
            Task(classify_documents, task_config=TaskConfig(needs=[ingest_data])),
            Task(
                extract_chunks_from_documents,
                max_chunk_size=get_max_chunk_tokens(),
                task_config=TaskConfig(needs=[classify_documents], output_batch_size=10),
            ),
            Task(
                extract_graph_from_data,
                graph_model=KnowledgeGraph,
                task_config=TaskConfig(needs=[extract_chunks_from_documents]),
            ),
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config=TaskConfig(needs=[extract_chunks_from_documents]),
            ),
            Task(
                add_data_points,
                task_config=TaskConfig(
                    needs=[merge_needs(summarize_text, extract_graph_from_data)]
                ),
            ),
        ]

    dataset_id = uuid5(NAMESPACE_OID, "codebase")

    if include_docs:
        non_code_pipeline_run = run_tasks(non_code_tasks, dataset_id, repo_path, "cognify_pipeline")
        async for run_info in non_code_pipeline_run:
            yield run_info

    async for run_info in run_tasks(tasks, dataset_id, repo_path, "cognify_code_pipeline"):
        yield run_info


if __name__ == "__main__":

    async def main():
        async for run_status in run_code_graph_pipeline("/Users/borisarzentar/Projects/graphrag"):
            print(f"{run_status.pipeline_name}: {run_status.status}")

        file_path = os.path.join(
            pathlib.Path(__file__).parent, ".artifacts", "graph_visualization.html"
        )
        await visualize_graph(file_path)

        search_results = await search(
            query_type=SearchType.CODE,
            query_text="How is Relationship weight calculated?",
        )

        for file in search_results:
            print(file["name"])

    asyncio.run(main())
