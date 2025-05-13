import os
import pathlib
import asyncio
from uuid import NAMESPACE_OID, uuid5
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability.get_observe import get_observe

from cognee.api.v1.search import SearchType, search
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.ingestion import ingest_data
from cognee.tasks.repo_processor import get_non_py_files, get_repo_file_dependencies

from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.infrastructure.llm import get_max_chunk_tokens

observe = get_observe()

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
        Task(get_repo_file_dependencies, detailed_extraction=detailed_extraction),
        # Task(summarize_code, task_config={"batch_size": 500}), # This task takes a long time to complete
        Task(add_data_points, task_config={"batch_size": 500}),
    ]

    if include_docs:
        # This tasks take a long time to complete
        non_code_tasks = [
            Task(get_non_py_files, task_config={"batch_size": 50}),
            Task(ingest_data, dataset_name="repo_docs", user=user),
            Task(classify_documents),
            Task(extract_chunks_from_documents, max_chunk_size=get_max_chunk_tokens()),
            Task(
                extract_graph_from_data,
                graph_model=KnowledgeGraph,
                task_config={"batch_size": 50},
            ),
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config={"batch_size": 50},
            ),
        ]

    dataset_id = await get_unique_dataset_id("codebase", user)

    if include_docs:
        non_code_pipeline_run = run_tasks(
            non_code_tasks, dataset_id, repo_path, user, "cognify_pipeline"
        )
        async for run_status in non_code_pipeline_run:
            yield run_status

    async for run_status in run_tasks(tasks, dataset_id, repo_path, user, "cognify_code_pipeline"):
        yield run_status


if __name__ == "__main__":

    async def main():
        async for run_status in run_code_graph_pipeline("REPO_PATH"):
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
