# ------------------------------------------------------------------------------
# Producer function that produces data points from documents and pushes them into the queue.
# ------------------------------------------------------------------------------
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.data_models import KnowledgeGraph

from distributed.app import app
from distributed.modal_image import image
from distributed.tasks.summarize_text import summarize_text
from distributed.tasks.extract_graph_from_data import extract_graph_from_data
from distributed.tasks.save_data_points import save_data_points


@app.function(image=image, timeout=86400, max_containers=100)
async def graph_extraction_worker(user, document_name: str, document_chunks: list):
    cognee_config = get_cognify_config()

    tasks = [
        Task(
            extract_graph_from_data,
            graph_model=KnowledgeGraph,
        ),
        Task(
            summarize_text,
            summarization_model=cognee_config.summarization_model,
        ),
        Task(save_data_points),
    ]

    async for _ in run_tasks(
        tasks,
        data=document_chunks,
        pipeline_name=f"modal_execution_file_{document_name}",
        user=user,
    ):
        pass

    print(f"File execution finished: {document_name}")

    return document_name
