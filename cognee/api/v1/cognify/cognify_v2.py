import asyncio
from cognee.shared.logging_utils import get_logger
from typing import Union, Optional

from pydantic import BaseModel

from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.utils import send_telemetry
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.modules.chunking.TextChunker import TextChunker

from .pipeline import cognee_pipeline


async def cognify(
    datasets: Union[str, list[str]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    ontology_file_path: Optional[str] = None,
):
    cognify_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_documents, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=get_max_chunk_tokens(),
            chunker=TextChunker,
        ),  # Extract text chunks based on the document type.
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=OntologyResolver(ontology_file=ontology_file_path),
            task_config={"batch_size": 10},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": 10},
        ),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return await cognee_pipeline(tasks=cognify_tasks, datasets=datasets, user=user)
