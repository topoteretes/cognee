"""
Cognify pipeline for layered knowledge graphs.

This module provides a pipeline for processing documents and extracting layered knowledge graphs
from them, following a similar structure to the cognify_v2 pipeline.
"""

import asyncio
import logging
from typing import Union, List, Dict, Any, Optional

from pydantic import BaseModel

from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.data.methods import get_datasets, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines import run_tasks
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
from cognee.tasks.graph import (
    extract_layered_graph_from_data,
    store_layered_graphs,
    process_layered_graphs_with_pipeline
)
from cognee.tasks.storage import add_data_points
from cognee.modules.chunking.TextChunker import TextChunker

logger = logging.getLogger("cognify.layered_graph")

async def cognify_layered_graph(
    datasets: Union[str, List[str]] = None,
    user: User = None,
    layer_config: Optional[List[Dict[str, Any]]] = None,
    pipeline_config: Optional[List[Dict[str, Any]]] = None,
    tasks: List[Task] = None,
):
    """
    Processes documents to extract layered knowledge graphs.
    
    Args:
        datasets: One or more dataset names or Dataset objects
        user: The user requesting the processing
        layer_config: Configuration for the layers in the graph
        pipeline_config: Configuration for the graph processing pipeline
        tasks: Optional custom task list to use instead of the default
        
    Returns:
        The results of the pipeline execution
    """
    if user is None:
        user = await get_default_user()

    existing_datasets = await get_datasets(user.id)

    if datasets is None or len(datasets) == 0:
        # If no datasets are provided, cognify all existing datasets.
        datasets = existing_datasets

    if isinstance(datasets[0], str):
        datasets = await get_datasets_by_name(datasets, user.id)

    existing_datasets_map = {
        generate_dataset_name(dataset.name): True for dataset in existing_datasets
    }

    awaitables = []

    if tasks is None:
        tasks = await get_default_layered_graph_tasks(
            user, layer_config, pipeline_config
        )

    for dataset in datasets:
        dataset_name = generate_dataset_name(dataset.name)

        if dataset_name in existing_datasets_map:
            awaitables.append(run_layered_graph_pipeline(dataset, user, tasks))

    return await asyncio.gather(*awaitables)

async def run_layered_graph_pipeline(dataset: Dataset, user: User, tasks: List[Task]):
    """
    Runs the layered graph pipeline on a dataset.
    
    Args:
        dataset: The dataset to process
        user: The user requesting the processing
        tasks: The tasks to run in the pipeline
        
    Returns:
        The pipeline execution results
    """
    data_documents: List[Data] = await get_dataset_data(dataset_id=dataset.id)

    dataset_id = dataset.id
    dataset_name = generate_dataset_name(dataset.name)

    send_telemetry("cognee.cognify.layered_graph EXECUTION STARTED", user.id)

    try:
        if not isinstance(tasks, list):
            raise ValueError("Tasks must be a list")

        for task in tasks:
            if not isinstance(task, Task):
                raise ValueError(f"Task {task} is not an instance of Task")

        pipeline_run = run_tasks(tasks, dataset.id, data_documents, "layered_graph_pipeline")
        pipeline_run_status = None

        async for run_status in pipeline_run:
            pipeline_run_status = run_status

        send_telemetry("cognee.cognify.layered_graph EXECUTION COMPLETED", user.id)
        return pipeline_run_status

    except Exception as error:
        send_telemetry("cognee.cognify.layered_graph EXECUTION ERRORED", user.id)
        raise error

def generate_dataset_name(dataset_name: str) -> str:
    """Generate a clean dataset name by replacing certain characters."""
    return dataset_name.replace(".", "_").replace(" ", "_")

async def get_default_layered_graph_tasks(
    user: User = None, 
    layer_config: Optional[List[Dict[str, Any]]] = None,
    pipeline_config: Optional[List[Dict[str, Any]]] = None,
    chunk_size=1024, 
    chunker=TextChunker
) -> List[Task]:
    """
    Get the default tasks for the layered graph pipeline.
    
    Args:
        user: The user requesting the processing
        layer_config: Configuration for the layers in the graph
        pipeline_config: Configuration for the graph processing pipeline
        chunk_size: The size of chunks to extract from documents
        chunker: The chunker to use for extracting chunks
        
    Returns:
        A list of Task objects
    """
    if user is None:
        user = await get_default_user()
    
    if layer_config is None:
        # Default layer configuration with base, classification, and inference layers
        layer_config = [
            {
                "name": "Base Layer",
                "description": "Basic entities and relationships extracted from content",
                "layer_type": "base",
                "prompt": "Extract the main entities and relationships from the content, focusing on key concepts, people, organizations, and events."
            },
            {
                "name": "Classification Layer",
                "description": "Classification of entities in the base layer",
                "layer_type": "classification",
                "prompt": "Classify the entities in the content into categories and create hierarchical relationships."
            },
            {
                "name": "Inference Layer",
                "description": "Inferred relationships and entities",
                "layer_type": "inference",
                "prompt": "Infer additional relationships and entities that are implied but not explicitly stated in the content."
            }
        ]
    
    if pipeline_config is None:
        # Default pipeline configuration with analysis and storage steps
        pipeline_config = [
            {
                "type": "analyze",
                "description": "Analyze the layered graph and calculate metrics"
            },
            {
                "type": "store",
                "description": "Store the layered graph in the graph database"
            }
        ]

    try:
        default_tasks = [
            Task(classify_documents),
            Task(check_permissions_on_documents, user=user, permissions=["write"]),
            Task(
                extract_chunks_from_documents,
                max_chunk_tokens=get_max_chunk_tokens(),
                chunker=chunker,
                chunk_size=chunk_size,
            ),  # Extract text chunks based on the document type.
            Task(
                extract_layered_graph_from_data,
                layer_config=layer_config,
                task_config={"batch_size": 10}
            ),  # Extract layered knowledge graphs from the document chunks.
            Task(
                process_layered_graphs_with_pipeline,
                pipeline_config=pipeline_config,
                task_config={"batch_size": 10}
            ),  # Process the layered graphs with the configured pipeline.
            Task(store_layered_graphs, task_config={"batch_size": 10}),  # Store the layered graphs.
            Task(add_data_points, task_config={"batch_size": 10}),  # Add the data points to the database.
        ]
    except Exception as error:
        send_telemetry("cognee.cognify.layered_graph DEFAULT TASKS CREATION ERRORED", user.id)
        raise error
    
    return default_tasks 