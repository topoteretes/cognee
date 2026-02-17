import asyncio
from typing import Dict, Type, List, Optional
from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.methods import upsert_edges
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.tasks.storage import index_graph_edges
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.graph.exceptions import (
    InvalidGraphModelError,
    InvalidDataChunksError,
    InvalidChunkGraphInputError,
    InvalidOntologyAdapterError,
)
from cognee.modules.cognify.config import get_cognify_config
from poc_single_add_datapoints.poc_expand_with_nodes_and_edges import (
    poc_expand_with_nodes_and_edges,
)


def _stamp_provenance_deep(data, pipeline_name, task_name, visited=None):
    """Recursively stamp all reachable DataPoints with provenance info."""
    if visited is None:
        visited = set()

    if isinstance(data, DataPoint):
        obj_id = id(data)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if data.source_pipeline is None:
            data.source_pipeline = pipeline_name
        if data.source_task is None:
            data.source_task = task_name

        for field_name in data.model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance_deep(field_value, pipeline_name, task_name, visited)

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance_deep(item, pipeline_name, task_name, visited)


async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_resolver: BaseOntologyResolver,
    context: Dict,
    use_single_add_datapoints_poc,
    pipeline_name: str = None,
    task_name: str = None,
) -> List[DocumentChunk]:
    """Integrate chunk graphs with ontology validation and store in databases.

    This function processes document chunks and their associated knowledge graphs,
    validates entities against an ontology resolver, and stores the integrated
    data points and edges in the configured databases.

    Args:
        data_chunks: List of document chunks containing source data
        chunk_graphs: List of knowledge graphs corresponding to each chunk
        graph_model: Pydantic model class for graph data validation
        ontology_resolver: Resolver for validating entities against ontology

    Returns:
        List of updated DocumentChunk objects with integrated data

    Raises:
        InvalidChunkGraphInputError: If input validation fails
        InvalidGraphModelError: If graph model validation fails
        InvalidOntologyAdapterError: If ontology resolver validation fails
    """

    if not isinstance(data_chunks, list) or not isinstance(chunk_graphs, list):
        raise InvalidChunkGraphInputError("data_chunks and chunk_graphs must be lists.")
    if len(data_chunks) != len(chunk_graphs):
        raise InvalidChunkGraphInputError(
            f"length mismatch: {len(data_chunks)} chunks vs {len(chunk_graphs)} graphs."
        )
    if not isinstance(graph_model, type) or not issubclass(graph_model, BaseModel):
        raise InvalidGraphModelError(graph_model)
    if ontology_resolver is None or not hasattr(ontology_resolver, "get_subgraph"):
        raise InvalidOntologyAdapterError(
            type(ontology_resolver).__name__ if ontology_resolver else "None"
        )

    graph_engine = await get_graph_engine()

    if graph_model is not KnowledgeGraph:
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        return data_chunks

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
    )

    if use_single_add_datapoints_poc:
        poc_expand_with_nodes_and_edges(
            data_chunks, chunk_graphs, ontology_resolver, existing_edges_map
        )
    else:
        graph_nodes, graph_edges = expand_with_nodes_and_edges(
            data_chunks, chunk_graphs, ontology_resolver, existing_edges_map
        )

        cognify_config = get_cognify_config()
        embed_triplets = cognify_config.triplet_embedding

        if len(graph_nodes) > 0:
            if pipeline_name or task_name:
                for node in graph_nodes:
                    _stamp_provenance_deep(node, pipeline_name, task_name)

            await add_data_points(
                data_points=graph_nodes, custom_edges=context, embed_triplets=embed_triplets
            )

        if len(graph_edges) > 0:
            await graph_engine.add_edges(graph_edges)
            await index_graph_edges(graph_edges)

            user = context["user"] if "user" in context else None

            if user:
                await upsert_edges(
                    graph_edges,
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    dataset_id=context["dataset"].id,
                    data_id=context["data"].id,
                )

    return data_chunks


async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    context: Dict,
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    use_single_add_datapoints_poc: bool = False,
    **kwargs,
) -> List[DocumentChunk]:
    """
    Extracts and integrates a knowledge graph from the text content of document chunks using a specified graph model.
    """

    if not isinstance(data_chunks, list) or not data_chunks:
        raise InvalidDataChunksError("must be a non-empty list of DocumentChunk.")
    if not all(hasattr(c, "text") for c in data_chunks):
        raise InvalidDataChunksError("each chunk must have a 'text' attribute")
    if not isinstance(graph_model, type) or not issubclass(graph_model, BaseModel):
        raise InvalidGraphModelError(graph_model)

    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=custom_prompt, **kwargs)
            for chunk in data_chunks
        ]
    )

    # Note: Filter edges with missing source or target nodes
    if graph_model == KnowledgeGraph:
        for graph in chunk_graphs:
            valid_node_ids = {node.id for node in graph.nodes}
            graph.edges = [
                edge
                for edge in graph.edges
                if edge.source_node_id in valid_node_ids and edge.target_node_id in valid_node_ids
            ]

    # Extract resolver from config if provided, otherwise get default
    if config is None:
        ontology_config = get_ontology_env_config()
        if (
            ontology_config.ontology_file_path
            and ontology_config.ontology_resolver
            and ontology_config.matching_strategy
        ):
            config: Config = {
                "ontology_config": {
                    "ontology_resolver": get_ontology_resolver_from_env(**ontology_config.to_dict())
                }
            }
        else:
            config: Config = {
                "ontology_config": {"ontology_resolver": get_default_ontology_resolver()}
            }

    ontology_resolver = config["ontology_config"]["ontology_resolver"]

    pipeline_name = context.get("pipeline_name") if isinstance(context, dict) else None
    task_name = "extract_graph_from_data"

    return await integrate_chunk_graphs(
        data_chunks,
        chunk_graphs,
        graph_model,
        ontology_resolver,
        context,
        use_single_add_datapoints_poc,
        pipeline_name=pipeline_name,
        task_name=task_name,
    )
