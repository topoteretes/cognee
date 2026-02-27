import asyncio
import numpy as np
from typing import Dict, Type, List, Optional
from pydantic import BaseModel

import pandas as pd
from pandas import DataFrame
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.models import Entity, EntityType
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


_df_lock = asyncio.Lock()
_reused_items_lock = asyncio.Lock()


async def cache_nodes(df, nodes):
    if df is None:
        return
    vector_engine = get_vector_engine()
    df_new = pd.DataFrame()
    for chunk in nodes:
        for _, node in chunk.contains:
            if node and (isinstance(node, Entity) or isinstance(node, EntityType)):
                vector = await vector_engine.embed_data(node.name)
                if node.name in df_new.columns:
                    continue
                # Store as numeric column (not list-in-cell) for fast vectorized ops.
                df_new[node.name] = pd.Series(vector[0], dtype=float)

    async with _df_lock:
        if not df_new.empty:
            # Drop only overlapping columns in one shot to avoid in-place mutation
            # during iteration and to tolerate any concurrent column changes.
            overlap = df_new.columns.intersection(df.columns)
            if len(overlap) > 0:
                df_new.drop(columns=overlap, inplace=True, errors="ignore")
        # avoid fragmentation, improve speed, keep the same df
        combined = pd.concat([df, df_new], axis=1)
        df._mgr = combined._mgr
        df._item_cache.clear()


async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_resolver: BaseOntologyResolver,
    context: Dict,
    df: DataFrame,
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

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks, chunk_graphs, ontology_resolver, existing_edges_map
    )

    cognify_config = get_cognify_config()
    embed_triplets = cognify_config.triplet_embedding
    await cache_nodes(df, graph_nodes)

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


def top_k_by_cosine(df: DataFrame, query_vector, k: int = 5) -> List[str]:
    """
    Returns top-k (name, cosine_similarity) pairs where each column of df is a vector.
    Assumes df columns are named by vector key and each column is a 1D vector.
    """
    if df is None or df.empty:
        return []

    # columns are vectors; shape: (dim, n)
    M = df.to_numpy(dtype=float)  # shape (dim, n_cols)
    q = np.asarray(query_vector, dtype=float)  # shape (dim,)

    q_norm = np.linalg.norm(q)
    if q_norm == 0 or M.size == 0:
        return []

    # cosine similarity for all columns at once
    denom = np.linalg.norm(M, axis=0) * q_norm
    # avoid divide-by-zero
    denom = np.where(denom == 0, np.inf, denom)
    sims = (M.T @ q) / denom  # shape (n_cols,)

    # top-k indices
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    names = df.columns.to_numpy()
    return [names[i] for i in idx]


async def build_prompt(chunk, df, vector_search_limit, custom_prompt) -> Optional[str]:
    vector_engine = get_vector_engine()
    query_vector = (await vector_engine.embedding_engine.embed_text([chunk.text]))[0]
    closest_matches = top_k_by_cosine(df, query_vector, k=vector_search_limit)

    prompt = custom_prompt
    for match in closest_matches:
        prompt = prompt + "\n  -" + match
    return prompt


def _count_reused_items_in_prompt_tail(prompt: Optional[str], graph, vector_search_limit) -> int:
    if not prompt or not getattr(graph, "nodes", None):
        return 0
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if not lines:
        return 0

    # Consider the last vector_search_limit lines (or fewer if prompt is shorter).
    tail_lines = lines[-vector_search_limit:]
    tail_blob = "\n".join(tail_lines).casefold()

    count = 0
    for node in graph.nodes:
        name = getattr(node, "name", None)
        if not name:
            continue
        if name.casefold() in tail_blob:
            count += 1
    return count


async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    context: Dict,
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    **kwargs,
) -> List[DocumentChunk]:
    """
    Extracts and integrates a knowledge graph from the text content of document chunks using a specified graph model.
    """
    vector_search_limit = kwargs.get("vector_search_limit") or 5
    df = kwargs.get("df", None)
    use_poc = kwargs.get("use_poc") or False

    if not isinstance(data_chunks, list) or not data_chunks:
        raise InvalidDataChunksError("must be a non-empty list of DocumentChunk.")
    if not all(hasattr(c, "text") for c in data_chunks):
        raise InvalidDataChunksError("each chunk must have a 'text' attribute")
    if not isinstance(graph_model, type) or not issubclass(graph_model, BaseModel):
        raise InvalidGraphModelError(graph_model)

    if use_poc:
        chunk_prompts = await asyncio.gather(
            *[build_prompt(chunk, df, vector_search_limit, custom_prompt) for chunk in data_chunks]
        )
        chunk_graphs = await asyncio.gather(
            *[
                extract_content_graph(chunk.text, graph_model, custom_prompt=prompt)
                for chunk, prompt in zip(data_chunks, chunk_prompts)
            ]
        )
    else:
        chunk_graphs = await asyncio.gather(
            *[
                extract_content_graph(
                    chunk.text, graph_model, custom_prompt=custom_prompt, **kwargs
                )
                for chunk in data_chunks
            ]
        )

    if use_poc:
        reused_total = sum(
            _count_reused_items_in_prompt_tail(prompt, graph, vector_search_limit)
            for prompt, graph in zip(chunk_prompts, chunk_graphs)
        )
        if reused_total:
            # "Atomic" update for shared kwargs dict across async tasks.
            async with _reused_items_lock:
                stats = kwargs.get("stats")
                if isinstance(stats, dict):
                    stats["reused_entities"] = (stats.get("reused_entities") or 0) + reused_total

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
        df,
        pipeline_name=pipeline_name,
        task_name=task_name,
    )
