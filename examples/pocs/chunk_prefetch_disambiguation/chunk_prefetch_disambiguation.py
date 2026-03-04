import asyncio
import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import Optional, List, Type

from pydantic import BaseModel

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType


def _count_reused_node_names_in_prompt_tail(
    prompt: Optional[str], graph, vector_search_limit
) -> int:
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


def _top_k_names_by_cosine(df: DataFrame, query_vector, k: int = 5) -> List[str]:
    if df is None or df.empty:
        return []

    # columns are vectors; shape: (dim, n)
    M = df.to_numpy(dtype=float)  # shape (dim, n_cols)
    q = np.asarray(query_vector, dtype=float)  # shape (dim,)

    if M.shape[0] != q.shape[0]:
        raise ValueError(f"Embedding dimension mismatch: stored={M.shape[0]} query={q.shape[0]}")

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


async def _build_disambiguation_prompt(
    chunk, df, vector_search_limit, custom_prompt
) -> Optional[str]:
    vector_engine = get_vector_engine()
    query_vector = (await vector_engine.embedding_engine.embed_text([chunk.text]))[0]
    closest_matches = _top_k_names_by_cosine(df, query_vector, vector_search_limit)

    prompt = custom_prompt
    for match in closest_matches:
        prompt = prompt + "\n  -" + match
    return prompt


def _update_reused_node_name_stats(chunk_graphs, chunk_prompts, **kwargs):
    vector_search_limit = kwargs.get("vector_search_limit") or 5
    reused_total = sum(
        _count_reused_node_names_in_prompt_tail(prompt, graph, vector_search_limit)
        for prompt, graph in zip(chunk_prompts, chunk_graphs)
    )
    if reused_total:
        stats = kwargs.get("stats")
        if isinstance(stats, dict):
            stats["reused_entities"] = (stats.get("reused_entities") or 0) + reused_total


async def _build_chunk_graphs_and_prompts(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    custom_prompt: Optional[str] = None,
    **kwargs,
):
    vector_search_limit = kwargs.get("vector_search_limit") or 5
    df = kwargs.get("df", None)
    llm_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in {
            "vector_search_limit",
            "df",
            "use_chunk_prefetch_disambiguation",
            "stats",
            "calculate_chunk_graphs",
            "cache_entity_embeddings",
        }
    }

    chunk_prompts = await asyncio.gather(
        *[
            _build_disambiguation_prompt(chunk, df, vector_search_limit, custom_prompt)
            for chunk in data_chunks
        ]
    )

    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=prompt, **llm_kwargs)
            for chunk, prompt in zip(data_chunks, chunk_prompts)
        ]
    )
    return chunk_graphs, chunk_prompts


async def cache_entity_embeddings(nodes, **kwargs) -> None:
    df = kwargs.get("df", None)
    if df is None:
        return
    vector_engine = get_vector_engine()
    df_new = pd.DataFrame()
    for chunk in nodes:
        for contained in getattr(chunk, "contains", None) or []:
            if isinstance(contained, tuple) and len(contained) == 2:
                _, node = contained
            else:
                node = contained
            if node and (isinstance(node, Entity) or isinstance(node, EntityType)):
                vector = await vector_engine.embed_data(node.name)
                if node.name in df_new.columns:
                    continue
                # Store as numeric column (not list-in-cell) for fast vectorized ops.
                df_new[node.name] = pd.Series(vector[0], dtype=float)

    if not df_new.empty:
        # Drop only overlapping columns in one shot to avoid in-place mutation
        # during iteration and to tolerate any concurrent column changes.
        overlap = df_new.columns.intersection(df.columns)
        if len(overlap) > 0:
            df_new.drop(columns=overlap, inplace=True, errors="ignore")
    # avoid fragmentation, improve speed, keep the same df
    df[df_new.columns] = df_new


async def calculate_chunk_graphs_chunk_prefetch_disambiguation(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    custom_prompt: str,
    **kwargs,
):
    extractor_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in {
            "calculate_chunk_graphs",
            "cache_entity_embeddings",
        }
    }
    chunk_graphs, chunk_prompts = await _build_chunk_graphs_and_prompts(
        data_chunks, graph_model, custom_prompt, **extractor_kwargs
    )

    _update_reused_node_name_stats(chunk_graphs, chunk_prompts, **extractor_kwargs)

    return chunk_graphs
