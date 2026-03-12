import asyncio
import numpy as np
from typing import Dict, Type, List, Optional, Any
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType


def _get_closest_match(df: DataFrame, query_vector) -> List[Any]:
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

    closest_idx = int(np.argmax(sims))
    similarity_val = float(sims[closest_idx])

    names = df.columns.to_numpy()
    return [names[closest_idx], similarity_val]


async def cache_and_replace_nodes(nodes, **kwargs):
    df = kwargs.get("df", None)
    similarity_threshold = kwargs.get("similarity_threshold", 1.0)
    stats = kwargs.get("stats", None)

    if df is None or stats is None:
        return
    vector_engine = get_vector_engine()
    df_new = pd.DataFrame()
    for chunk in nodes:
        for _, node in chunk.contains:
            if node and (isinstance(node, Entity) or isinstance(node, EntityType)):
                vector = await vector_engine.embed_data([node.name])
                closest_match = _get_closest_match(df, vector[0])
                if len(closest_match) > 0:
                    print(f"node={node.name}, closest_match={closest_match}")
                    if closest_match[1] > similarity_threshold:
                        node.name = closest_match[0]
                        if isinstance(stats, dict):
                            stats["reused_entities"] = (stats.get("reused_entities") or 0) + 1
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


async def calculate_chunk_graphs_post_extraction_canonicalization(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    custom_prompt: Optional[str] = None,
    **kwargs,
):
    extractor_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in {
            "calculate_chunk_graphs",
            "cache_entity_embeddings",
            "df",
            "similarity_threshold",
            "stats",
        }
    }

    return await asyncio.gather(
        *[
            extract_content_graph(
                chunk.text, graph_model, custom_prompt=custom_prompt, **extractor_kwargs
            )
            for chunk in data_chunks
        ]
    )
