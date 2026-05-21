import asyncio
import os
import time
import cognee
import numpy as np
from typing import Dict, Type, List, Optional, Any
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.modules.chunking.models import DocumentChunk


def _get_closest_match(df: DataFrame, query_vector) -> list[Any] | tuple[Any, float]:
    if df is None or df.empty:
        return None

    # columns are vectors; shape: (dim, n)
    M = df.to_numpy(dtype=float)  # shape (dim, n_cols)
    q = np.asarray(query_vector, dtype=float)  # shape (dim,)

    q_norm = np.linalg.norm(q)
    if q_norm == 0 or M.size == 0:
        return None

    # cosine similarity for all columns at once
    denom = np.linalg.norm(M, axis=0) * q_norm
    # avoid divide-by-zero
    denom = np.where(denom == 0, np.inf, denom)
    sims = (M.T @ q) / denom  # shape (n_cols,)

    closest_idx = int(np.argmax(sims))
    similarity_val = float(sims[closest_idx])

    names = df.columns.to_numpy()
    return names[closest_idx], similarity_val


async def _get_closest_match_1(
    node, vector_engine, df: DataFrame, df_new: DataFrame, similarity_threshold, stats
) -> tuple[Any, Any, float]:
    query_vector = await vector_engine.embed_data(node.name)
    if node.name not in df_new.columns and node.name not in df.columns:
        df_new[node.name] = pd.Series(query_vector[0], dtype=float)

    if df is None or df.empty:
        return None
    # columns are vectors; shape: (dim, n)
    M = df.to_numpy(dtype=float)  # shape (dim, n_cols)
    q = np.asarray(query_vector[0], dtype=float)  # shape (dim,)

    q_norm = np.linalg.norm(q)
    if q_norm == 0 or M.size == 0:
        return None

    # cosine similarity for all columns at once
    denom = np.linalg.norm(M, axis=0) * q_norm
    # avoid divide-by-zero
    denom = np.where(denom == 0, np.inf, denom)
    sims = (M.T @ q) / denom  # shape (n_cols,)

    closest_idx = int(np.argmax(sims))
    similarity_val = float(sims[closest_idx])

    names = df.columns.to_numpy()
    closest_match_name = names[closest_idx]
    print(
        f"node={node.name}, closest_match={closest_match_name}, match_similarity={similarity_val}"
    )
    if similarity_val > similarity_threshold:
        node.name = closest_match_name
        if isinstance(stats, dict):
            stats["reused_entities"] = (stats.get("reused_entities") or 0) + 1

    return node.name, names[closest_idx], similarity_val


async def cache_and_replace_nodes(graphs, **kwargs):
    df = kwargs.get("df", None)
    similarity_threshold = kwargs.get("similarity_threshold", 1.0)
    stats = kwargs.get("stats", None)

    vector_engine = get_vector_engine()
    df_new = pd.DataFrame()
    for graph in graphs:
        await asyncio.gather(
            *[
                _get_closest_match_1(node, vector_engine, df, df_new, similarity_threshold, stats)
                for node in graph.nodes
            ]
        )

    if not df_new.empty:
        # Drop only overlapping columns in one shot to avoid in-place mutation
        # during iteration and to tolerate any concurrent column changes.
        overlap = df_new.columns.intersection(df.columns)
        if len(overlap) > 0:
            df_new.drop(columns=overlap, inplace=True, errors="ignore")
    # avoid fragmentation, improve speed, keep the same df
    df[df_new.columns] = df_new


async def _get_entity_names_from_graph() -> set[str]:
    graph_engine = await get_graph_engine()
    nodes, _ = await graph_engine.get_graph_data()
    names = set()
    for _node_id, props in nodes:
        props = props or {}
        if props.get("type") in {"Entity", "GraphEntity"}:
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _report_disambiguation_rate(graph_entity_names, entities_to_disambiguate):
    unresolved_entities_count = 0
    for name in graph_entity_names:
        if name in entities_to_disambiguate:
            unresolved_entities_count += 1
    print(
        f"Disambiguated entities: {(len(entities_to_disambiguate) - unresolved_entities_count) / len(entities_to_disambiguate) * 100}%"
    )


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

    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(
                chunk.text, graph_model, custom_prompt=custom_prompt, **extractor_kwargs
            )
            for chunk in data_chunks
        ]
    )

    return chunk_graphs


async def post_extraction_canonicalization(
    parts_dir,
    custom_prompt,
    disambiguated_entities_names_file: Optional[str] = None,
):
    df = pd.DataFrame()
    kwargs = {
        "calculate_chunk_graphs": calculate_chunk_graphs_post_extraction_canonicalization,
        "cache_entity_embeddings": cache_and_replace_nodes,
        "df": df,
        "similarity_threshold": 0.8,
        "stats": {"reused_entities": 0},
    }

    parent_folder = os.path.dirname(os.path.abspath(__file__))
    if disambiguated_entities_names_file is None:
        disambiguated_entities_names_file = os.path.join(
            parent_folder, "data", "example2", "expected_disambiguation_entities.txt"
        )
    with open(disambiguated_entities_names_file, "r", encoding="utf-8") as f:
        disambiguated_entities_names = f.read().split("\n")

    start = time.perf_counter()
    for part in sorted(parts_dir.glob("part_*.txt")):
        print(part)
        text = part.read_text(encoding="utf-8").replace("\n", " ")
        await cognee.add(text)
        await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.6f} seconds")

        graph_entity_names = await _get_entity_names_from_graph()
        _report_disambiguation_rate(graph_entity_names, disambiguated_entities_names)

    print(f"Reused instances: {kwargs.get('stats').get('reused_entities')}")
