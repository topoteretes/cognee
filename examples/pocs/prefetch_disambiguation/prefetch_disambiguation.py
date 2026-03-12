import asyncio
import os
import time
import cognee
import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import Optional, List, Type

from pydantic import BaseModel
from nltk.tokenize import sent_tokenize

from cognee.infrastructure.databases.graph import get_graph_engine
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


def _report_disambiguation_rate(graph_entity_names, entities_to_disambiguate):
    unresolved_entities_count = 0
    for name in graph_entity_names:
        if name in entities_to_disambiguate:
            unresolved_entities_count += 1
    print(
        f"Disambiguated entities: {(len(entities_to_disambiguate) - unresolved_entities_count) / len(entities_to_disambiguate) * 100}%"
    )


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


def _build_disambiguation_prompt(
    chunk_embedding, df, vector_search_limit, custom_prompt
) -> Optional[str]:
    closest_matches = _top_k_names_by_cosine(df, chunk_embedding, vector_search_limit)

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

    chunk_texts = [chunk.text for chunk in data_chunks]
    vector_engine = get_vector_engine()
    chunk_embeddings = await vector_engine.embedding_engine.embed_text(chunk_texts)
    chunk_prompts = [
        _build_disambiguation_prompt(chunk_embedding, df, vector_search_limit, custom_prompt)
        for chunk_embedding in chunk_embeddings
    ]

    chunk_graphs = await asyncio.gather(
        *[
            extract_content_graph(chunk.text, graph_model, custom_prompt=prompt, **llm_kwargs)
            for chunk, prompt in zip(data_chunks, chunk_prompts)
        ]
    )
    return chunk_graphs, chunk_prompts


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


async def prefetch_disambiguation(
    parts_dir,
    vector_search_limit,
    split_by_sentence,
    custom_prompt,
    disambiguated_entities_names_file: Optional[str] = None,
):
    df = pd.DataFrame()
    kwargs = {
        "vector_search_limit": vector_search_limit,
        "calculate_chunk_graphs": calculate_chunk_graphs_prefetch_disambiguation,
        "cache_entity_embeddings": cache_entity_embeddings,
        "df": df,
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
        if split_by_sentence:
            text = list(dict.fromkeys(sent_tokenize(text)))
        await cognee.add(text)
        await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.6f} seconds")

        graph_entity_names = await _get_entity_names_from_graph()
        _report_disambiguation_rate(graph_entity_names, disambiguated_entities_names)

    print(f"Reused instances: {kwargs.get('stats').get('reused_entities')}")


async def cache_entity_embeddings(graphs, **kwargs) -> None:
    df = kwargs.get("df", None)
    if df is None:
        return
    vector_engine = get_vector_engine()
    df_new = pd.DataFrame()
    for graph in graphs:
        entity_names = [node.name for node in graph.nodes]
        if not entity_names:
            continue
        entity_vectors = await vector_engine.embed_data(entity_names)
        for name, vector in zip(entity_names, entity_vectors):
            if name in df_new.columns or name in df.columns:
                continue
            # Store as numeric column (not list-in-cell) for fast vectorized ops.
            df_new[name] = pd.Series(vector, dtype=float)

    if not df_new.empty:
        # Drop only overlapping columns in one shot to avoid in-place mutation
        # during iteration and to tolerate any concurrent column changes.
        overlap = df_new.columns.intersection(df.columns)
        if len(overlap) > 0:
            df_new.drop(columns=overlap, inplace=True, errors="ignore")
    # avoid fragmentation, improve speed, keep the same df
    df[df_new.columns] = df_new


async def calculate_chunk_graphs_prefetch_disambiguation(
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
