import asyncio
from typing import Optional

from pandas import DataFrame

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from pathlib import Path
import os
import pandas as pd

from cognee.infrastructure.databases.graph import get_graph_engine
from utils import _normalize_nodes, _normalize_edges, _compare


async def main(
    example,
    use_poc,
    vector_search_limit: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    df: Optional[DataFrame] = None,
    stats: Optional[dict] = None,
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if use_poc else ''}cognify_disambiguate_{example}_result.html",
    )

    kwargs = {"vector_search_limit": vector_search_limit}
    if use_poc:
        kwargs["use_poc"] = use_poc
        kwargs["df"] = df
        kwargs["stats"] = {"reused_entities": 0}

    text = Path("data/example1/part_1.txt").read_text(encoding="utf-8")
    for line in text.split("\n"):
        await cognee.add(line)
    await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)

    text = Path("data/example1/part_2.txt").read_text(encoding="utf-8")
    for line in text.split("\n"):
        await cognee.add(line)
    await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)

    print(kwargs.get("stats").get("reused_entities"))
    await visualize_graph(graph_visualization_path)

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    return nodes, edges


async def _run():
    with open("prompts/prompt1.txt", "r") as f:
        custom_prompt_text = f.read()

    for i in range(1, 2):
        example_id = str(i)
        df = pd.DataFrame()
        stats = {"reused_entities": 0}
        nodes, edges = await main(
            example="example" + example_id,
            use_poc=True,
            vector_search_limit=5,
            custom_prompt=custom_prompt_text,
            df=df,
            stats=stats,
        )
        poc_nodes, poc_edges = await main(
            example="example" + example_id,
            use_poc=True,
            vector_search_limit=10,
            custom_prompt=custom_prompt_text,
            df=df,
            stats=stats,
        )
        # nodes, edges = await main(
        #     example="example" + example_id,
        #     use_poc=False,
        # )
        poc_nodes = _normalize_nodes(poc_nodes)
        poc_edges = _normalize_edges(poc_edges)
        nodes = _normalize_nodes(nodes)
        edges = _normalize_edges(edges)

        _compare("Regular", nodes, "POC", poc_nodes)
        _compare("Regular", edges, "POC", poc_edges)


if __name__ == "__main__":
    asyncio.run(_run())
