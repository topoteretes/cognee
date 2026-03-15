import asyncio
from typing import Optional

from pandas import DataFrame

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from pathlib import Path
import os
import pandas as pd

from examples.pocs.post_extraction_canonicalization.post_extraction_canonicalization import (
    cache_and_replace_nodes,
    calculate_chunk_graphs_post_extraction_canonicalization,
)


async def main(
    example,
    use_post_extraction_canonicalization_disambiguation: Optional[bool] = False,
    custom_prompt: Optional[str] = None,
    df: Optional[DataFrame] = None,
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if use_post_extraction_canonicalization_disambiguation else ''}cognify_disambiguate_{example}_result.html",
    )

    parts_dir = Path(__file__).resolve().parent / "data" / example
    kwargs = {}
    if use_post_extraction_canonicalization_disambiguation:
        kwargs["calculate_chunk_graphs"] = calculate_chunk_graphs_post_extraction_canonicalization
        kwargs["cache_entity_embeddings"] = cache_and_replace_nodes
        kwargs["df"] = df
        kwargs["similarity_threshold"] = 0.7
        kwargs["stats"] = {"reused_entities": 0}

    for part in sorted(parts_dir.glob("part_*.txt")):
        print(part)
        text = part.read_text(encoding="utf-8")
        await cognee.add(text)
        await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)

    if use_post_extraction_canonicalization_disambiguation:
        print(f"Reused instances: {kwargs.get('stats').get('reused_entities')}")
    await visualize_graph(graph_visualization_path)


async def _run():
    prompt_path = os.path.join(Path(__file__).resolve().parent, "prompts", "prompt1.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        custom_prompt_text = f.read()

    df = pd.DataFrame()
    await main(
        example="example1",
        use_post_extraction_canonicalization_disambiguation=True,
        custom_prompt=custom_prompt_text,
        df=df,
    )
    await main(
        example="example1",
    )


if __name__ == "__main__":
    asyncio.run(_run())
