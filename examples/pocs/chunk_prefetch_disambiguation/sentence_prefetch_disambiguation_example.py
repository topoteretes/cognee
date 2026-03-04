import asyncio
from typing import Optional

from pandas import DataFrame

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from pathlib import Path
import os
import pandas as pd


async def main(
    example,
    use_chunk_prefetch_disambiguation: Optional[bool] = False,
    vector_search_limit: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    df: Optional[DataFrame] = None,
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if use_chunk_prefetch_disambiguation else ''}cognify_disambiguate_{example}_result.html",
    )

    parts_dir = Path(__file__).resolve().parent / "data" / example
    kwargs = {"vector_search_limit": vector_search_limit}
    if use_chunk_prefetch_disambiguation:
        kwargs["use_chunk_prefetch_disambiguation"] = use_chunk_prefetch_disambiguation
        kwargs["df"] = df
        kwargs["stats"] = {"reused_entities": 0}

    for part in sorted(parts_dir.glob("part_*.txt")):
        print(part)
        lines = part.read_text(encoding="utf-8").split("\n")
        for line in lines:
            await cognee.add(line)
        await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)

    if use_chunk_prefetch_disambiguation:
        print(f"Reused instances: {kwargs.get('stats').get('reused_entities')}")
    await visualize_graph(graph_visualization_path)


async def _run():
    prompt_path = os.path.join(Path(__file__).resolve().parent, "prompts", "prompt1.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        custom_prompt_text = f.read()

    for i in range(1, 2):
        example_id = str(i)
        df = pd.DataFrame()
        await main(
            example="example" + example_id,
            use_chunk_prefetch_disambiguation=True,
            vector_search_limit=5,
            custom_prompt=custom_prompt_text,
            df=df,
        )
        await main(
            example="example" + example_id,
        )


if __name__ == "__main__":
    asyncio.run(_run())
