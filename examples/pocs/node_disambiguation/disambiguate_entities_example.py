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
    use_poc,
    vector_search_limit: Optional[int] = None,
    custom_prompt: Optional[str] = None,
    df: Optional[DataFrame] = None,
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if use_poc else ''}cognify_disambiguate_{example}_result.html",
    )

    parts_dir = Path(__file__).resolve().parent / "data" / example
    kwargs = {"vector_search_limit": vector_search_limit}
    if use_poc:
        kwargs["use_poc"] = use_poc
        kwargs["df"] = df
    for part in sorted(parts_dir.glob("part_*.txt")):
        print(part)
        text = part.read_text(encoding="utf-8")
        await cognee.add(text)
        await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt, **kwargs)

    await visualize_graph(graph_visualization_path)


async def _run():
    with open("prompts/prompt1.txt", "r") as f:
        custom_prompt_text = f.read()

    for i in range(1, 2):
        example_id = str(i)
        df = pd.DataFrame()
        await main(
            example="example" + example_id,
            use_poc=True,
            vector_search_limit=5,
            custom_prompt=custom_prompt_text,
            df=df,
        )
        await main(
            example="example" + example_id,
            use_poc=False,
        )


if __name__ == "__main__":
    asyncio.run(_run())
