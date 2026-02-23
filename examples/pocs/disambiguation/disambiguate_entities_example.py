import asyncio
from typing import Optional

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from disambiguate_entities import disambiguate_entities_pipeline
from pathlib import Path
import os


async def main(
    example, use_poc, vector_search_limit: Optional[int] = None, custom_prompt: Optional[str] = None
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if use_poc else ''}cognify_disambiguate_{example}_result.html",
    )

    with open(
        os.path.join(Path(__file__).resolve().parent, "data/" + example + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()
    text = text.split("\n")
    for line in text:
        await cognee.add(line)
        if use_poc:
            kwargs = {"vector_search_limit": vector_search_limit}
            await disambiguate_entities_pipeline(custom_prompt=custom_prompt, **kwargs)
        else:
            await cognee.cognify()

    await visualize_graph(graph_visualization_path)


async def _run():
    with open("prompts/prompt1.txt", "r") as f:
        custom_prompt_text = f.read()

    for i in range(1, 5):
        example_id = str(i)
        await main(
            example="example" + example_id,
            use_poc=True,
            vector_search_limit=5,
            custom_prompt=custom_prompt_text,
        )
        await main(
            example="example" + example_id,
            use_poc=False,
        )


if __name__ == "__main__":
    asyncio.run(_run())
