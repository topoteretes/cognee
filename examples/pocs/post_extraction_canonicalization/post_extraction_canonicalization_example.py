import asyncio
import time
import cognee
import os

from typing import Optional, Callable, Awaitable, Any
from cognee.api.v1.visualize.visualize import visualize_graph
from pathlib import Path
from examples.pocs.post_extraction_canonicalization.post_extraction_canonicalization import (
    post_extraction_canonicalization,
)


async def main(
    example,
    post_extraction_canonicalization_fun: (
        Callable[[str, int, bool, str | None], Awaitable[Any]] | None
    ) = None,
    custom_prompt: Optional[str] = None,
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if post_extraction_canonicalization_fun else ''}cognify_disambiguate_{example}_result.html",
    )

    parts_dir = Path(__file__).resolve().parent / "data" / example

    if post_extraction_canonicalization_fun:
        await post_extraction_canonicalization_fun(parts_dir, custom_prompt)
    else:
        start = time.perf_counter()
        for part in sorted(parts_dir.glob("part_*.txt")):
            print(part)
            text = part.read_text(encoding="utf-8").replace("\n", " ")
            await cognee.add(text)
            await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt)
            elapsed = time.perf_counter() - start
            print(f"Elapsed: {elapsed:.6f} seconds")

    await visualize_graph(graph_visualization_path)


async def _run():
    prompt_path = os.path.join(Path(__file__).resolve().parent, "prompts", "prompt2.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        custom_prompt_text = f.read()

    await main(
        example="example2",
        post_extraction_canonicalization_fun=post_extraction_canonicalization,
        custom_prompt=custom_prompt_text,
    )
    # await main(
    #     example="example1",
    # )


if __name__ == "__main__":
    asyncio.run(_run())
