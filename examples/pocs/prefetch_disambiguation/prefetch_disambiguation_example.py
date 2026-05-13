import asyncio
import time
from typing import Optional, Callable, Any, Awaitable

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from pathlib import Path
import os
import nltk
from nltk.tokenize import sent_tokenize

from examples.pocs.prefetch_disambiguation.prefetch_disambiguation import (
    prefetch_disambiguation,
)


async def main(
    example,
    prefetch_disambiguation_fun: (
        Callable[[str, int, bool, str | None], Awaitable[Any]] | None
    ) = None,
    split_by_sentence: Optional[bool] = False,
    vector_search_limit: Optional[int] = None,
    custom_prompt: Optional[str] = None,
):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    graph_visualization_path = os.path.join(
        os.path.dirname(__file__),
        f"results/{'poc_' if prefetch_disambiguation_fun else ''}cognify_disambiguate_{example}_result.html",
    )

    parts_dir = Path(__file__).resolve().parent / "data" / example

    if prefetch_disambiguation_fun:
        await prefetch_disambiguation_fun(
            parts_dir, vector_search_limit, split_by_sentence, custom_prompt
        )
    else:
        for part in sorted(parts_dir.glob("part_*.txt")):
            print(part)
            text = part.read_text(encoding="utf-8").replace("\n", " ")
            if split_by_sentence:
                text = list(dict.fromkeys(sent_tokenize(text)))
            start = time.perf_counter()
            await cognee.add(text)
            await cognee.cognify(chunk_size=1024, custom_prompt=custom_prompt)
            elapsed = time.perf_counter() - start
            print(f"Elapsed: {elapsed:.6f} seconds")
    await visualize_graph(graph_visualization_path)


async def _run():
    prompt_path = os.path.join(Path(__file__).resolve().parent, "prompts", "prompt3.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        custom_prompt_text = f.read()

    await main(
        example="example2",
        prefetch_disambiguation_fun=prefetch_disambiguation,
        # split_by_sentence=True,
        vector_search_limit=40,
        custom_prompt=custom_prompt_text,
    )


if __name__ == "__main__":
    nltk.download("punkt_tab")
    asyncio.run(_run())
