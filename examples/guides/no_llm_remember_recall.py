"""remember → recall with no LLM API key at all.

Nothing to configure: with no LLM key and no embedding settings, cognee
extracts the graph and chunk summaries with the local GLiNER2 model
(GRAPH_EXTRACTOR=auto resolves to gliner) and embeds with fastembed on CPU.
With no usable LLM key, recall() defaults to CHUNKS (vector search, no LLM),
and a pipeline with no LLM task skips the first-run LLM connection probe.
Anything ending in *_COMPLETION still needs an LLM to write the answer.

Requirements::

    pip install "cognee[gliner]"

fastembed is a core dependency. First run downloads the GLiNER model
(~750 MB) and the bge-small embedding model (~67 MB).
"""

import asyncio
import os

# Make sure no key leaks in from the shell: the point is to prove the pipeline
# runs without one.
for var in ("LLM_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(var, None)

# Per-turn feedback analysis is an LLM call; without it recall is LLM-free.
os.environ["AUTO_FEEDBACK"] = "false"

import cognee  # noqa: E402  (environment must be set before the import)
from cognee import SearchType  # noqa: E402

TEXT = (
    "Marie Curie was born in Warsaw and worked at the University of Paris. "
    "She won the Nobel Prize in Physics in 1903 with Pierre Curie and Henri Becquerel."
)


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # remember() runs add + cognify and then improve(). Without session_ids,
    # improve() only runs the default enrichment (triplet/vector indexing) —
    # embeddings, no LLM — so it is safe to leave self_improvement on.
    await cognee.remember(TEXT, dataset_name="no_llm")

    # No query_type: with no usable LLM key this resolves to CHUNKS.
    results = await cognee.recall("Where was Marie Curie born?", datasets=["no_llm"], top_k=3)
    print(f"\ndefault ({results[0].search_type}): {len(results)} result(s)")
    for item in results:
        print("  -", item.text.replace("\n", " | "))

    # The GLiNER-built summaries are searchable too.
    results = await cognee.recall(
        "Where was Marie Curie born?",
        query_type=SearchType.SUMMARIES,
        datasets=["no_llm"],
        top_k=3,
    )
    print(f"\nSUMMARIES: {len(results)} result(s)")
    for item in results:
        print("  -", item.text.replace("\n", " | "))


if __name__ == "__main__":
    asyncio.run(main())
