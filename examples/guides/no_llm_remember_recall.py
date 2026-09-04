"""remember → recall with no LLM API key at all.

Graph and chunk summaries come from the local GLiNER2 model
(GRAPH_EXTRACTION_BACKEND=gliner), embeddings from fastembed running on CPU.
On this backend recall() defaults to CHUNKS (vector search, no LLM) and the
first-run environment check skips the LLM connection probe. Anything ending in
*_COMPLETION still needs an LLM to write the answer.

Requirements::

    pip install "cognee[gliner]" "fastembed<=0.8.0"

First run downloads the GLiNER model (~800 MB) and the bge-small embedding
model (~130 MB).
"""

import asyncio
import os

# Make sure no key leaks in from the shell: the point is to prove the pipeline
# runs without one.
for var in ("LLM_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(var, None)

os.environ.update(
    {
        "GRAPH_EXTRACTION_BACKEND": "gliner",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIMENSIONS": "384",
        "EMBEDDING_MAX_TOKENS": "512",
        # Per-turn feedback analysis is an LLM call; without it recall is LLM-free.
        "AUTO_FEEDBACK": "false",
    }
)

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

    # No query_type: on the gliner backend this is CHUNKS.
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
