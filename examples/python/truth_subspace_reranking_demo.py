"""Truth-Subspace Reranking — runnable demo.

What it shows
-------------
A finished "session" teaches the system a preference (here: the user cares about
*coffee*, not tea). We distill that into a **truth subspace** (anchor vectors built
from the lessons), project every corpus chunk onto it, and store the per-chunk
coordinates on the graph node. At query time the HYBRID retriever reads those
coordinates and nudges ranking toward the learned preference.

The demo retrieves the SAME ambiguous query twice — once with truth weighting OFF
(exact baseline) and once ON — and prints a side-by-side rank diff so you can see
the coffee chunks rise.

Requirements
------------
- An LLM + embeddings provider configured (e.g. LLM_API_KEY in your .env). cognify
  and retrieval call the embedding/LLM APIs.

Run it
------
    python examples/python/truth_subspace_reranking_demo.py

It uses a dedicated dataset ("truth_subspace_demo") and does NOT prune, so it will
not touch your other cognee data. Re-running is safe (adds are content-addressed,
the subspace build is a full recompute).
"""

import asyncio
import os
import sys

# Run against THIS checkout of cognee even if a different copy is pip-installed,
# so the truth-subspace code in this working tree is the one that executes.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.retrieval.hybrid.results import payload, result_id
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.truth_subspace.build import build_truth_subspace
from cognee.modules.users.methods import get_default_user

DATASET = "truth_subspace_demo"
CORPUS_NODE_SET = ["beverages"]
LESSONS_NODE_SET = ["session_learnings"]  # the node set build_truth_subspace reads anchors from

# A small two-theme corpus. Each short doc becomes its own chunk.
CORPUS = [
    "Espresso is brewed by forcing hot water through finely ground coffee under high pressure.",
    "A pour-over coffee drips a slow stream of hot water over a paper filter of ground coffee.",
    "Cold brew coffee steeps coarse coffee grounds in cold water for twelve hours or more.",
    "A French press steeps coffee grounds in hot water, then a metal plunger separates them.",
    "Green tea is brewed with water below boiling to avoid a bitter, astringent flavor.",
    "Black tea is steeped in fully boiling water for three to five minutes before serving.",
    "Herbal tisanes are caffeine-free infusions of dried herbs, flowers, and dried fruit.",
    "Matcha is a powdered green tea whisked into hot water with a bamboo whisk until frothy.",
]

# What a finished session "learned" about the user. These become the truth anchors.
# They are about coffee, so coffee chunks align more strongly with the subspace.
LESSONS = [
    "The user is a dedicated coffee drinker who cares about espresso extraction and pour-over technique.",
    "We learned the user wants coffee brewing recommendations specifically, and is not interested in tea.",
    "For this user, coffee details — grind size, water temperature, and bloom time — matter most.",
]

QUERY = "How should I prepare my morning drink at home?"

# Number of chunks to rank.
TOP_K = len(CORPUS)


def _theme(text: str) -> str:
    coffee = ("coffee", "espresso", "pour-over", "french press", "cold brew")
    return "☕ coffee" if any(w in text.lower() for w in coffee) else "🍵 tea   "


def _snippet(text: str, width: int = 62) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


async def ranked_chunks(dataset_obj, query: str, use_truth_weight: bool):
    """Return the hybrid retriever's ranked chunk dicts, within the dataset's DB context."""
    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        retriever = HybridRetriever(
            chunks_top_k=TOP_K,
            entities_top_k=0,  # focus the demo on chunk-lane reranking
            facts_top_k=0,
            node_name=CORPUS_NODE_SET,  # rank only the corpus, not the lesson chunks
            use_truth_weight=use_truth_weight,
        )
        objects = await retriever.get_retrieved_objects(query=query)
    return objects.get("chunks", [])


def _text(chunk) -> str:
    return payload(chunk).get("text", "")


def print_ranking(title: str, chunks: list):
    print(f"\n{title}")
    print("  " + "-" * 74)
    for i, chunk in enumerate(chunks, 1):
        text = _text(chunk)
        print(f"  {i:>2}. {_theme(text)}  {_snippet(text)}")


def print_diff(baseline: list, truthful: list):
    base_rank = {result_id(c): i for i, c in enumerate(baseline, 1)}
    print("\nRANK CHANGE WITH TRUTH WEIGHTING ON (vs baseline)")
    print("  " + "-" * 74)
    for new_rank, chunk in enumerate(truthful, 1):
        cid = result_id(chunk)
        old = base_rank.get(cid)
        if old is None:
            delta = "  new"
        elif old > new_rank:
            delta = f"  ↑{old - new_rank}"
        elif old < new_rank:
            delta = f"  ↓{new_rank - old}"
        else:
            delta = "   ="
        text = _text(chunk)
        print(f"  {new_rank:>2}. {_theme(text)}  {_snippet(text, 52)}{delta}")


async def main():
    print("=" * 78)
    print("Truth-Subspace Reranking demo")
    print("=" * 78)

    # 1) Ingest the corpus and build the knowledge graph.
    print(f"\n[1/5] Adding {len(CORPUS)} corpus docs and cognifying (dataset='{DATASET}')…")
    await cognee.add(CORPUS, dataset_name=DATASET, node_set=CORPUS_NODE_SET)
    await cognee.cognify(datasets=[DATASET])

    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([DATASET], "write", user)
    dataset_obj = datasets[0]

    # 2) Baseline ranking — truth weighting OFF (exact current behavior).
    print(f"\n[2/5] Baseline retrieval (truth weighting OFF) for:\n      “{QUERY}”")
    baseline = await ranked_chunks(dataset_obj, QUERY, use_truth_weight=False)
    print_ranking("BASELINE RANKING", baseline)

    # 3) A finished session's learnings → seed the session_learnings node set.
    print(f"\n[3/5] Recording {len(LESSONS)} session learnings (favoring coffee)…")
    await cognee.add(LESSONS, dataset_name=DATASET, node_set=LESSONS_NODE_SET)
    await cognee.cognify(datasets=[DATASET])

    # 4) Build the truth subspace: anchors from lessons → coords on every corpus chunk.
    print("\n[4/5] Building the truth subspace (build_truth_subspace)…")
    result = await build_truth_subspace(dataset=DATASET, session_ids=None, user=user)
    print(f"      anchors={result['anchors']}  nodes_scored={result['nodes_scored']}")
    if result["anchors"] == 0 or result["nodes_scored"] == 0:
        print("      ⚠ No anchors or no scored nodes — truth weighting will be a no-op.")

    # 5) Truth-weighted ranking — same query, truth weighting ON.
    print("\n[5/5] Retrieval with truth weighting ON for the same query…")
    truthful = await ranked_chunks(dataset_obj, QUERY, use_truth_weight=True)
    print_ranking("TRUTH-WEIGHTED RANKING", truthful)

    print_diff(baseline, truthful)

    base_top = _theme(_text(baseline[0])) if baseline else "?"
    truth_top = _theme(_text(truthful[0])) if truthful else "?"
    print("\n" + "=" * 78)
    print(f"Top result moved from  {base_top.strip()}  →  {truth_top.strip()}")
    print("The learned coffee preference reshaped retrieval ordering. ✔")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
