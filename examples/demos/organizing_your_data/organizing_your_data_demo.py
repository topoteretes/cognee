"""Organizing your data: one dataset, node sets, or separate datasets.

Run with:

    uv run python examples/demos/organizing_your_data/organizing_your_data_demo.py

The scenario: you feed cognee a mix of content types — technical documentation
AND sales-call transcripts — and answers start blending the two. A question
about API rate limits comes back seasoned with whatever a sales rep promised a
customer on a call. The graph is doing its job (linking related facts); the
problem is that everything lives in one undifferentiated pile.

This demo ingests the same small corpus (two docs, two call transcripts from
``data/``) three ways and asks the same questions each time:

    (1) ALL DATA IN ONE DATASET       simplest; retrieval sees everything at
                                      once — this reproduces the confusion
    (2) ONE DATASET, NODE SETS        every item tagged with one or more node
                                      sets: soft, overlappable groups inside
                                      one shared graph; recall can be scoped
                                      per query, cross-domain links survive
    (3) SEPARATE DATASETS             tech vs sales behind a hard boundary
        (+ node sets inside each)     (own permissions, isolated storage,
                                      independent forget); node sets still
                                      slice finer inside each dataset

Rule of thumb: node sets are tags, datasets are walls.

Requires a configured LLM provider (see CLAUDE.md). Wording of answers varies
by model.
"""

import asyncio
from pathlib import Path

import cognee
from cognee import SearchType

DATA_DIR = Path(__file__).resolve().parent / "data"

TECH_DOCS = [
    str(DATA_DIR / "api_reference.md"),
    str(DATA_DIR / "architecture_guide.md"),
]
SALES_CALL_INITECH = str(DATA_DIR / "sales_call_initech.txt")
SALES_CALL_HOOLI = str(DATA_DIR / "sales_call_hooli.txt")

RATE_LIMIT_QUERY = "What rate limits does the Acme API enforce?"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def show(query: str, scope: str, results, note: str = "") -> None:
    """Print one recall as a comparable QUERY / SCOPE / ANSWER block."""
    print(f"\n  QUERY : {query}", flush=True)
    print(f"  SCOPE : {scope}", flush=True)
    if note:
        print(f"  NOTE  : {note}", flush=True)
    for entry in results:
        dataset = getattr(entry, "dataset_name", None)
        prefix = f"[{dataset}] " if dataset else ""
        text = str(entry.text).replace("\n", "\n          ")
        print(f"  ANSWER: {prefix}{text}", flush=True)


async def section_1_everything_in_one_dataset():
    banner("(1) ALL DATA IN ONE DATASET — the setup that confuses the LLM")

    await cognee.remember(
        TECH_DOCS + [SALES_CALL_INITECH, SALES_CALL_HOOLI], self_improvement=False
    )

    answer = await cognee.recall(RATE_LIMIT_QUERY, query_type=SearchType.HYBRID_COMPLETION)
    show(
        RATE_LIMIT_QUERY,
        "everything (no filter)",
        answer,
        note="docs and sales calls share one retrieval pool — the documented limit "
        "(100/min) competes with the rep's 'unlimited requests' promise",
    )


async def section_2_one_dataset_with_node_sets():
    banner("(2) ONE DATASET, SEPARATE NODE SETS — tags inside one shared graph")

    # An item can carry several tags at once: everything here is also tagged
    # "acme_api", so a shared grouping cuts across the docs/calls split.
    await cognee.remember(TECH_DOCS, node_set=["tech_docs", "acme_api"], self_improvement=False)
    await cognee.remember(
        [SALES_CALL_INITECH, SALES_CALL_HOOLI],
        node_set=["sales_calls", "acme_api"],
        self_improvement=False,
    )

    docs_answer = await cognee.recall(
        RATE_LIMIT_QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        node_name=["tech_docs"],
    )
    show(RATE_LIMIT_QUERY, "node_name=['tech_docs']", docs_answer, note="the documented truth only")

    sales_query = "What did we promise customers about rate limits?"
    sales_answer = await cognee.recall(
        sales_query,
        query_type=SearchType.HYBRID_COMPLETION,
        node_name=["sales_calls"],
    )
    show(sales_query, "node_name=['sales_calls']", sales_answer, note="what the reps actually said")

    # The graph is still ONE graph — cross-domain questions work when you
    # want them to, by simply not filtering.
    cross_query = "Where do our sales promises contradict the technical documentation?"
    cross_answer = await cognee.recall(cross_query, query_type=SearchType.GRAPH_COMPLETION)
    show(
        cross_query,
        "everything (no filter, on purpose)",
        cross_answer,
        note="a cross-domain question spanning both groups",
    )


async def section_3_separate_datasets():
    banner("(3) SEPARATE DATASETS (tech vs sales) — hard walls, node sets inside")

    # Graphs are built independently: a recall scoped to one dataset can never
    # surface content from the other. Node sets still apply INSIDE a dataset —
    # here the sales dataset tags each call by account.
    await cognee.remember(
        TECH_DOCS,
        dataset_name="tech_docs",
        node_set=["api_reference"],
        self_improvement=False,
    )
    await cognee.remember(
        SALES_CALL_INITECH,
        dataset_name="sales_calls",
        node_set=["initech"],
        self_improvement=False,
    )
    await cognee.remember(
        SALES_CALL_HOOLI,
        dataset_name="sales_calls",
        node_set=["hooli"],
        self_improvement=False,
    )

    docs_answer = await cognee.recall(
        RATE_LIMIT_QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=["tech_docs"],
    )
    show(RATE_LIMIT_QUERY, "datasets=['tech_docs']", docs_answer, note="sales calls cannot leak in")

    initech_query = "What was discussed with Initech?"
    initech_answer = await cognee.recall(
        initech_query,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=["sales_calls"],
        node_name=["initech"],
    )
    show(
        initech_query,
        "datasets=['sales_calls'] + node_name=['initech']",
        initech_answer,
        note="one account's calls only",
    )

    # Caution: recall() without `datasets` spans ALL datasets you can read —
    # separating data at write time is not enough, queries must opt into the
    # scope they want.
    unscoped_answer = await cognee.recall(RATE_LIMIT_QUERY, query_type=SearchType.HYBRID_COMPLETION)
    show(
        RATE_LIMIT_QUERY,
        "everything (no dataset filter)",
        unscoped_answer,
        note="spans both datasets again — scope must be chosen per query",
    )


async def main():
    await cognee.forget(everything=True)
    await section_1_everything_in_one_dataset()

    await cognee.forget(everything=True)
    await section_2_one_dataset_with_node_sets()

    await cognee.forget(everything=True)
    await section_3_separate_datasets()

    banner("TAKEAWAY")
    print(
        "  Node sets are tags: soft groups inside one shared graph — scoped AND\n"
        "  cross-domain questions both work. Datasets are walls: own permissions,\n"
        "  isolated storage, independent forget — content never crosses over.\n"
        "  Mixing docs and sales calls? Start with node sets; move the domains\n"
        "  into separate datasets when they must never contaminate each other.",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
