"""End-to-end demo of truth centroid slots changing HybridRetriever ranking.

This demo uses real cognee ingestion, cognification, vector search, graph state,
and embeddings. It skips only the live session flow: instead, it writes distilled
learnings directly into the ``session_learnings`` node set.

Run from the repository root:

    uv run python examples/demos/truth_centroid_slots_demo.py
"""

import asyncio
import os
import pathlib
import sys

os.environ.setdefault("COGNEE_CLI_MODE", "true")
os.environ.setdefault("COGNEE_LOG_FILE", "false")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LOG_LEVEL", "ERROR")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.retrieval.hybrid.results import payload
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.truth_subspace import align
from cognee.modules.truth_subspace.build import build_truth_subspace
from cognee.modules.truth_subspace.centroids import load_centroids
from cognee.modules.users.methods import get_default_user


DATASET = "truth_centroid_slots_demo"
CORPUS_NODE_SET = ["truth_demo_corpus"]
LEARNINGS_NODE_SET = ["session_learnings"]
K = 2

QUERY = "How should I prepare a warm drink at home?"

DOCUMENTS = [
    "Home warm drink preparation guide for coffee: grind size, bloom time, and water temperature.",  # A
    "Home warm drink preparation guide for espresso: dose, pressure, extraction time, and crema.",  # B
    "Home warm drink preparation guide for green tea: steeping time, cooler water, and bitter notes.",  # C
    "Home warm drink preparation guide for herbal tea: chamomile, mint, flowers, and longer steeping.",  # D
]

LEARNING_BATCHES = [
    (
        "Batch 1: accepted coffee learnings",
        [
            "# Session learning — 2026-06-26 (session demo-coffee)\n\nThe user cares about coffee brewing, espresso extraction, grind size, and pour-over technique. (Learned while resolving a warm drink recommendation where coffee details were preferred.)",
            "# Session learning — 2026-06-26 (session demo-coffee)\n\nFor this user, coffee preparation details should be preferred over tea background. (Learned while comparing coffee and tea guidance for the same user intent.)",
        ],
    ),
    (
        "Batch 2: accepted tea learnings",
        [
            "# Session learning — 2026-06-26 (session demo-tea)\n\nThe user is now focused on tea steeping, herbal infusions, chamomile, and water temperature. (Learned after the user's warm drink questions shifted toward tea preparation.)",
            "# Session learning — 2026-06-26 (session demo-tea)\n\nFor this user, tea preparation details should be preferred for warm drink questions. (Learned while revising the durable preference from later session evidence.)",
        ],
    ),
]


def configure_demo_storage() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    cognee.config.set_graph_database_provider("kuzu")
    cognee.config.set_vector_db_provider("lancedb")
    cognee.config.system_root_directory(str(repo_root / ".cognee_system/truth_slots_demo"))
    cognee.config.data_root_directory(str(repo_root / ".data_storage/truth_slots_demo"))


async def hybrid_order(dataset_obj, use_truth_weight: bool) -> list[str]:
    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        retriever = HybridRetriever(
            chunks_top_k=len(DOCUMENTS),
            entities_top_k=0,
            facts_top_k=0,
            text_summaries_top_k=0,
            node_name=CORPUS_NODE_SET,
            use_truth_weight=use_truth_weight,
        )
        retrieved = await retriever.get_retrieved_objects(query=QUERY)
    return [
        _label_for_text(payload(chunk).get("text", "")) for chunk in retrieved.get("chunks", [])
    ]


async def graph_truth_table(dataset_obj):
    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        vector_engine = await get_vector_engine_async()
        graph_engine = await get_graph_engine()
        centroids = await load_centroids(vector_engine, str(dataset_obj.id), K)
        nodes, _edges = await graph_engine.get_graph_data()

        labeled_node_ids = []
        for node_id, node_data in nodes:
            if not isinstance(node_data, dict) or node_data.get("type") != DocumentChunk.__name__:
                continue
            label = _label_for_text(str(node_data.get("text") or ""))
            if label == "?":
                continue
            labeled_node_ids.append((str(node_id), label))

        node_ids = [node_id for node_id, _label in labeled_node_ids]
        truth_state = await graph_engine.get_node_truth_state(node_ids)
        rows = []
        for node_id, label in labeled_node_ids:
            state = truth_state.get(node_id, {})
            coords = list(state.get("truth_alignment") or [])
            rows.append(
                {
                    "label": label,
                    "slot_0": coords[0] if len(coords) > 0 else None,
                    "slot_1": coords[1] if len(coords) > 1 else None,
                    "epoch": state.get("truth_epoch"),
                }
            )

    rows.sort(key=lambda row: row["label"])
    return centroids, rows


def _label_for_text(text: str) -> str:
    normalized = " ".join(text.split())
    for index, document in enumerate(DOCUMENTS):
        document_text = " ".join(document.split())
        if document_text in normalized or normalized in document_text:
            return chr(65 + index)
    return "?"


def print_documents() -> None:
    print("Documents")
    print("---------")
    for index, document in enumerate(DOCUMENTS):
        print(f"{chr(65 + index)}. {document}")


def print_order(title: str, order: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(" > ".join(order) if order else "[no chunks retrieved]")


async def slot_summaries(centroids, learning_texts: list[str]) -> dict[int, str]:
    if not learning_texts:
        return {}

    embedding_engine = get_embedding_engine()
    learning_vectors = await embedding_engine.embed_text(learning_texts)
    summaries = {}
    for centroid in centroids:
        nearest_index = max(
            range(len(learning_vectors)),
            key=lambda index: align.cosine(centroid.centroid, learning_vectors[index]),
        )
        summaries[centroid.slot] = _shorten(learning_texts[nearest_index], 72)
    return summaries


def print_slots(centroids, summaries: dict[int, str]) -> None:
    print("\nTruth slots")
    print("-----------")
    for slot in range(K):
        centroid = next((item for item in centroids if item.slot == slot), None)
        if centroid is None:
            print(f"slot {slot}: [empty]")
            continue
        summary = summaries.get(slot, "[no nearest learning]")
        print(f"slot {slot}: count={centroid.count} epoch={centroid.truth_epoch}")
        print(f"        nearest learning: {summary}")


def print_truth_table(rows: list[dict]) -> None:
    print("\nGraph values stored on DocumentChunk nodes")
    print("-----------------------------------------")
    print("doc | slot 0 | slot 1 | epoch")
    print("----|--------|--------|------")
    for row in rows:
        slot_0 = _format_coord(row["slot_0"])
        slot_1 = _format_coord(row["slot_1"])
        print(f"{row['label']}   | {slot_0:>6} | {slot_1:>6} | {row['epoch']}")


def _format_coord(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _shorten(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


async def main() -> None:
    configure_demo_storage()
    print("Resetting isolated demo store...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print_documents()
    print(f"\nQuery: {QUERY}")

    print("\nIngesting documents A-D and running cognify...")
    await cognee.add(DOCUMENTS, dataset_name=DATASET, node_set=CORPUS_NODE_SET)
    await cognee.cognify(datasets=[DATASET])
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([DATASET], "write", user)
    dataset_obj = datasets[0]

    print_order("Baseline HybridRetriever order", await hybrid_order(dataset_obj, False))

    accepted_learnings = []
    for title, learnings in LEARNING_BATCHES:
        print("\n" + title)
        print("=" * len(title))
        accepted_learnings.extend(learnings)
        for learning in learnings:
            print(f"learning: {learning}")

        print("Cognifying learnings and rebuilding truth slots...")
        await cognee.add(learnings, dataset_name=DATASET, node_set=LEARNINGS_NODE_SET)
        await cognee.cognify(datasets=[DATASET])
        result = await build_truth_subspace(
            dataset=DATASET,
            session_ids=None,
            user=user,
            k=K,
        )
        print(
            f"build result: slots={result['anchors']} "
            f"nodes_scored={result['nodes_scored']} epoch={result['truth_epoch']}"
        )

        centroids, rows = await graph_truth_table(dataset_obj)
        summaries = await slot_summaries(centroids, accepted_learnings)
        print_slots(centroids, summaries)
        print_truth_table(rows)
        print_order(
            "HybridRetriever order with truth weighting",
            await hybrid_order(dataset_obj, True),
        )


if __name__ == "__main__":
    asyncio.run(main())
