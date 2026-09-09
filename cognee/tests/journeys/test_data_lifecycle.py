"""Journey 4: data lifecycle across the relational, vector and graph stores.

remember -> update a document -> forget one document -> forget the dataset ->
forget everything -> remember again. After every step the three stores must
agree with what the user did, and recall must reflect it.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.modules.search.types import SearchType
from cognee.tests.journeys import _support

DATASET = "journey_lifecycle"

DOC_A_V1 = (
    "Title: Meridian Kite Club\n\nThe Meridian Kite Club is chaired by Olamide Farquhar and flies "
    "from Gannet Hill every Sunday."
)
DOC_A_V2 = (
    "Title: Meridian Kite Club\n\nThe Meridian Kite Club is chaired by Ingrid Solvay and flies "
    "from Gannet Hill every Sunday."
)
DOC_B = (
    "Title: Bramblefield Orchard\n\nBramblefield Orchard grows the Winter Pippin apple and is "
    "run by the Okonkwo family."
)
DOC_C = (
    "Title: Saltmarsh Choir\n\nThe Saltmarsh Choir rehearses in the Corn Exchange under conductor "
    "Yusuf Haldane."
)


async def _chunk_search_text(query: str) -> str:
    results = await cognee.search(
        query_text=query, query_type=SearchType.CHUNKS, datasets=[DATASET]
    )
    return _support.result_text(results)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_update_forget_and_reremember_keep_all_stores_consistent(clean_env, default_user):
    # --- remember three documents ---------------------------------------------
    result = await cognee.remember([DOC_A_V1, DOC_B, DOC_C], dataset_name=DATASET)
    assert result.status == "completed"
    dataset_id = result.dataset_id
    from uuid import UUID

    dataset_id = UUID(str(dataset_id))

    full = await _support.snapshot_dataset(dataset_id, default_user)
    assert full.data_rows == 3, full
    assert full.graph_nodes > 0 and full.graph_edges > 0, full
    assert "farquhar" in await _chunk_search_text("Who chairs the Meridian Kite Club?")

    # --- update document A: the old fact must disappear, the new one appear ----
    rows = await cognee.datasets.list_data(dataset_id, default_user)
    doc_a = next(r for r in rows if "kite" in (r.name or "").lower() or _looks_like(r, "Meridian"))
    await cognee.update(data_id=doc_a.id, data=DOC_A_V2, dataset_id=dataset_id, user=default_user)

    after_update = await _support.snapshot_dataset(dataset_id, default_user)
    assert after_update.data_rows == 3, f"update changed the number of data rows: {after_update}"

    chunk_texts = await _support.vector_texts(dataset_id, default_user)
    assert any("solvay" in t for t in chunk_texts), "updated chunk was not embedded"
    assert not any("farquhar" in t for t in chunk_texts), "stale chunk survived the update"

    text = await _chunk_search_text("Who chairs the Meridian Kite Club?")
    assert "solvay" in text and "farquhar" not in text, (
        f"search still returns stale content: {text[:300]}"
    )

    # --- forget one document -----------------------------------------------------
    rows = await cognee.datasets.list_data(dataset_id, default_user)
    doc_b = next(r for r in rows if _looks_like(r, "Bramblefield"))
    await cognee.forget(data_id=doc_b.id, dataset_id=dataset_id, user=default_user)

    after_forget = await _support.snapshot_dataset(dataset_id, default_user)
    assert after_forget.data_rows == 2, after_forget
    assert after_forget.graph_nodes < after_update.graph_nodes, (
        f"graph did not shrink after forgetting a document: {after_update} -> {after_forget}"
    )
    chunk_texts = await _support.vector_texts(dataset_id, default_user)
    assert not any("pippin" in t for t in chunk_texts), "forgotten document's chunk still embedded"
    assert not any(
        "pippin" in n for n in await _support.graph_node_names(dataset_id, default_user)
    ), "forgotten document's entities still in the graph"
    assert "pippin" not in await _chunk_search_text("Which apple does Bramblefield Orchard grow?")

    # The two remaining documents are untouched.
    assert "haldane" in await _chunk_search_text("Who conducts the Saltmarsh Choir?")
    assert "solvay" in await _chunk_search_text("Who chairs the Meridian Kite Club?")

    # --- forget the whole dataset --------------------------------------------------
    await cognee.forget(dataset=DATASET, user=default_user)
    names = [d.name for d in await cognee.datasets.list_datasets(default_user)]
    assert DATASET not in names, f"dataset still listed after forget: {names}"

    # --- remember again into a fresh dataset, then forget everything ------------------
    again = await cognee.remember([DOC_B], dataset_name=DATASET)
    assert again.status == "completed"
    assert "pippin" in await _chunk_search_text("Which apple does Bramblefield Orchard grow?")

    await cognee.forget(everything=True, user=default_user)
    assert not await cognee.datasets.list_datasets(default_user), (
        "datasets survived forget(everything)"
    )

    # --- and the system is still usable afterwards ---------------------------------
    final = await cognee.remember([DOC_C], dataset_name=DATASET)
    assert final.status == "completed"
    assert "haldane" in await _chunk_search_text("Who conducts the Saltmarsh Choir?")
    assert "pippin" not in await _chunk_search_text(
        "Which apple does Bramblefield Orchard grow?"
    ), "content from before forget(everything) resurfaced"


def _looks_like(row, needle: str) -> bool:
    """Match a data row to its source text via any of the identifying columns."""
    needle = needle.lower()
    for attr in ("name", "label", "raw_data_location", "original_data_location"):
        value = getattr(row, attr, None)
        if isinstance(value, str) and needle in value.lower():
            return True
    location = getattr(row, "raw_data_location", None)
    if isinstance(location, str):
        try:
            from pathlib import Path

            path = Path(location.replace("file://", ""))
            if path.exists() and needle in path.read_text(errors="ignore").lower():
                return True
        except Exception:
            pass
    return False
