"""Journey: a chunk search scoped to one source document (issue #6424).

``node_name`` with ``node_type=TextDocument`` names the documents to search in.
CHUNKS and RAG_COMPLETION must return that document's chunks only; a type that
cannot apply the filter must say so instead of answering unscoped.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.types import SearchType
from cognee.tests.journeys import _support

DATASET = "journey_document_scope"


def _chunks(results) -> list[dict]:
    """The chunk payloads of a CHUNKS search: one ``search_result`` list per dataset."""
    return [chunk for result in results for chunk in result["search_result"]]


DOCUMENTS = {
    "doc_a": "Apples are red fruit. Apples grow on trees.",
    "doc_b": "Bananas are yellow fruit. Bananas grow in bunches.",
}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_chunk_search_scoped_to_one_document(clean_env, default_user, tmp_path):
    paths = []
    for name, body in DOCUMENTS.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(body)
        paths.append(str(path))
    await cognee.add(paths, dataset_name=DATASET)
    await cognee.cognify([DATASET])

    unscoped = _chunks(
        await cognee.search("fruit", query_type=SearchType.CHUNKS, datasets=[DATASET], top_k=10)
    )
    assert sorted(chunk["document_name"] for chunk in unscoped) == ["doc_a", "doc_b"]

    scoped = _chunks(
        await cognee.search(
            "fruit",
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
            top_k=10,
            node_type=TextDocument,
            node_name=["doc_a"],
        )
    )
    assert [chunk["document_name"] for chunk in scoped] == ["doc_a"]
    assert "apples" in scoped[0]["text"].lower()

    answer = await cognee.search(
        "Which fruit is described?",
        query_type=SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        node_type=TextDocument,
        node_name=["doc_b"],
        only_context=True,
    )
    context = _support.content_text(answer)
    assert "bananas" in context and "apples" not in context

    # The default node_type is NodeSet: a document name is not a node-set tag.
    assert (
        _chunks(
            await cognee.search(
                "fruit", query_type=SearchType.CHUNKS, datasets=[DATASET], node_name=["doc_a"]
            )
        )
        == []
    )

    with pytest.raises(QueryValidationError, match="SUMMARIES does not apply node_name"):
        await cognee.search(
            "fruit",
            query_type=SearchType.SUMMARIES,
            datasets=[DATASET],
            node_type=TextDocument,
            node_name=["doc_a"],
        )
