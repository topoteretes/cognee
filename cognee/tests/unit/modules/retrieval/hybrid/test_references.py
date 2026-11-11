from unittest.mock import MagicMock
from uuid import UUID

import pytest

from cognee.modules.retrieval.hybrid.references import (
    cite_hybrid_completions,
    hybrid_cited_chunks,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever

DATASET_ID = UUID("33333333-3333-3333-3333-333333333333")

ANSWER = "Revenue grew 12 percent."


def _chunk(text=None):
    chunk = MagicMock()
    chunk.id = "chunk-1"
    chunk.payload = {
        "document_name": "report.pdf",
        "chunk_index": 0,
        "text": text or "Revenue grew 12 percent year over year.",
    }
    return chunk


def test_disabled_returns_completions_verbatim():
    completions = [ANSWER]
    cited = cite_hybrid_completions(
        completions,
        {"chunks": [_chunk()]},
        enabled=False,
    )

    assert cited is completions
    assert cited == [ANSWER]


def test_enabled_appends_evidence_for_overlapping_chunk():
    cited = cite_hybrid_completions(
        [ANSWER],
        {"chunks": [_chunk()]},
        enabled=True,
    )

    assert len(cited) == 1
    assert cited[0].startswith(f"{ANSWER}\n\nEvidence:\n")
    assert "- chunk 1 of document report.pdf (chunk_id: chunk-1):" in cited[0]


def test_enabled_omits_evidence_when_chunk_does_not_overlap():
    cited = cite_hybrid_completions(
        ["Penguins live in Antarctica."],
        {"chunks": [_chunk()]},
        enabled=True,
    )

    assert cited == ["Penguins live in Antarctica."]


def test_non_dict_retrieved_objects_leave_completions_verbatim():
    cited = cite_hybrid_completions([ANSWER], [_chunk()], enabled=True)

    assert cited == [ANSWER]


def test_batch_cites_each_completion_from_its_own_chunks():
    cited = cite_hybrid_completions(
        [ANSWER, "Penguins live in Antarctica."],
        [
            {"chunks": [_chunk()]},
            {"chunks": [_chunk(text="Penguins live in Antarctica on ice.")]},
        ],
        enabled=True,
    )

    assert cited[0].startswith(f"{ANSWER}\n\nEvidence:\n")
    assert cited[1].startswith("Penguins live in Antarctica.\n\nEvidence:\n")


def test_batch_completion_without_matching_entry_is_untouched():
    cited = cite_hybrid_completions(
        [ANSWER, "second"],
        [{"chunks": [_chunk()]}],
        enabled=True,
    )

    assert "Evidence:" in cited[0]
    assert cited[1] == "second"


def _doc_chunk(chunk_id, data_id, text):
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.score = 0.2
    chunk.payload = {
        "document_name": f"{data_id}.txt",
        "document_id": data_id,
        "chunk_index": 0,
        "text": text,
    }
    return chunk


VAULT = _doc_chunk("chunk-vault", "data-vault", "The vault recovery phrase is amber-falcon-42.")
COFFEE = _doc_chunk("chunk-coffee", "data-coffee", "The coffee machine is cleaned every Friday.")


def test_cited_chunks_are_the_chunks_the_evidence_block_cites():
    answer = "The vault recovery phrase is amber-falcon-42."
    retrieved = {"chunks": [COFFEE, VAULT]}

    assert hybrid_cited_chunks([answer], retrieved) == [VAULT]
    assert "data_id: data-vault" in cite_hybrid_completions([answer], retrieved, enabled=True)[0]


def test_cited_chunks_follow_each_batch_completion_and_skip_non_strings():
    cited = hybrid_cited_chunks(
        ["amber-falcon-42", {"structured": True}],
        [{"chunks": [VAULT, COFFEE]}, {"chunks": [COFFEE]}],
    )

    assert cited == [VAULT]


@pytest.mark.asyncio
async def test_retriever_reports_cited_chunks_as_structured_evidence():
    retriever = HybridRetriever(include_references=True)
    retrieved = {"chunks": [COFFEE, VAULT]}
    await retriever.append_references(["The vault recovery phrase is amber-falcon-42."], retrieved)

    evidence = retriever.get_context_evidence(retrieved, dataset_id=DATASET_ID)

    assert [(reference.data_id, reference.role) for reference in evidence] == [
        ("data-vault", "supports_assertion")
    ]
    assert evidence[0].dataset_id == str(DATASET_ID)


@pytest.mark.asyncio
async def test_retriever_reports_no_evidence_without_include_references():
    retriever = HybridRetriever(include_references=False)
    retrieved = {"chunks": [VAULT]}
    await retriever.append_references(["amber-falcon-42"], retrieved)

    assert retriever.get_context_evidence(retrieved, dataset_id=DATASET_ID) == []
