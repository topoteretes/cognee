from unittest.mock import MagicMock

from cognee.modules.retrieval.hybrid.references import cite_hybrid_completions


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
