from cognee.modules.retrieval.hybrid.context import format_hybrid_context_batch


def test_format_hybrid_context_batch_zips_per_query():
    contexts = format_hybrid_context_batch(
        ["## Global context\nWorld", ""],
        [
            {"chunks": [{"id": "c1", "text": "Passage one"}], "entities": []},
            {
                "chunks": [],
                "entities": [{"id": "e1", "name": "Entity", "edges": []}],
            },
        ],
    )

    assert contexts == [
        "## Global context\nWorld\n\n## Relevant passages\nPassage one",
        "## Relevant entities\n### Entity",
    ]


def test_format_hybrid_context_batch_handles_empty_inputs():
    assert format_hybrid_context_batch([], []) == []
