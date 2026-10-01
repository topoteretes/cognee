from cognee.modules.retrieval.hybrid.context import (
    format_hybrid_context,
    format_hybrid_context_batch,
)


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


def test_preamble_and_passage_notes_are_rendered_only_when_given():
    retrieved = {"chunks": [{"id": "c1", "text": "Passage one"}, {"id": "c2", "text": "Two"}]}

    assert format_hybrid_context("", retrieved) == "## Relevant passages\nPassage one\n---\nTwo"
    assert (
        format_hybrid_context(
            "", retrieved, preamble="## Time window\nperiod", passage_notes={"c2": "time: 1950"}
        )
        == "## Time window\nperiod\n\n## Relevant passages\nPassage one\n---\ntime: 1950\nTwo"
    )
