import json
from unittest.mock import AsyncMock, patch

import pytest

from cognee.eval_framework.beam import preprocessed_runtime
from cognee.eval_framework.beam.preprocessed_runtime import (
    DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
    build_beam_preprocessed_conversation_corpus,
    ingest_preprocessed_corpus,
)
from cognee.eval_framework.benchmark_adapters.beam_preprocessed_adapter import (
    BEAMPreprocessedAdapter,
)
from cognee.modules.chunking.TextChunker import TextChunker


def _sample_beam_row():
    return {
        "conversation_id": "conv-1",
        "chat": [
            [
                {"role": "user", "content": "Hello", "time_anchor": "Jan-01-2024"},
                {
                    "role": "assistant",
                    "content": "Hi there. " + "memory " * 120,
                },
            ],
            [
                {"role": "user", "content": "What version do we use?"},
                {"role": "assistant", "content": "version: '3.8'\nname: CI"},
            ],
        ],
        "probing_questions": {
            "information_extraction": [
                {
                    "question": "What version is configured?",
                    "answer": "3.8",
                    "rubric": ["Mentions version 3.8"],
                    "difficulty": "easy",
                }
            ]
        },
    }


def test_beam_preprocessed_adapter_emits_multiple_documents():
    row = _sample_beam_row()

    with patch(
        "cognee.eval_framework.benchmark_adapters.beam_preprocessed_adapter.load_beam_row",
        return_value=row,
    ):
        adapter = BEAMPreprocessedAdapter(
            split="100K",
            conversation_index=0,
            preprocessed_max_chunk_size=60,
        )
        corpus, questions = adapter.load_corpus()

    assert len(corpus) > 1
    assert all(doc.startswith("[Session ") for doc in corpus)
    assert any("User: Hello" in doc for doc in corpus)
    assert any("Assistant:" in doc for doc in corpus)
    assert questions == [
        {
            "question": "What version is configured?",
            "answer": "3.8",
            "question_type": "information_extraction",
            "rubric": ["Mentions version 3.8"],
            "difficulty": "easy",
            "conversation_id": "conv-1",
        }
    ]


@pytest.mark.asyncio
async def test_ingest_preprocessed_corpus_batches_add_and_cognify_in_order():
    docs = [f"doc {index}" for index in range(12)]
    events = []

    async def record(event_name, *args, **kwargs):
        events.append((event_name, args, kwargs))

    async def prune_data():
        await record("prune_data")

    async def prune_system(metadata=True):
        await record("prune_system", metadata=metadata)

    async def add(*args, **kwargs):
        await record("add", *args, **kwargs)

    async def cognify(*args, **kwargs):
        await record("cognify", *args, **kwargs)

    with (
        patch.object(
            preprocessed_runtime.cognee.prune,
            "prune_data",
            new=AsyncMock(side_effect=prune_data),
        ),
        patch.object(
            preprocessed_runtime.cognee.prune,
            "prune_system",
            new=AsyncMock(side_effect=prune_system),
        ),
        patch.object(
            preprocessed_runtime.cognee,
            "add",
            new=AsyncMock(side_effect=add),
        ),
        patch.object(
            preprocessed_runtime.cognee,
            "cognify",
            new=AsyncMock(side_effect=cognify),
        ),
    ):
        await ingest_preprocessed_corpus(
            docs,
            dataset_name="beam_preprocessed_test",
            chunk_size=1200,
        )

    assert [event[0] for event in events] == [
        "prune_data",
        "prune_system",
        "add",
        "cognify",
        "add",
        "cognify",
    ]
    add_calls = [event for event in events if event[0] == "add"]
    assert [len(call[1][0]) for call in add_calls] == [
        DEFAULT_PREPROCESSED_DOCS_PER_ADD_BATCH,
        2,
    ]
    cognify_calls = [event for event in events if event[0] == "cognify"]
    assert all(call[2]["chunker"] is TextChunker for call in cognify_calls)
    assert all(call[2]["chunk_size"] == 1200 for call in cognify_calls)


@pytest.mark.asyncio
async def test_build_beam_preprocessed_conversation_corpus_writes_questions(tmp_path):
    corpus = ["[Session 1, Turn 1]\nUser: Hello\nAssistant: Hi there"]
    questions = [
        {
            "question": "What happened?",
            "answer": "Hi there",
            "question_type": "information_extraction",
            "conversation_id": "conv-1",
        }
    ]

    with (
        patch.object(
            BEAMPreprocessedAdapter,
            "load_corpus",
            return_value=(corpus, questions),
        ),
        patch(
            "cognee.eval_framework.beam.preprocessed_runtime.ingest_preprocessed_corpus",
            new=AsyncMock(),
        ) as mock_ingest,
        patch(
            "cognee.eval_framework.beam.preprocessed_runtime.create_and_insert_questions_table",
            new=AsyncMock(),
        ) as mock_insert,
    ):
        params = await build_beam_preprocessed_conversation_corpus(
            conversation_index=0,
            output_dir=tmp_path,
            split="100K",
            answering_questions=False,
            docs_per_add_batch=10,
            preprocessed_max_chunk_size=800,
            cognify_chunk_size=1200,
        )

    assert params["chunker"] is TextChunker
    assert params["chunk_size"] == 1200
    assert params["docs_per_add_batch"] == 10
    assert params["ingestion_mode"] == "batched_preprocessed"
    assert (
        json.loads((tmp_path / "beam_questions_conv0.json").read_text(encoding="utf-8"))
        == questions
    )
    mock_ingest.assert_awaited_once()
    mock_insert.assert_awaited_once_with(questions_payload=questions)
