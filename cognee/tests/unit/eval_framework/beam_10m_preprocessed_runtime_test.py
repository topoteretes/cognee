import json
from unittest.mock import AsyncMock, patch

import pytest

from cognee.eval_framework.beam.preprocessed_10m_runtime import (
    BEAM10MPreprocessedAdapter,
    answer_beam_10m_questions,
    build_beam_10m_preprocessed_conversation_corpus,
)
from cognee.modules.chunking.TextChunker import TextChunker


def _sample_10m_row():
    return {
        "conversation_id": "beam10m-conv-1",
        "chat": [
            {
                "plan-1": [
                    {
                        "batch_number": 1,
                        "turns": [
                            [
                                {"role": "user", "content": "Hello"},
                                {"role": "assistant", "content": "Hi there"},
                            ]
                        ],
                    }
                ],
                "plan-2": [
                    {
                        "batch_number": 4,
                        "turns": [
                            [
                                {"role": "user", "content": "What changed?"},
                                {"role": "assistant", "content": "version: '3.8'"},
                            ]
                        ],
                    }
                ],
            }
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


def test_beam_10m_preprocessed_adapter_groups_documents_by_plan():
    row = _sample_10m_row()

    with patch(
        "cognee.eval_framework.benchmark_adapters.beam_10m_preprocessed_adapter.load_beam_10m_row",
        return_value=row,
    ):
        adapter = BEAM10MPreprocessedAdapter(
            conversation_index=0,
            preprocessed_max_chunk_size=80,
        )
        plan_corpus, questions = adapter.load_plan_corpus()

    assert [plan_name for plan_name, _ in plan_corpus] == ["plan-1", "plan-2"]
    assert all(documents for _, documents in plan_corpus)
    assert plan_corpus[0][1][0].startswith("[PLAN-1, Session 1, Turn 1]")
    assert plan_corpus[1][1][0].startswith("[PLAN-2, Session 4, Turn 1]")
    assert questions == [
        {
            "question": "What version is configured?",
            "answer": "3.8",
            "question_type": "information_extraction",
            "rubric": ["Mentions version 3.8"],
            "difficulty": "easy",
            "conversation_id": "beam10m-conv-1",
        }
    ]


@pytest.mark.asyncio
async def test_build_beam_10m_preprocessed_corpus_prunes_once_and_ingests_each_plan(tmp_path):
    plan_documents = [
        ("plan-1", [f"plan-1 doc {index}" for index in range(12)]),
        ("plan-2", ["plan-2 doc 0"]),
    ]
    questions = [
        {
            "question": "What changed?",
            "answer": "version 3.8",
            "question_type": "information_extraction",
            "conversation_id": "beam10m-conv-1",
        }
    ]

    with (
        patch.object(
            BEAM10MPreprocessedAdapter,
            "load_plan_corpus",
            return_value=(plan_documents, questions),
        ),
        patch(
            "cognee.eval_framework.beam.preprocessed_10m_runtime.prune_preprocessed_ingestion_state",
            new=AsyncMock(),
        ) as mock_prune,
        patch(
            "cognee.eval_framework.beam.preprocessed_10m_runtime.ingest_preprocessed_corpus",
            new=AsyncMock(),
        ) as mock_ingest,
        patch(
            "cognee.eval_framework.beam.preprocessed_10m_runtime.create_and_insert_questions_table",
            new=AsyncMock(),
        ) as mock_insert,
    ):
        params = await build_beam_10m_preprocessed_conversation_corpus(
            conversation_index=0,
            output_dir=tmp_path,
            plans=["plan-1", "plan-2"],
            docs_per_add_batch=10,
            preprocessed_max_chunk_size=800,
            cognify_chunk_size=1200,
            chunks_per_batch=40,
        )

    assert params["chunker"] is TextChunker
    assert params["chunk_size"] == 1200
    assert params["ingestion_mode"] == "batched_preprocessed_10m"
    mock_prune.assert_awaited_once()
    assert mock_ingest.await_count == 2
    first_call = mock_ingest.await_args_list[0]
    second_call = mock_ingest.await_args_list[1]
    assert first_call.args[0] == plan_documents[0][1]
    assert first_call.kwargs["batch_label"] == "plan-1 preprocessed"
    assert first_call.kwargs["skip_prune"] is True
    assert second_call.args[0] == plan_documents[1][1]
    assert second_call.kwargs["batch_label"] == "plan-2 preprocessed"
    assert (
        json.loads((tmp_path / "beam10m_questions_conv0.json").read_text(encoding="utf-8"))
        == questions
    )
    mock_insert.assert_awaited_once_with(questions_payload=questions)


@pytest.mark.asyncio
async def test_answer_beam_10m_questions_uses_tuned_router(tmp_path):
    params = {
        "questions_path": str(tmp_path / "questions.json"),
        "answers_path": str(tmp_path / "answers.json"),
    }
    questions = [
        {
            "question": "What version is configured?",
            "answer": "3.8",
            "question_type": "information_extraction",
            "question_idx": 0,
            "conversation_id": "beam10m-conv-1",
        }
    ]
    answers = [{"question": questions[0]["question"], "answer": "3.8"}]
    (tmp_path / "questions.json").write_text(json.dumps(questions), encoding="utf-8")

    router = AsyncMock()
    router.answer_questions = AsyncMock(return_value=answers)

    with (
        patch(
            "cognee.eval_framework.beam.preprocessed_10m_runtime.BEAMRouter",
            return_value=router,
        ) as mock_router,
        patch(
            "cognee.eval_framework.beam.preprocessed_10m_runtime.create_and_insert_answers_table",
            new=AsyncMock(),
        ) as mock_insert,
    ):
        result = await answer_beam_10m_questions(params)

    assert result == answers
    mock_router.assert_called_once_with(
        top_k_overrides={"summarization": 150, "DEFAULT": 50},
        context_extension_rounds=8,
        wide_search_top_k=300,
        triplet_distance_penalty=4.0,
    )
    router.answer_questions.assert_awaited_once_with(questions)
    assert json.loads((tmp_path / "answers.json").read_text(encoding="utf-8")) == answers
    mock_insert.assert_awaited_once_with(answers)
