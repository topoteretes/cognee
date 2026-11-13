import json
from pathlib import Path

import pytest

from cognee.eval_framework.answer_generation import run_question_answering_module as qa_module
from cognee.eval_framework.answer_generation.answer_generation_executor import retriever_options
from cognee.eval_framework.integrity.checkpoint import AnswerCheckpoint, journal_path
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.tests.unit.eval_framework.integrity_helpers import (
    StubRetriever,
    make_answer,
    make_questions,
)


def test_resume_reuses_ok_answers_and_redoes_failed_ones(tmp_path):
    answers_path = str(tmp_path / "answers.json")
    first = AnswerCheckpoint(answers_path, make_questions(3), resume=False)
    first.record(make_answer(0))
    first.record(make_answer(1, "llm_error"))

    second = AnswerCheckpoint(answers_path, make_questions(3), resume=True)
    assert [q["question"] for q in second.pending()] == ["q1", "q2"]
    assert second.reused == 1


def test_without_resume_the_journal_restarts(tmp_path):
    answers_path = str(tmp_path / "answers.json")
    AnswerCheckpoint(answers_path, make_questions(2), resume=False).record(make_answer(0))
    fresh = AnswerCheckpoint(answers_path, make_questions(2), resume=False)
    assert len(fresh.pending()) == 2


def test_torn_last_journal_line_is_dropped_but_a_corrupt_middle_line_raises(tmp_path):
    answers_path = str(tmp_path / "answers.json")
    good = json.dumps(make_answer(0))
    with open(journal_path(answers_path), "w") as handle:
        handle.write(good + '\n{"question": "q1", "ans')
    resumed = AnswerCheckpoint(answers_path, make_questions(2), resume=True)
    assert [q["question"] for q in resumed.pending()] == ["q1"]
    # The torn tail was cut off the file, so the next append starts on a clean line.
    resumed.record(make_answer(1))
    assert len(Path(journal_path(answers_path)).read_text().splitlines()) == 2

    with open(journal_path(answers_path), "w") as handle:
        handle.write('{"broken\n' + good + "\n")
    with pytest.raises(RunIntegrityError, match="line 1"):
        AnswerCheckpoint(answers_path, make_questions(2), resume=True)


def test_assemble_raises_when_a_question_has_no_answer(tmp_path):
    checkpoint = AnswerCheckpoint(str(tmp_path / "a.json"), make_questions(2), resume=False)
    checkpoint.record(make_answer(0))
    with pytest.raises(RunIntegrityError, match="No answer was recorded"):
        checkpoint.assemble()


def test_duplicate_questions_are_each_answered_once(tmp_path):
    questions = make_questions(1) * 2
    checkpoint = AnswerCheckpoint(str(tmp_path / "a.json"), questions, resume=False)
    checkpoint.record(make_answer(0))
    checkpoint.record(make_answer(0))
    assert len(checkpoint.assemble()) == 2


@pytest.mark.asyncio
async def test_interrupted_then_resumed_run_asks_only_what_is_missing(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.json"
    answers_path = tmp_path / "answers.json"
    questions_path.write_text(json.dumps(make_questions(4)))
    params = {
        "answering_questions": True,
        "questions_path": str(questions_path),
        "answers_path": str(answers_path),
        "qa_engine": "stub",
    }

    async def no_db(_answers):
        return None

    monkeypatch.setattr(qa_module, "create_and_insert_answers_table", no_db)

    class Interrupted(StubRetriever):
        async def get_completion_from_context(self, query, retrieved_objects, context, **kwargs):
            if query == "q2":
                raise KeyboardInterrupt  # not an Exception: the run dies mid-way
            return await super().get_completion_from_context(
                query, retrieved_objects, context, **kwargs
            )

    monkeypatch.setitem(retriever_options, "stub", lambda **kwargs: Interrupted())
    with pytest.raises(KeyboardInterrupt):
        await qa_module.run_question_answering(params)
    assert not answers_path.exists()  # no half-written answers file

    resumed_retriever = StubRetriever()
    monkeypatch.setitem(retriever_options, "stub", lambda **kwargs: resumed_retriever)
    answers = await qa_module.run_question_answering({**params, "resume_verified": True})

    assert resumed_retriever.asked == ["q2", "q3"]
    assert [a["question"] for a in answers] == ["q0", "q1", "q2", "q3"]
    assert json.loads(answers_path.read_text()) == answers


def test_crash_during_resume_keeps_ok_answers(tmp_path, monkeypatch):
    answers_path = str(tmp_path / "answers.json")
    first = AnswerCheckpoint(answers_path, make_questions(3), resume=False)
    first.record(make_answer(0))
    first.record(make_answer(1))
    before = Path(journal_path(answers_path)).read_text()

    # Resuming must not rewrite the journal, so a crash while opening or appending loses nothing.
    def boom(self, record):
        raise OSError("disk full")

    monkeypatch.setattr(AnswerCheckpoint, "_append", boom)
    second = AnswerCheckpoint(answers_path, make_questions(3), resume=True)
    with pytest.raises(OSError):
        second.record(make_answer(2))
    assert Path(journal_path(answers_path)).read_text() == before

    monkeypatch.undo()
    third = AnswerCheckpoint(answers_path, make_questions(3), resume=True)
    assert [q["question"] for q in third.pending()] == ["q2"]


def _qa_params(tmp_path, questions=3):
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(make_questions(questions)))
    return {
        "answering_questions": True,
        "questions_path": str(questions_path),
        "answers_path": str(tmp_path / "answers.json"),
        "qa_engine": "stub",
    }


@pytest.mark.asyncio
async def test_unverified_resume_does_not_reuse_a_leftover_journal(tmp_path, monkeypatch):
    params = _qa_params(tmp_path)
    AnswerCheckpoint(params["answers_path"], make_questions(3), resume=False).record(make_answer(0))

    async def no_db(_answers):
        return None

    monkeypatch.setattr(qa_module, "create_and_insert_answers_table", no_db)
    retriever = StubRetriever()
    monkeypatch.setitem(retriever_options, "stub", lambda **kwargs: retriever)
    # `resume` alone (set by the user) is not enough; only the runner's verified flag reuses.
    await qa_module.run_question_answering({**params, "resume": True})
    assert retriever.asked == ["q0", "q1", "q2"]


@pytest.mark.asyncio
async def test_answer_errors_propagate_unless_a_failure_rate_is_set(tmp_path, monkeypatch):
    params = _qa_params(tmp_path)

    async def no_db(_answers):
        return None

    monkeypatch.setattr(qa_module, "create_and_insert_answers_table", no_db)
    monkeypatch.setitem(retriever_options, "stub", lambda **kwargs: StubRetriever(fail_on={"q1"}))
    with pytest.raises(RuntimeError, match="vector store unavailable"):
        await qa_module.run_question_answering(params)

    answers = await qa_module.run_question_answering({**params, "max_answer_failure_rate": 0.5})
    assert [a["status"] for a in answers] == ["ok", "retrieval_error", "ok"]
