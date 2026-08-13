"""Guards on the recall-coverage judge: call ordering, and what costs nothing.

The judge's whole design is about which calls are *not* made, and each of those
is a number's honesty rather than a saving:

* an **empty context scores 0 with no LLM call at all** — no context, no
  coverage, which is a definition rather than an approximation;
* a row that **scored 0 makes no completion call**, because there is nothing to
  have answered from and a generated answer would only be the model's own
  knowledge; ``answer`` is ``null`` there;
* the coverage prompt **never sees an answer**, because a fluent answer reads as
  good coverage even when the context was thin;
* there is **one judgement per row**, on a 0-10 scale — the answer that follows it
  is a generation nothing is scored from, so there is no second verdict to
  disagree with the number every average is a mean of.

Every LLM call is faked. ``MOCK_EMBEDDING``-style stand-ins are not used here:
this module makes no embedding calls at all.
"""

import asyncio
import importlib
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.judge import (
    MIN_JUDGE_SCORE,
    CoverageAnswer,
    JudgedRow,
    coverage_model,
    judge_row,
    judge_rows,
    score_context_coverage,
)
from cognee.modules.recall_coverage.replay import ReplayedRow
from cognee.modules.recall_coverage.types import CoverageParams, QuestionSource

judge_module = importlib.import_module("cognee.modules.recall_coverage.judge")

COVERAGE = "ContextCoverage"
ANSWER = "CoverageAnswer"


def _params(**overrides) -> CoverageParams:
    return CoverageParams.from_config(RecallCoverageConfig(_env_file=None), **overrides)


def _question(text: str = "Where are the runbooks?") -> DedupedQuestion:
    return DedupedQuestion(
        text=text,
        user_id=uuid4(),
        dataset_id=None,
        source=QuestionSource.OBSERVED.value,
        relevance=1,
        first_asked_at=None,
        last_asked_at=None,
        curated_question_id=None,
        canonical_index=0,
        ask_indices=[0],
        query_ids=[],
    )


def _replayed(context="the runbooks live in infra-docs", *, error=None) -> ReplayedRow:
    return ReplayedRow(
        retrieval_context=context, dataset_name="infra-docs", payload_count=1, error=error
    )


class _FakeLLM:
    """Dispatches on the requested ``response_model``, recording call order."""

    def __init__(self, *, score=4, reason="context has it", answer="In infra-docs."):
        self.calls: list[str] = []
        self.prompts: list[dict] = []
        self._score = score
        self._reason = reason
        self._answer = answer

    def __call__(self, *, text_input, system_prompt, response_model, **kwargs):
        name = response_model.__name__
        self.calls.append(name)
        self.prompts.append(
            {"model": name, "text_input": text_input, "system_prompt": system_prompt}
        )

        if name == COVERAGE:
            return response_model(score=self._score, reason=self._reason)
        if name == ANSWER:
            return response_model(answer=self._answer)
        raise AssertionError(f"unexpected response model {name}")


def _patched(fake: _FakeLLM):
    return patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=fake,
    )


# --------------------------------------------------------------------------
# The calls that must not happen
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_context_scores_zero_with_no_llm_call_at_all():
    for empty in (None, ""):
        with patch.object(
            judge_module.LLMGateway, "acreate_structured_output", new_callable=AsyncMock
        ) as mock_llm:
            row = await judge_row(_question().text, _replayed(empty), params=_params())

        mock_llm.assert_not_called()
        assert row.coverage_score == MIN_JUDGE_SCORE == 0
        # Nothing was retrieved, so there is nothing an answer could come from.
        assert row.answer is None
        assert row.error is None


@pytest.mark.asyncio
async def test_a_row_scoring_zero_makes_no_completion_call():
    fake = _FakeLLM(score=0, reason="nothing relevant retrieved")

    with _patched(fake):
        row = await judge_row(_question().text, _replayed(), params=_params())

    # Exactly one call, and it is the coverage call: no completion follows a zero.
    assert fake.calls == [COVERAGE]
    assert row.coverage_score == 0
    assert row.answer is None
    assert row.coverage_reason == "nothing relevant retrieved"


@pytest.mark.asyncio
async def test_a_row_whose_replay_failed_is_not_judged_at_all():
    with patch.object(
        judge_module.LLMGateway, "acreate_structured_output", new_callable=AsyncMock
    ) as mock_llm:
        row = await judge_row(
            _question().text, _replayed(None, error="retriever exploded"), params=_params()
        )

    mock_llm.assert_not_called()
    assert row == JudgedRow(coverage_score=None, answer=None, error="retriever exploded")
    assert row.was_judged is False


@pytest.mark.asyncio
async def test_the_coverage_prompt_never_contains_the_answer():
    fake = _FakeLLM(score=3, answer="A very fluent answer about runbooks.")

    with _patched(fake):
        await judge_row(_question().text, _replayed(), params=_params())

    coverage_prompt = next(call for call in fake.prompts if call["model"] == COVERAGE)
    assert "A very fluent answer about runbooks." not in coverage_prompt["text_input"]
    assert "A very fluent answer about runbooks." not in coverage_prompt["system_prompt"]
    # And the answer call happened strictly after the coverage call.
    assert fake.calls.index(COVERAGE) < fake.calls.index(ANSWER)


@pytest.mark.asyncio
async def test_the_judge_scores_the_whole_context_not_a_storage_bounded_excerpt():
    """``store_context_max_chars`` must not reach the coverage prompt.

    A context whose answer sits past the storage bound would otherwise be scored
    as a gap that memory in fact covered — the score would move whenever an
    operator tuned a column size, and at ``0`` every row would score 0.
    """
    context = "filler. " * 800 + "ESCALATION goes to the on-call SRE."
    fake = _FakeLLM(score=5)

    with _patched(fake):
        row = await judge_row(
            _question().text,
            _replayed(context),
            params=_params(store_context_max_chars=100),
        )

    coverage_prompt = next(call for call in fake.prompts if call["model"] == COVERAGE)
    assert "ESCALATION goes to the on-call SRE." in coverage_prompt["text_input"]
    assert row.coverage_score == 5


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_scored_row_makes_exactly_two_calls_in_order():
    """One judgement, then one generation. There is no second verdict.

    A separate ``answered`` boolean judged from the generated answer would be a
    second, sloppier coverage score that could contradict the first — and the
    first is the number every reported average is a mean of.
    """
    fake = _FakeLLM(score=4, answer="They live in infra-docs.")

    with _patched(fake):
        row = await judge_row(_question().text, _replayed(), params=_params())

    assert fake.calls == [COVERAGE, ANSWER]
    assert row.coverage_score == 4
    assert row.answer == "They live in infra-docs."
    assert row.was_judged is True


@pytest.mark.asyncio
async def test_the_question_reaches_every_prompt():
    fake = _FakeLLM()
    question = "What is our incident escalation path out of hours?"

    with _patched(fake):
        await judge_row(question, _replayed(), params=_params())

    assert all(question in call["text_input"] for call in fake.prompts)


@pytest.mark.asyncio
async def test_an_empty_generated_answer_is_stored_as_null():
    fake = _FakeLLM(score=3, answer="   ")

    with _patched(fake):
        row = await judge_row(_question().text, _replayed(), params=_params())

    assert row.answer is None
    assert row.coverage_score == 3


# --------------------------------------------------------------------------
# Bounds, clamping and retries
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_score_above_the_maximum_is_clamped(monkeypatch):
    """A provider that ignores ``le`` must not push a mean above the maximum."""

    class _Unbounded:
        __name__ = COVERAGE

        def __init__(self, **kwargs):
            self.score = 99
            self.reason = "over the top"

    async def over_the_top(*, text_input, system_prompt, response_model, **kwargs):
        return _Unbounded()

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=over_the_top,
    ):
        score, reason = await score_context_coverage(
            "q",
            "context",
            judge_score_max=10,
            judge_reason_max_chars=500,
            judge_max_retries=0,
        )

    assert score == 10
    assert reason == "over the top"


@pytest.mark.asyncio
async def test_an_over_long_reason_is_truncated_rather_than_costing_the_score():
    """The reason limit is soft: a chatty rationale must not cost a coverage score."""
    fake = _FakeLLM(score=3, reason="r" * 300)

    with _patched(fake):
        row = await judge_row(
            _question().text, _replayed(), params=_params(judge_reason_max_chars=40)
        )

    assert row.coverage_score == 3
    assert row.coverage_reason == "r" * 40


def test_the_reason_length_is_not_a_hard_constraint():
    # A ``max_length`` here would turn a long rationale into a validation error
    # that burns every retry and leaves the row unscored.
    assert coverage_model(10, 10)(score=1, reason="r" * 400).reason == "r" * 400


def test_the_score_bounds_come_from_the_configured_maximum():
    """0-10 by default, and the bound is the parameter rather than a literal."""
    model = coverage_model(10, 120)
    assert model(score=0, reason="ok").score == 0
    assert model(score=10, reason="ok").score == 10
    with pytest.raises(Exception):
        model(score=11, reason="ok")
    with pytest.raises(Exception):
        model(score=-1, reason="ok")

    # Reason before score, so the model justifies first and scores second.
    assert list(model.model_fields) == ["reason", "score"]
    assert list(CoverageAnswer.model_fields) == ["answer"]


def test_the_model_classes_are_cached_per_configured_bound():
    assert coverage_model(10, 500) is coverage_model(10, 500)
    assert coverage_model(10, 500) is not coverage_model(6, 500)


@pytest.mark.asyncio
async def test_a_flaky_judge_call_is_retried_up_to_judge_max_retries():
    attempts = {"count": 0}
    fake = _FakeLLM(score=4)

    def flaky(**kwargs):
        if kwargs["response_model"].__name__ == COVERAGE:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("provider hiccup")
        return fake(**kwargs)

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=flaky,
    ):
        row = await judge_row(_question().text, _replayed(), params=_params(judge_max_retries=2))

    assert attempts["count"] == 3
    assert row.coverage_score == 4


@pytest.mark.asyncio
async def test_a_permanently_failing_coverage_call_leaves_the_scores_null():
    def always_fails(**kwargs):
        raise RuntimeError("provider down")

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=always_fails,
    ):
        row = await judge_row(_question().text, _replayed(), params=_params(judge_max_retries=1))

    assert row.coverage_score is None
    assert row.answer is None
    # Class-prefixed and bounded: this string is persisted and returned by the API.
    assert row.error == "RuntimeError: provider down"


@pytest.mark.asyncio
async def test_a_failing_answer_call_keeps_the_coverage_score_that_already_succeeded():
    fake = _FakeLLM(score=4)

    def fails_after_coverage(**kwargs):
        if kwargs["response_model"].__name__ == COVERAGE:
            return fake(**kwargs)
        raise RuntimeError("completion down")

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=fails_after_coverage,
    ):
        row = await judge_row(_question().text, _replayed(), params=_params(judge_max_retries=0))

    # The coverage score is the number every aggregate is a mean of, so it
    # survives the failure of the generation that follows it.
    assert row.coverage_score == 4
    assert row.answer is None
    assert row.error == "RuntimeError: completion down"


# --------------------------------------------------------------------------
# Batch behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judging_is_index_aligned_and_rejects_a_length_mismatch():
    questions = [_question("a"), _question("b")]

    with pytest.raises(ValueError):
        await judge_rows(questions, [_replayed()], params=_params())


@pytest.mark.asyncio
async def test_judge_rows_scores_each_row_against_its_own_context():
    questions = [_question("a"), _question("b"), _question("c")]
    replayed = [_replayed("context a"), _replayed(None), _replayed("context c")]

    seen: list[str] = []

    def by_context(**kwargs):
        if kwargs["response_model"].__name__ == COVERAGE:
            seen.append(kwargs["text_input"])
            return kwargs["response_model"](score=3, reason="ok")
        return kwargs["response_model"](answer="an answer")

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=by_context,
    ):
        rows = await judge_rows(questions, replayed, params=_params())

    assert [row.coverage_score for row in rows] == [3, 0, 3]
    # The zero-scoring row has no answer, because nothing was retrieved for it.
    assert rows[1].answer is None
    # The empty-context row never reached the LLM.
    assert len(seen) == 2
    assert any("context a" in prompt for prompt in seen)
    assert any("context c" in prompt for prompt in seen)


@pytest.mark.asyncio
async def test_judging_is_bounded_by_judge_max_concurrent():
    in_flight = 0
    peak = 0

    async def slow(*, text_input, system_prompt, response_model, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        if response_model.__name__ == COVERAGE:
            return response_model(score=3, reason="ok")
        return response_model(answer="an answer")

    questions = [_question(f"q{index}") for index in range(10)]
    replayed = [_replayed(f"context {index}") for index in range(10)]

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=slow,
    ):
        rows = await judge_rows(questions, replayed, params=_params(judge_max_concurrent=2))

    assert len(rows) == 10
    # Two rows in flight, each making at most one call at a time.
    assert peak <= 2


@pytest.mark.asyncio
async def test_judging_nothing_calls_nothing():
    with patch.object(
        judge_module.LLMGateway, "acreate_structured_output", new_callable=AsyncMock
    ) as mock_llm:
        assert await judge_rows([], [], params=_params()) == []

    mock_llm.assert_not_called()


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------


def test_every_judge_prompt_file_exists_and_renders():
    """A missing prompt file would otherwise send ``system_prompt=None``."""
    for system_file in (
        judge_module.COVERAGE_SYSTEM_PROMPT,
        judge_module.ANSWER_SYSTEM_PROMPT,
    ):
        assert judge_module._system_prompt(system_file).strip()

    coverage_input = judge_module._input_prompt(
        judge_module.COVERAGE_INPUT_PROMPT,
        {
            "question": "Where are the runbooks?",
            "context": "infra-docs",
            "min_score": 0,
            "max_score": 10,
            "reason_max_chars": 500,
        },
    )
    assert "Where are the runbooks?" in coverage_input
    # The endpoints are rendered from the parameters, so moving judge_score_max
    # cannot leave the prompt describing a scale nothing is scored on.
    assert "0 to 10" in coverage_input

    answer_input = judge_module._input_prompt(
        judge_module.ANSWER_INPUT_PROMPT, {"question": "q", "context": "c"}
    )
    assert "q" in answer_input and "c" in answer_input


def test_only_the_two_surviving_judge_prompts_are_shipped():
    """The ``answered`` verdict is gone, and so are its prompt files."""
    from pathlib import Path

    from cognee.root_dir import get_absolute_path

    directory = Path(get_absolute_path(judge_module.PROMPT_DIRECTORY))
    shipped = {path.name for path in directory.glob("recall_coverage_judge_*.txt")}

    assert shipped == {
        judge_module.COVERAGE_SYSTEM_PROMPT,
        judge_module.COVERAGE_INPUT_PROMPT,
    }


def test_preload_reads_every_system_prompt_before_anything_is_spent():
    """Fail-before-spend: a packaging bug must not surface per row after the replay."""
    judge_module.preload_judge_prompts()

    with patch.object(judge_module, "read_query_prompt", lambda *args, **kwargs: None):
        with pytest.raises(FileNotFoundError):
            judge_module.preload_judge_prompts()


def test_the_coverage_rubric_anchors_zero_on_the_context():
    """0 = "the retrieved context cannot answer this at all", stated in the prompt."""
    system = judge_module._system_prompt(judge_module.COVERAGE_SYSTEM_PROMPT)

    assert "cannot answer this at all" in system
    # And the whole scale is asked for, so scores do not cluster in the middle.
    assert "Use the whole scale" in system


def test_a_missing_system_prompt_fails_loudly(monkeypatch):
    monkeypatch.setattr(judge_module, "read_query_prompt", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError):
        judge_module._system_prompt(judge_module.COVERAGE_SYSTEM_PROMPT)
