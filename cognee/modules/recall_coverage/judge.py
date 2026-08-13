"""Score how well the retrieved context covers each replayed question.

Phase 3 of a recall-coverage run. **One judge call per row**, plus at most one
generation:

1. **``coverage_score``**, an integer ``0..judge_score_max`` (0-10 by default),
   from *(question, retrieved context)* with **the answer absent from the
   prompt**. This is the only judgement, and the number every reported average is
   a mean of.
2. **The answer**, generated from the same context — but only when the coverage
   score is above zero. At zero nothing was retrieved that helps, so there is
   nothing to have answered from, and paying for a completion to confirm that is
   money spent on a foregone conclusion. ``answer`` is NULL there. This is also
   why the replay retrieves with ``only_context=True``.

Two design decisions the obvious shortcuts get wrong:

* **The judge never sees an answer.** Handing a judge a fluent answer and asking
  it to score the *context* is contamination: a confident answer reads as good
  coverage even when the context was thin and the model filled the gap from its
  own weights. So the coverage call runs first, on the context alone, and the
  answer is generated afterwards — a stored artefact for a human to read, not an
  input to any score.
* **Correctness is rejected outright.** No golden answer exists for a question
  scraped out of real traffic, so a correctness judge would be grading its own
  generation. Coverage of the retrieved context is the answerable question, and
  it is the one memory is actually responsible for.

**Empty context scores 0 with no LLM call at all.** No context, no coverage: that
is the definition, not an approximation of one, so it is decided in Python.

A row whose replay failed (``ReplayedRow.error``) is not judged and keeps a NULL
score: "we could not ask" is not evidence about memory.
"""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Awaitable, Callable, Optional, Sequence

from pydantic import BaseModel, Field, create_model

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.recall_coverage.dedup import DedupedQuestion
from cognee.modules.recall_coverage.replay import ReplayedRow, error_text
from cognee.modules.recall_coverage.types import CoverageParams
from cognee.root_dir import get_absolute_path
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")

PROMPT_DIRECTORY = "./modules/recall_coverage/prompts"

COVERAGE_SYSTEM_PROMPT = "recall_coverage_judge_coverage_system.txt"
COVERAGE_INPUT_PROMPT = "recall_coverage_judge_coverage_input.txt"
ANSWER_SYSTEM_PROMPT = "recall_coverage_answer_system.txt"
ANSWER_INPUT_PROMPT = "recall_coverage_answer_input.txt"

# The bottom of the coverage scale. Its top is ``judge_score_max``, a parameter;
# the bottom is 0 by definition — "the retrieved context cannot answer this at
# all" — so it is not a knob.
MIN_JUDGE_SCORE = 0


@dataclass(frozen=True)
class JudgedRow:
    """The judge's verdict on one replayed question row.

    ``coverage_score`` is ``None`` exactly when the row was not judged (its replay
    failed). A judged row always has a score; ``answer`` stays ``None`` when the
    score is 0, because nothing was retrieved to answer from.

    ``coverage_reason`` is kept for logging and threshold tuning. No column stores
    it, deliberately: a stored free-text rationale invites reading the report as a
    review of the answer rather than of the memory.
    """

    coverage_score: Optional[int]
    answer: Optional[str]
    coverage_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def was_judged(self) -> bool:
        return self.coverage_score is not None


class CoverageAnswer(BaseModel):
    """The answer produced from the retrieved context alone.

    No length bound: unlike the topic label and the judge's reason, the answer has
    no configured limit because it is stored whole, not rendered into a chip, and a
    reader who has to judge whether memory could answer must see the answer memory
    gave rather than a truncated one.
    """

    answer: str


@lru_cache
def coverage_model(judge_score_max: int, judge_reason_max_chars: int) -> type[BaseModel]:
    """Structured-output model for the coverage score.

    Built per ``(judge_score_max, judge_reason_max_chars)`` rather than declared
    with literal bounds: both are configuration parameters, and baking either
    number into a class would make it the one magic number in the module. Cached
    because the class identity is what the structured-output framework keys its
    schema on.

    ``reason`` is declared **before** ``score`` so the model states its
    justification first and the score is conditioned on it, rather than the
    reason being a post-hoc defence of a number already emitted. The prompts under
    ``prompts/`` carry the rubric's anchors; this class carries only its bounds.

    The reason length is a **soft** limit — asked for in the field description and
    in the prompt, enforced by truncation in Python, deliberately not a
    ``max_length`` constraint. A chatty rationale is a cosmetic problem, but a
    validation error on it would exhaust the retries and cost the row its
    coverage score, which is the number every aggregate is a mean of. The score
    bound stays a hard constraint: it tells the provider the scale, and an integer
    range is cheap to satisfy.
    """
    return create_model(
        "ContextCoverage",
        reason=(
            str,
            Field(
                description=(
                    "Which parts of what the question needs are present in the context, "
                    f"in at most {judge_reason_max_chars} characters."
                ),
            ),
        ),
        score=(
            int,
            Field(
                ge=MIN_JUDGE_SCORE,
                le=judge_score_max,
                description="How much of what is needed to answer the question the context holds.",
            ),
        ),
    )


def _system_prompt(file_name: str) -> str:
    """Read a system prompt, turning a missing file into a loud failure.

    ``read_query_prompt`` returns ``None`` for a missing file rather than raising,
    which would send ``system_prompt=None`` and have the judge score from no
    rubric at all — a plausible-looking number produced by nothing.
    """
    prompt = read_query_prompt(file_name, base_directory=get_absolute_path(PROMPT_DIRECTORY))
    if not prompt:
        raise FileNotFoundError(
            f"recall-coverage judge prompt {file_name} is missing from "
            f"{get_absolute_path(PROMPT_DIRECTORY)}"
        )
    return prompt


def _input_prompt(file_name: str, context: dict) -> str:
    return render_prompt(file_name, context, base_directory=get_absolute_path(PROMPT_DIRECTORY))


def preload_judge_prompts() -> None:
    """Read every judge system prompt once, raising if any is missing.

    Called by the pipeline before the replay phase, for the same reason the
    fingerprint check runs before it: a missing prompt file is a packaging
    problem that would otherwise surface as one identical per-row error on every
    row of a "complete" run — after the full replay cost was already paid.
    """
    for file_name in (COVERAGE_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT):
        _system_prompt(file_name)


async def _with_retries(
    call: Callable[[], Awaitable[object]], *, judge_max_retries: int, what: str
) -> object:
    """Retry a judge call, then give up and let the caller record the error.

    ``judge_max_retries`` counts retries *after* the first attempt, so the default
    of 2 makes three attempts. Structured-output failures here are usually a
    provider hiccup or a schema violation, both of which a retry fixes; a row that
    still fails is recorded as an error rather than being scored by a fallback,
    because an invented score is worse than a missing one.
    """
    attempts = max(1, judge_max_retries + 1)
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            return await call()
        except Exception as error:
            last_error = error
            logger.warning(
                "recall_coverage: %s failed on attempt %s of %s: %s",
                what,
                attempt + 1,
                attempts,
                error,
            )

    raise last_error if last_error else RuntimeError(f"recall_coverage: {what} made no attempt")


async def score_context_coverage(
    question: str,
    retrieval_context: str,
    *,
    judge_score_max: int,
    judge_reason_max_chars: int,
    judge_max_retries: int,
) -> tuple[int, str]:
    """The judge call: score the context against the question. No answer in sight.

    The returned score is clamped into ``0..judge_score_max``: ``le``/``ge`` on the
    response model is a request to the provider, and a provider that ignores it
    must not be able to push a topic average above the top of the scale every
    number in the report is read against.
    """
    response_model = coverage_model(judge_score_max, judge_reason_max_chars)
    text_input = _input_prompt(
        COVERAGE_INPUT_PROMPT,
        {
            "question": question,
            "context": retrieval_context,
            "min_score": MIN_JUDGE_SCORE,
            "max_score": judge_score_max,
            "reason_max_chars": judge_reason_max_chars,
        },
    )
    system_prompt = _system_prompt(COVERAGE_SYSTEM_PROMPT)

    response = await _with_retries(
        lambda: LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=response_model,
        ),
        judge_max_retries=judge_max_retries,
        what="context coverage scoring",
    )

    score = int(getattr(response, "score", MIN_JUDGE_SCORE) or MIN_JUDGE_SCORE)
    reason = str(getattr(response, "reason", "") or "").strip()
    return max(MIN_JUDGE_SCORE, min(score, judge_score_max)), reason[:judge_reason_max_chars]


async def generate_answer(question: str, retrieval_context: str, *, judge_max_retries: int) -> str:
    """One LLM call answering the question from the retrieved context only.

    A generation, not a judgement, and nothing is scored from it: it exists so a
    reader of the report can see what memory could actually say. The prompt forbids
    outside knowledge, because an answer the model knew anyway would describe the
    model rather than the memory.
    """
    text_input = _input_prompt(
        ANSWER_INPUT_PROMPT, {"question": question, "context": retrieval_context}
    )
    system_prompt = _system_prompt(ANSWER_SYSTEM_PROMPT)

    response = await _with_retries(
        lambda: LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=CoverageAnswer,
        ),
        judge_max_retries=judge_max_retries,
        what="answer generation",
    )

    return str(getattr(response, "answer", "") or "").strip()


async def judge_row(question: str, replayed: ReplayedRow, *, params: CoverageParams) -> JudgedRow:
    """Judge one replayed row: coverage, then — only above zero — the answer.

    Three short-circuits, in order:

    * a replay error leaves the score NULL and the row is not judged at all;
    * an empty context scores 0 with **no LLM call** — no context, no coverage;
    * a score of 0 stops there: ``answer`` stays NULL, with no completion call.
    """
    if replayed.error:
        return JudgedRow(coverage_score=None, answer=None, error=replayed.error)

    if not replayed.has_context:
        return JudgedRow(
            coverage_score=MIN_JUDGE_SCORE,
            answer=None,
            coverage_reason="Nothing was retrieved for this question.",
        )

    context = replayed.retrieval_context or ""

    try:
        score, coverage_reason = await score_context_coverage(
            question,
            context,
            judge_score_max=params.judge_score_max,
            judge_reason_max_chars=params.judge_reason_max_chars,
            judge_max_retries=params.judge_max_retries,
        )
    except Exception as error:
        logger.warning("recall_coverage: coverage scoring gave up on a row: %s", error)
        return JudgedRow(coverage_score=None, answer=None, error=error_text(error))

    if score <= MIN_JUDGE_SCORE:
        return JudgedRow(
            coverage_score=MIN_JUDGE_SCORE, answer=None, coverage_reason=coverage_reason
        )

    try:
        answer = await generate_answer(
            question, context, judge_max_retries=params.judge_max_retries
        )
    except Exception as error:
        # The coverage score survived, and it is the number every average is a
        # mean of, so it is kept rather than thrown away with the answer.
        logger.warning("recall_coverage: answer generation gave up on a scored row: %s", error)
        return JudgedRow(
            coverage_score=score,
            answer=None,
            coverage_reason=coverage_reason,
            error=error_text(error),
        )

    return JudgedRow(coverage_score=score, answer=answer or None, coverage_reason=coverage_reason)


async def judge_rows(
    questions: Sequence[DedupedQuestion],
    replayed: Sequence[ReplayedRow],
    *,
    params: CoverageParams,
) -> list[JudgedRow]:
    """Judge every replayed row concurrently, index-aligned with ``questions``.

    Bounded by ``asyncio.Semaphore(judge_max_concurrent)`` and run concurrently,
    like ``BeamEvalAdapter`` rather than ``DirectLLMEvalAdapter``'s sequential
    loop: hundreds of rows at two sequential calls each is an unacceptable wall
    clock for a report somebody is waiting on.

    Index alignment between ``questions`` and ``replayed`` is the whole contract —
    a shifted row does not fail, it scores one question's context against another
    question — so a length mismatch raises.
    """
    if not questions:
        return []

    if len(questions) != len(replayed):
        raise ValueError(
            f"recall-coverage judging got {len(replayed)} replayed rows for "
            f"{len(questions)} questions; the two must be index-aligned."
        )

    semaphore = asyncio.Semaphore(max(1, params.judge_max_concurrent))

    async def _judge_one(question: DedupedQuestion, row: ReplayedRow) -> JudgedRow:
        async with semaphore:
            return await judge_row(question.text, row, params=params)

    return list(
        await asyncio.gather(
            *(_judge_one(question, row) for question, row in zip(questions, replayed))
        )
    )


__all__ = [
    "MIN_JUDGE_SCORE",
    "CoverageAnswer",
    "JudgedRow",
    "coverage_model",
    "generate_answer",
    "judge_row",
    "judge_rows",
    "preload_judge_prompts",
    "score_context_coverage",
]
