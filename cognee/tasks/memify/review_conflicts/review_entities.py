"""Send the subjects' sections to the LLM in calls, one at a time, and yield the conflicts to write.

``select_review_facts`` decides which facts each section holds, ``build_review_prompt``
writes a call's prompt, and ``accept_review`` decides what of a response is kept. This
module decides which sections share a call, what a section too large for one call
leaves out, and how a failed call is retried.
"""

import json
from collections.abc import AsyncIterator, Iterator

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.utils import get_llm_token_ceiling, get_llm_tokenizer
from cognee.shared.logging_utils import get_logger

from .accept_review import accept_review_response
from .build_review_prompt import ReviewPrompt, build_review_prompt
from .models import ConflictsToFinalize, ConflictsToWrite, ReviewScope
from .schema import ReviewResponse
from .select_review_facts import (
    FactComparisons,
    find_fact_comparisons,
    select_lists_to_leave_out,
    select_section_facts,
    select_subject_ids,
)

# One call's input, prompt and response schema included, and at most half the LLM
# token ceiling, as for cognify's chunks. Sections are packed into it whole; a section
# too large for it first loses its list-like names, see _fit_section.
REVIEW_INPUT_TOKENS = 8_000
SUBJECTS_PER_CALL = 25
MAX_RETRY_SPLIT_DEPTH = 1

PROMPT_FILE = "review_conflicts.txt"

logger = get_logger("review_conflicts")


def _compute_user_prompt_token_budget(system_prompt: str, tokenizer) -> int:
    """What a call may spend on sections once it has paid the prompt and the schema."""
    overhead = tokenizer.count_tokens(system_prompt) + tokenizer.count_tokens(
        json.dumps(ReviewResponse.model_json_schema(), sort_keys=True)
    )
    return min(REVIEW_INPUT_TOKENS, get_llm_token_ceiling() // 2) - overhead


# --- which sections share a call ---------------------------------------------


def _fit_section(
    scope: ReviewScope,
    comparisons: FactComparisons,
    subject: str,
    tokenizer,
    user_prompt_token_budget: int,
) -> tuple[set[str], int] | None:
    """The subject's section and its size alone in a call, less list-like names until it fits.

    The names ``select_lists_to_leave_out`` allows go one at a time, in its order. A
    section that cannot fit, such as one whose header alone is too large, is not
    reviewed, and not retried: a retry would only skip it again. None says so.
    """
    section = select_section_facts(scope, comparisons, subject)
    lists = select_lists_to_leave_out(scope, subject, section)
    left_out: list[str] = []
    while True:
        prompt = build_review_prompt(scope, comparisons, {subject: section})
        tokens = tokenizer.count_tokens(prompt.text)
        if tokens <= user_prompt_token_budget or not lists:
            break
        name, fact_ids = lists.pop(0)
        left_out.append(name)
        section = section - fact_ids
    if tokens > user_prompt_token_budget or (left_out and not section):
        logger.warning("Conflict review skipped entity %s: too large for one call", subject)
        return None
    if left_out:
        logger.warning(
            "Conflict review left out %s of entity %s to fit one call", left_out, subject
        )
    return section, tokens


def _plan_calls(
    scope: ReviewScope,
    comparisons: FactComparisons,
    subject_ids: list[str],
    tokenizer,
    user_prompt_token_budget: int,
    skipped_entity_ids: list[str],
) -> Iterator[dict[str, set[str]]]:
    """Pack sections, in order, into calls of at most SUBJECTS_PER_CALL subjects.

    The sizes of a call's sections, each measured alone in a call, add up to at most the
    user prompt token budget. The prompt sent is not measured. It names the values and
    documents its sections share only once, so it is usually smaller than that sum, and
    its labels are longer (n12 rather than n1), so it can be larger.
    """
    sections: dict[str, set[str]] = {}
    used = 0
    for subject in subject_ids:
        fitted = _fit_section(scope, comparisons, subject, tokenizer, user_prompt_token_budget)
        if fitted is None:
            skipped_entity_ids.append(subject)
            continue
        section, tokens = fitted
        if sections and (
            used + tokens > user_prompt_token_budget or len(sections) == SUBJECTS_PER_CALL
        ):
            yield sections
            sections, used = {}, 0
            # The section was selected before the writer stored the call just yielded.
            fitted = _fit_section(scope, comparisons, subject, tokenizer, user_prompt_token_budget)
            if fitted is None:
                skipped_entity_ids.append(subject)
                continue
            section, tokens = fitted
        sections[subject] = section
        used += tokens
    if sections:
        yield sections


# --- one call and its retry --------------------------------------------------


async def _ask_llm(prompt: ReviewPrompt, system_prompt: str) -> ReviewResponse:
    """Send one prompt; the response comes back parsed, not yet accepted."""
    response = await LLMGateway.acreate_structured_output(
        text_input=prompt.text,
        system_prompt=system_prompt,
        response_model=ReviewResponse,
    )
    return ReviewResponse.model_validate(response)


async def _review(
    scope: ReviewScope,
    comparisons: FactComparisons,
    sections: dict[str, set[str]],
    system_prompt: str,
    splits_left: int,
    failed_entity_ids: list[str],
) -> AsyncIterator[ConflictsToWrite]:
    """One LLM call, or — when it fails — two calls over half the subjects each.

    A call fails when the LLM raises or its response is rejected, and both are retried.
    Building the prompt and writing the conflicts stay outside the retry. The halves keep the
    facts selected for the call they split, so the second half's facts were selected
    before the first half's conflicts were written: selecting them again could make a half
    too large for one call.
    """
    prompt = build_review_prompt(scope, comparisons, sections)
    try:
        response = await _ask_llm(prompt, system_prompt)
        to_write = accept_review_response(response, scope, prompt)
    except LLMPaymentRequiredError:
        raise
    except Exception as error:
        if splits_left and len(sections) > 1:
            # A call with half the subjects may still succeed where the whole call did not.
            items = list(sections.items())
            midpoint = len(items) // 2
            for half in (dict(items[:midpoint]), dict(items[midpoint:])):
                async for to_write in _review(
                    scope, comparisons, half, system_prompt, splits_left - 1, failed_entity_ids
                ):
                    yield to_write
        else:
            failed_entity_ids.extend(sections)
            logger.warning(
                "Conflict review failed for entities %s: %s", list(sections), error, exc_info=True
            )
        return
    failed_entity_ids.extend(to_write.unreviewed_entity_ids)
    yield to_write


async def review_entities(
    scope: ReviewScope, failed_entity_ids: list[str], skipped_entity_ids: list[str]
) -> AsyncIterator[ConflictsToWrite | ConflictsToFinalize]:
    """Review the scope's subjects in LLM calls, and yield each call's conflicts to the writer.

    The conflicts the reader found to drop go first, and ``ConflictsToFinalize`` last. The calls
    run one at a time on purpose: the writer applies each call's conflicts to this ``ReviewScope``
    before the next call's facts are selected and its prompt is built, so that call
    sees the marks, citations and conflicts the last write left, as
    ``test_next_call_selects_facts_after_the_last_write`` checks. Yielding also keeps a
    failed write in the writer, outside the retry around each LLM call.

    The subjects the run failed to review are added to ``failed_entity_ids``, and
    those too large for one call to ``skipped_entity_ids``: the caller's lists, because
    the pipeline runner returns no task output to its caller.
    """
    system_prompt = render_prompt(PROMPT_FILE, {})
    tokenizer = get_llm_tokenizer()
    user_prompt_token_budget = _compute_user_prompt_token_budget(system_prompt, tokenizer)
    comparisons = find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    subject_ids = select_subject_ids(scope, comparisons)
    to_finalize = ConflictsToFinalize(dropped_conflict_ids=list(scope.conflict_ids_to_drop))
    if scope.conflict_ids_to_drop:
        yield ConflictsToWrite(scope, dropped_conflict_ids=list(scope.conflict_ids_to_drop))
    calls = _plan_calls(
        scope, comparisons, subject_ids, tokenizer, user_prompt_token_budget, skipped_entity_ids
    )
    for sections in calls:
        async for to_write in _review(
            scope,
            comparisons,
            sections,
            system_prompt,
            MAX_RETRY_SPLIT_DEPTH,
            failed_entity_ids,
        ):
            yield to_write
            to_finalize.accepted_conflicts.extend(to_write.accepted_conflicts)
            to_finalize.dropped_conflict_ids.extend(to_write.dropped_conflict_ids)
    yield to_finalize
