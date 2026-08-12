"""Guards on the collapse/dedup half of recall-coverage phase 1.

The invariants these protect are the ones that decide what the reported numbers
*mean*:

* the partition key is ``(user_id, dataset_id)``, so a fanned search stays one
  ask per dataset and two teammates asking the same thing stay two rows;
* the retry cooldown — not the fan-out rule — is the counting rule, and what it
  swallowed is reported rather than lost;
* ``occurrence_count`` counts distinct asks, so the same question on three
  separate days is demand while eight retries in five minutes is not;
* the cost of dedup is bounded by ``max_questions``, not by how much history the
  tenant has.

Vectors are supplied explicitly (or from a hand-written fake engine) so every
similarity in here is exact and the threshold assertions cannot drift.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest

from cognee.modules.recall_coverage.dedup import (
    Ask,
    assign_question_groups,
    collapse_asks,
    collapse_text_key,
    dedup_asks,
    group_by_similarity,
)
from cognee.modules.recall_coverage.embedding import embed_normalized, normalize_rows
from cognee.modules.recall_coverage.types import QuestionSource
from cognee.modules.search.operations.get_queries import QueryWindowRow

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

RUNBOOKS = "Where are the runbooks?"
RUNBOOKS_REPHRASED = "Where do I find the runbooks?"
CREDENTIALS = "How do I rotate credentials?"

# Explicit unit-ish vectors: RUNBOOKS and RUNBOOKS_REPHRASED sit at cosine
# ~0.995 (a near-duplicate), CREDENTIALS is orthogonal to both.
VECTORS: dict[str, list[float]] = {
    RUNBOOKS: [1.0, 0.0, 0.0],
    RUNBOOKS_REPHRASED: [10.0, 1.0, 0.0],
    CREDENTIALS: [0.0, 0.0, 1.0],
}


class _FakeEngine:
    """Deterministic embeddings for the texts used in this module."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.batches: list[list[str]] = []

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_vector_size(self) -> int:
        return 3

    async def embed_text(self, texts):
        self.batches.append(list(texts))
        return [list(VECTORS[text]) for text in texts]


def _row(
    text: str,
    *,
    user_id,
    dataset_id,
    created_at: datetime,
    query_type: str = "GRAPH_COMPLETION",
) -> QueryWindowRow:
    return QueryWindowRow(
        query_id=uuid4(),
        text=text,
        query_type=query_type,
        user_id=user_id,
        dataset_id=dataset_id,
        created_at=created_at,
    )


def _collapse(rows, *, fanout=5, cooldown=300, max_questions=150):
    # Rows are handed over newest-first, exactly as ``get_queries`` returns them.
    return collapse_asks(
        sorted(rows, key=lambda row: row.created_at, reverse=True),
        fanout_window_seconds=fanout,
        retry_cooldown_seconds=cooldown,
        max_questions=max_questions,
    )


def _matrix(asks) -> np.ndarray:
    return normalize_rows([VECTORS[ask.text] for ask in asks])


def test_collapse_text_key_ignores_case_and_whitespace():
    assert collapse_text_key("  Where  are the RUNBOOKS? ") == collapse_text_key(RUNBOOKS)


def test_one_fanned_ask_stays_three_rows_in_three_partitions():
    """One search over three datasets is three asks, not one.

    ``log_search_history`` writes one ``queries`` row per payload, i.e. per
    dataset. Under ``(user_id, dataset_id)`` partitioning those three rows are
    already *correctly* three question rows: each dataset was asked once and each
    answers independently. The fan-out rule only marks them as one search so the
    string is embedded once — it must never collapse them.
    """
    user_id = uuid4()
    datasets = [uuid4(), uuid4(), uuid4()]
    rows = [
        _row(RUNBOOKS, user_id=user_id, dataset_id=dataset_id, created_at=BASE_TIME)
        for dataset_id in datasets
    ]

    collapsed = _collapse(rows)

    assert len(collapsed.asks) == 3
    assert collapsed.collapsed_retry_count == 0
    assert collapsed.distinct_ask_count == 3
    # Recognised as one fanned search, purely as provenance.
    assert collapsed.fanout_group_count == 1
    assert len({ask.fanout_group_id for ask in collapsed.asks}) == 1

    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert result.partition_count == 3
    assert len(result.questions) == 3
    assert {question.dataset_id for question in result.questions} == set(datasets)
    assert [question.occurrence_count for question in result.questions] == [1, 1, 1]


def test_same_text_on_three_days_in_one_partition_counts_three_occurrences():
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(days=day),
        )
        for day in range(3)
    ]

    collapsed = _collapse(rows)

    # Three separate days is genuine demand, so the cooldown must not touch it.
    assert len(collapsed.asks) == 3
    assert collapsed.collapsed_retry_count == 0

    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert len(result.questions) == 1
    question = result.questions[0]
    assert question.occurrence_count == 3
    assert question.first_asked_at == BASE_TIME - timedelta(days=2)
    assert question.last_asked_at == BASE_TIME
    assert question.was_asked is True
    assert question.source == QuestionSource.OBSERVED.value
    assert len(question.query_ids) == 3


def test_eight_identical_asks_inside_the_cooldown_collapse_to_one():
    """An agent looping is not eight times the demand — and the loop is reported."""
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(seconds=30 * step),
        )
        for step in range(8)
    ]

    collapsed = _collapse(rows, cooldown=300)

    assert len(collapsed.asks) == 1
    assert collapsed.distinct_ask_count == 1
    assert collapsed.collapsed_retry_count == 7
    assert collapsed.recall_row_count == 8
    assert collapsed.asks[0].retry_collapsed_count == 7
    assert len(collapsed.asks[0].query_ids) == 8
    # The ask spans the whole loop.
    assert collapsed.asks[0].first_seen == BASE_TIME - timedelta(seconds=210)
    assert collapsed.asks[0].last_seen == BASE_TIME

    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert [question.occurrence_count for question in result.questions] == [1]


def test_asks_outside_the_cooldown_stay_separate():
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(RUNBOOKS, user_id=user_id, dataset_id=dataset_id, created_at=BASE_TIME),
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(seconds=301),
        ),
    ]

    collapsed = _collapse(rows, cooldown=300)

    assert len(collapsed.asks) == 2
    assert collapsed.collapsed_retry_count == 0


def test_rows_never_merge_across_different_users():
    dataset_id = uuid4()
    anna, ben = uuid4(), uuid4()
    rows = [
        _row(RUNBOOKS, user_id=anna, dataset_id=dataset_id, created_at=BASE_TIME),
        _row(RUNBOOKS, user_id=ben, dataset_id=dataset_id, created_at=BASE_TIME),
    ]

    collapsed = _collapse(rows)
    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert collapsed.collapsed_retry_count == 0
    assert result.partition_count == 2
    assert {question.user_id for question in result.questions} == {anna, ben}
    assert [question.occurrence_count for question in result.questions] == [1, 1]


def test_rows_never_merge_across_different_datasets():
    user_id = uuid4()
    first, second = uuid4(), uuid4()
    rows = [
        _row(RUNBOOKS, user_id=user_id, dataset_id=first, created_at=BASE_TIME),
        # Far outside the fan-out window, so this is not one fanned search either.
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=second,
            created_at=BASE_TIME - timedelta(hours=4),
        ),
    ]

    collapsed = _collapse(rows)
    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert collapsed.fanout_group_count == 2
    assert result.partition_count == 2
    assert {question.dataset_id for question in result.questions} == {first, second}


def test_rows_with_no_dataset_form_their_own_partition():
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(RUNBOOKS, user_id=user_id, dataset_id=dataset_id, created_at=BASE_TIME),
        _row(RUNBOOKS, user_id=user_id, dataset_id=None, created_at=BASE_TIME),
    ]

    collapsed = _collapse(rows)
    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert result.partition_count == 2
    assert None in {question.dataset_id for question in result.questions}


def test_near_duplicate_text_merges_inside_a_partition():
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(RUNBOOKS, user_id=user_id, dataset_id=dataset_id, created_at=BASE_TIME),
        _row(
            RUNBOOKS_REPHRASED,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(hours=2),
        ),
        _row(
            CREDENTIALS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(hours=3),
        ),
    ]

    collapsed = _collapse(rows)
    matrix = _matrix(collapsed.asks)
    result = dedup_asks(collapsed.asks, matrix, dedup_threshold=0.92)

    assert len(result.questions) == 2
    merged = next(question for question in result.questions if question.occurrence_count == 2)
    # Canonical text is the earliest-asked member of the cluster.
    assert merged.text == RUNBOOKS_REPHRASED
    assert merged.first_asked_at == BASE_TIME - timedelta(hours=2)
    assert merged.last_asked_at == BASE_TIME

    # Above the near-duplicate similarity nothing merges any more.
    strict = dedup_asks(collapsed.asks, matrix, dedup_threshold=0.999)
    assert len(strict.questions) == 3


def test_curated_member_supplies_the_canonical_text_and_counts_no_ask():
    user_id, dataset_id = uuid4(), uuid4()
    curated_id = uuid4()
    asks = [
        Ask(
            text=RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            first_seen=BASE_TIME,
            last_seen=BASE_TIME,
            query_ids=[uuid4()],
        ),
        Ask(
            text=RUNBOOKS_REPHRASED,
            user_id=user_id,
            dataset_id=dataset_id,
            source=QuestionSource.CURATED.value,
            curated_question_id=curated_id,
        ),
    ]

    result = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92)

    assert len(result.questions) == 1
    question = result.questions[0]
    # A human wrote it, so the curated wording is the better label.
    assert question.text == RUNBOOKS_REPHRASED
    assert question.source == QuestionSource.CURATED.value
    assert question.curated_question_id == curated_id
    assert question.was_asked is True
    # The curated member is not itself an ask.
    assert question.occurrence_count == 1


def test_a_purely_curated_cluster_has_no_asks_and_no_timestamps():
    asks = [
        Ask(
            text=CREDENTIALS,
            user_id=uuid4(),
            dataset_id=uuid4(),
            source=QuestionSource.CURATED.value,
            curated_question_id=uuid4(),
        )
    ]

    question = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92).questions[0]

    assert question.was_asked is False
    assert question.occurrence_count == 0
    assert question.first_asked_at is None
    assert question.last_asked_at is None


def test_question_group_id_is_shared_across_partitions_for_matching_text():
    """The UI collapses Anna's row and Ben's row on an id, not on exact text."""
    anna, ben = uuid4(), uuid4()
    dataset_id = uuid4()
    asks = [
        Ask(
            text=RUNBOOKS,
            user_id=anna,
            dataset_id=dataset_id,
            first_seen=BASE_TIME,
            last_seen=BASE_TIME,
        ),
        # Different partition, and deliberately different canonical text: exact
        # string matching in the UI would miss this pair.
        Ask(
            text=RUNBOOKS_REPHRASED,
            user_id=ben,
            dataset_id=dataset_id,
            first_seen=BASE_TIME,
            last_seen=BASE_TIME,
        ),
        Ask(
            text=CREDENTIALS,
            user_id=ben,
            dataset_id=dataset_id,
            first_seen=BASE_TIME,
            last_seen=BASE_TIME,
        ),
    ]
    matrix = _matrix(asks)
    result = dedup_asks(asks, matrix, dedup_threshold=0.92)

    assert result.partition_count == 2
    assert len(result.questions) == 3

    group_count = assign_question_groups(result.questions, matrix, dedup_threshold=0.92)

    by_text = {question.text: question for question in result.questions}
    assert group_count == 2
    assert by_text[RUNBOOKS].question_group_id == by_text[RUNBOOKS_REPHRASED].question_group_id
    assert by_text[CREDENTIALS].question_group_id != by_text[RUNBOOKS].question_group_id
    # Every row carries one, its own when nothing matched.
    assert all(question.question_group_id is not None for question in result.questions)
    # Grouping never merges the rows themselves — every aggregate is a mean over
    # rows, so a shared group id must not change any count.
    assert [question.occurrence_count for question in result.questions] == [1, 1, 1]


@pytest.mark.parametrize("history_size", [50, 500, 5000])
def test_comparison_count_is_bounded_by_max_questions(history_size):
    """Cost depends on ``max_questions``, not on how much history exists.

    There is no ANN index and no LSH in this repo; the truncation *is* the bound.
    """
    max_questions = 10
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(
            f"question number {index}",
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(hours=index),
        )
        for index in range(history_size)
    ]

    collapsed = _collapse(rows, max_questions=max_questions)

    assert collapsed.recall_row_count == history_size
    assert collapsed.distinct_ask_count == history_size
    assert len(collapsed.asks) == max_questions
    assert collapsed.dropped_ask_count == history_size - max_questions
    # Newest first: truncation drops the oldest asks.
    assert collapsed.asks[0].text == "question number 0"

    generator = np.random.default_rng(seed=history_size)
    matrix = normalize_rows(generator.normal(size=(max_questions, 8)).tolist())
    result = dedup_asks(collapsed.asks, matrix, dedup_threshold=0.92)

    # Identical for every history size in the parametrisation: the bound is
    # ``max_questions``, and one partition of 10 is 45 pairs however many rows
    # the tenant has accumulated.
    assert result.comparison_count == max_questions * (max_questions - 1) // 2


def test_dedup_rejects_a_misaligned_matrix():
    asks = [Ask(text=RUNBOOKS, user_id=uuid4(), dataset_id=uuid4())]

    with pytest.raises(ValueError):
        dedup_asks(asks, normalize_rows([[1.0, 0.0], [0.0, 1.0]]), dedup_threshold=0.92)


def test_group_by_similarity_is_single_link():
    # A-B and B-C are above the threshold while A-C is not; single-link makes
    # them one group, which is what the greedy grouping in the spec means.
    vectors = normalize_rows([[1.0, 0.0], [0.92, 0.39], [0.7, 0.71]])
    groups, comparisons = group_by_similarity(vectors, 0.92)

    assert groups == [[0, 1, 2]]
    assert comparisons == 3


def test_empty_window_collapses_and_dedups_to_nothing():
    collapsed = _collapse([])
    result = dedup_asks(collapsed.asks, normalize_rows([]), dedup_threshold=0.92)

    assert collapsed.recall_row_count == 0
    assert collapsed.distinct_ask_count == 0
    assert result.questions == []
    assert result.comparison_count == 0
    assert assign_question_groups(result.questions, normalize_rows([]), dedup_threshold=0.92) == 0


@pytest.mark.asyncio
async def test_fanned_ask_embeds_its_string_once():
    """The only thing the fan-out rule buys: N identical strings, one embedding."""
    user_id = uuid4()
    rows = [
        _row(RUNBOOKS, user_id=user_id, dataset_id=uuid4(), created_at=BASE_TIME) for _ in range(3)
    ]
    collapsed = _collapse(rows)
    engine = _FakeEngine()

    matrix = await embed_normalized(engine, [ask.text for ask in collapsed.asks])

    assert engine.batches == [[RUNBOOKS]]
    assert matrix.shape == (3, 3)

    result = dedup_asks(collapsed.asks, matrix, dedup_threshold=0.92)
    assert len(result.questions) == 3
