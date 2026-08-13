"""Guards on the collapse/dedup half of recall-coverage phase 1.

The invariants these protect are the ones that decide what the reported numbers
*mean*:

* the partition key is ``(user_id, dataset_id)``, so a fanned search stays one
  ask per dataset and two teammates asking the same thing stay two rows;
* the retry cooldown — not the fan-out rule — is the counting rule, and what it
  swallowed is reported rather than lost;
* ``relevance`` counts distinct asks, so the same question on three separate days
  is demand while eight retries in five minutes is not — and it doubles as the
  "was this asked at all" test, which is why there is no ``was_asked`` flag to
  disagree with it;
* the cost of dedup is bounded by ``max_questions``, not by how much history the
  tenant has;
* a row's ``session_id`` comes from the **earliest observed** member of its
  cluster, never from the canonical one, because the canonical member is the
  user-defined text and carries no session.

Vectors are supplied explicitly (or from a hand-written fake engine) so every
similarity in here is exact and the threshold assertions cannot drift.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import numpy as np
import pytest

from cognee.modules.recall_coverage.dedup import (
    Ask,
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
    session_id: Optional[str] = None,
) -> QueryWindowRow:
    return QueryWindowRow(
        query_id=uuid4(),
        text=text,
        query_type=query_type,
        user_id=user_id,
        dataset_id=dataset_id,
        created_at=created_at,
        session_id=session_id,
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
    assert [question.relevance for question in result.questions] == [1, 1, 1]


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
    assert question.relevance == 3
    assert question.first_asked_at == BASE_TIME - timedelta(days=2)
    assert question.last_asked_at == BASE_TIME
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
    assert collapsed.recall_count == 8
    assert collapsed.asks[0].retry_collapsed_count == 7
    assert len(collapsed.asks[0].query_ids) == 8
    # The ask spans the whole loop.
    assert collapsed.asks[0].first_seen == BASE_TIME - timedelta(seconds=210)
    assert collapsed.asks[0].last_seen == BASE_TIME

    result = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92)

    assert [question.relevance for question in result.questions] == [1]


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
    assert [question.relevance for question in result.questions] == [1, 1]


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
    merged = next(question for question in result.questions if question.relevance == 2)
    # Canonical text is the earliest-asked member of the cluster.
    assert merged.text == RUNBOOKS_REPHRASED
    assert merged.first_asked_at == BASE_TIME - timedelta(hours=2)
    assert merged.last_asked_at == BASE_TIME

    # Above the near-duplicate similarity nothing merges any more.
    strict = dedup_asks(collapsed.asks, matrix, dedup_threshold=0.999)
    assert len(strict.questions) == 3


def test_user_defined_member_supplies_the_canonical_text_and_counts_no_ask():
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
            source=QuestionSource.USER_DEFINED.value,
            curated_question_id=curated_id,
        ),
    ]

    result = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92)

    assert len(result.questions) == 1
    question = result.questions[0]
    # A human wrote it, so the user-defined wording is the better label.
    assert question.text == RUNBOOKS_REPHRASED
    assert question.source == QuestionSource.USER_DEFINED.value
    assert question.curated_question_id == curated_id
    # The user-defined member is not itself an ask, so relevance counts only the
    # observed one — and being non-zero is exactly "somebody asked this".
    assert question.relevance == 1


def test_a_purely_user_defined_cluster_has_no_asks_and_no_timestamps():
    asks = [
        Ask(
            text=CREDENTIALS,
            user_id=uuid4(),
            dataset_id=uuid4(),
            source=QuestionSource.USER_DEFINED.value,
            curated_question_id=uuid4(),
        )
    ]

    question = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92).questions[0]

    # relevance 0 *is* "nobody asked this", which is what keeps the row out of
    # every average without a second flag that could disagree.
    assert question.relevance == 0
    assert question.first_asked_at is None
    assert question.last_asked_at is None
    # No session, so the row reports no agent.
    assert question.session_id is None


# --- session ids, i.e. per-row agent attribution -----------------------------


def test_the_session_id_survives_the_collapse_onto_the_ask():
    """The first link of the attribution chain: ``queries`` row -> ask."""
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME,
            session_id="codex_a1",
        )
    ]

    collapsed = _collapse(rows)

    assert [ask.session_id for ask in collapsed.asks] == ["codex_a1"]


def test_a_retry_loop_keeps_one_session_on_the_surviving_ask():
    """The cooldown key deliberately excludes the session, so one ask keeps one.

    Adding the session to the key would change what ``relevance`` counts in order
    to fix a label — attribution is a filter over a flat table, not a partition of
    it. The walk is newest-first, so the newest row is the one that opened the ask.
    """
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(seconds=30 * step),
            session_id="claude_a1",
        )
        for step in range(8)
    ]

    collapsed = _collapse(rows, cooldown=300)

    assert len(collapsed.asks) == 1
    assert collapsed.asks[0].session_id == "claude_a1"


def test_two_agents_inside_the_cooldown_are_one_ask_attributed_to_the_newest():
    """One ask, one session: the row that opened it wins, and nothing is counted twice."""
    user_id, dataset_id = uuid4(), uuid4()
    rows = [
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME,
            session_id="codex_a1",
        ),
        _row(
            RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            created_at=BASE_TIME - timedelta(seconds=10),
            session_id="claude_a1",
        ),
    ]

    collapsed = _collapse(rows, cooldown=300)

    assert len(collapsed.asks) == 1
    assert collapsed.asks[0].session_id == "codex_a1"
    assert collapsed.asks[0].retry_collapsed_count == 1


def test_a_cluster_takes_its_session_from_the_earliest_observed_member():
    """Not from the canonical member — that one is the user-defined text.

    ``_canonical_member`` prefers the user-defined wording, which carries no
    session at all. Keying attribution off it would report ``agent: null`` for a
    cluster real agents asked, which is the whole trap this guards.
    """
    user_id, dataset_id = uuid4(), uuid4()
    asks = [
        Ask(
            text=RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            first_seen=BASE_TIME,
            last_seen=BASE_TIME,
            query_ids=[uuid4()],
            session_id="claude_a1",
        ),
        Ask(
            text=RUNBOOKS_REPHRASED,
            user_id=user_id,
            dataset_id=dataset_id,
            source=QuestionSource.USER_DEFINED.value,
            curated_question_id=uuid4(),
        ),
    ]

    question = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92).questions[0]

    assert question.text == RUNBOOKS_REPHRASED
    assert question.session_id == "claude_a1"


def test_a_cluster_spanning_two_agents_is_attributed_to_the_first_asker():
    """One ``agent`` column means a tie-break, and it is the same one ``first_asked_at`` uses."""
    user_id, dataset_id = uuid4(), uuid4()
    asks = [
        Ask(
            text=RUNBOOKS,
            user_id=user_id,
            dataset_id=dataset_id,
            first_seen=BASE_TIME,
            last_seen=BASE_TIME,
            session_id="codex_a1",
        ),
        Ask(
            text=RUNBOOKS_REPHRASED,
            user_id=user_id,
            dataset_id=dataset_id,
            first_seen=BASE_TIME - timedelta(hours=2),
            last_seen=BASE_TIME - timedelta(hours=2),
            session_id="claude_a1",
        ),
    ]

    question = dedup_asks(asks, _matrix(asks), dedup_threshold=0.92).questions[0]

    assert question.relevance == 2
    assert question.first_asked_at == BASE_TIME - timedelta(hours=2)
    assert question.session_id == "claude_a1"


def test_an_observed_row_with_no_session_carries_none_rather_than_a_label():
    """dedup is config-free: it carries the raw id, and the label is resolved above it."""
    user_id, dataset_id = uuid4(), uuid4()
    rows = [_row(RUNBOOKS, user_id=user_id, dataset_id=dataset_id, created_at=BASE_TIME)]

    collapsed = _collapse(rows)
    question = dedup_asks(collapsed.asks, _matrix(collapsed.asks), dedup_threshold=0.92).questions[
        0
    ]

    assert question.relevance == 1
    assert question.session_id is None


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

    assert collapsed.recall_count == history_size
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

    assert collapsed.recall_count == 0
    assert collapsed.distinct_ask_count == 0
    assert result.questions == []
    assert result.comparison_count == 0


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
