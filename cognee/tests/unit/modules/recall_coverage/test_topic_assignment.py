"""Guards on recall-coverage topic assignment — spec section 2, phase 2 step 7.

Four invariants, each protecting a number the report prints:

* **Threshold AND margin.** A question that clears the threshold but sits between
  two topics goes to the sink, not to whichever topic won by a hair — a coin flip
  there moves two ``topics[].avg_score`` values.
* **One question, one topic**, and the sink is the wire literal ``"other"``, never
  a stored row.
* **A fingerprint mismatch fails the run.** Stored centroids are never re-embedded
  behind the operator's back: a cosine between two embedding spaces is a
  confident number about nothing.
* **Topics are owner-scoped and label-blind**, so two agents' runs are scored
  against the same taxonomy and their per-topic scores are comparable.

Every vector here is written out by hand, so each similarity is exact and no
assertion depends on a live embedding provider.
"""

import math
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from cognee.modules.recall_coverage.assign import (
    assign_topics,
    canonical_matrix,
    require_matching_fingerprint,
    stack_centroids,
)
from cognee.modules.recall_coverage.embedding import (
    EmbeddingFingerprint,
    engine_fingerprint,
    normalize_rows,
)
from cognee.modules.recall_coverage.exceptions import EmbeddingFingerprintMismatchError
from cognee.modules.recall_coverage.types import SINK_TOPIC_ID, SINK_TOPIC_LABEL

MODEL = "openai/text-embedding-3-large"
FINGERPRINT = EmbeddingFingerprint(model=MODEL, dimensions=2)


def _topic(centroid, label="Billing & invoices", model=MODEL, dimensions=2):
    """A stand-in for ``repository.TopicRecord`` — the five attributes assign reads."""
    return SimpleNamespace(
        id=uuid4(),
        label=label,
        centroid=tuple(float(value) for value in centroid),
        embedding_model=model,
        embedding_dimensions=dimensions,
    )


def _unit(x, y):
    norm = math.hypot(x, y)
    return (x / norm, y / norm)


def _vectors(*rows):
    return normalize_rows([list(row) for row in rows])


# --- Threshold and margin ---------------------------------------------------


def test_a_question_on_a_topic_is_assigned_to_it():
    billing = _topic((1.0, 0.0), label="Billing & invoices")
    incidents = _topic((0.0, 1.0), label="Incidents")

    result = assign_topics(
        _vectors((1.0, 0.05)),
        [billing, incidents],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )

    assignment = result.assignments[0]
    assert assignment.topic_id == billing.id
    assert assignment.topic_label == "Billing & invoices"
    assert assignment.similarity > 0.99
    assert result.sink_indices == []
    assert result.assigned_topic_ids == (billing.id,)


def test_below_the_threshold_goes_to_the_sink():
    """A question weakly related to its best topic is not that topic's question."""
    billing = _topic((1.0, 0.0))

    result = assign_topics(
        # cos ~= 0.44 to billing: related, not about it.
        _vectors((0.5, 1.0)),
        [billing],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )

    assignment = result.assignments[0]
    assert assignment.is_sink
    assert assignment.topic_id is None
    assert assignment.topic_label == SINK_TOPIC_LABEL
    assert assignment.wire_topic_id == SINK_TOPIC_ID
    assert result.sink_indices == [0]
    assert result.assigned_topic_ids == ()


def test_a_question_between_two_topics_goes_to_the_sink_on_the_margin():
    """The margin, not the threshold, is what rejects this one.

    The question sits at 45 degrees between two orthogonal topics: cosine ~0.707
    to both, comfortably over ``assignment_threshold``, but the gap between best
    and runner-up is ~0. Assigning it would credit or blame one of two topics at
    random; the sink is the honest answer and the signal that the taxonomy is
    missing something.
    """
    billing = _topic((1.0, 0.0), label="Billing & invoices")
    incidents = _topic((0.0, 1.0), label="Incidents")
    question = _vectors(_unit(1.0, 1.0))

    tie = assign_topics(
        question,
        [billing, incidents],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )
    assert tie.assignments[0].is_sink
    assert tie.assignments[0].similarity == pytest.approx(0.7071, abs=1e-3)
    assert tie.assignments[0].runner_up_similarity == pytest.approx(0.7071, abs=1e-3)

    # The threshold alone would have assigned it, which is the bug the margin
    # exists to prevent.
    without_margin = assign_topics(
        question,
        [billing, incidents],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.0,
    )
    assert without_margin.assignments[0].topic_id == billing.id


def test_a_single_topic_has_no_runner_up_so_the_margin_cannot_reject():
    """With one topic there is nothing to be ambiguous between."""
    billing = _topic((1.0, 0.0))

    result = assign_topics(
        _vectors((1.0, 0.0)),
        [billing],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        # A margin no pair of similarities could ever satisfy.
        assignment_margin=1.0,
    )

    assert result.assignments[0].topic_id == billing.id
    assert result.assignments[0].runner_up_similarity is None


# --- One question, one topic; the sink ---------------------------------------


def test_each_question_gets_exactly_one_assignment_in_input_order():
    billing = _topic((1.0, 0.0), label="Billing & invoices")
    incidents = _topic((0.0, 1.0), label="Incidents")

    result = assign_topics(
        _vectors((1.0, 0.0), (0.0, 1.0), _unit(1.0, 1.0)),
        [billing, incidents],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )

    assert [assignment.topic_id for assignment in result.assignments] == [
        billing.id,
        incidents.id,
        None,
    ]
    assert result.sink_indices == [2]
    assert result.sink_question_count == 1
    assert set(result.assigned_topic_ids) == {billing.id, incidents.id}


def test_no_topics_sends_every_question_to_the_sink():
    """The first run happens before a single topic has been accepted."""
    result = assign_topics(
        _vectors((1.0, 0.0), (0.0, 1.0)),
        [],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )

    assert result.sink_indices == [0, 1]
    assert all(assignment.wire_topic_id == SINK_TOPIC_ID for assignment in result.assignments)
    assert result.assigned_topic_ids == ()


def test_no_questions_assigns_nothing():
    result = assign_topics(
        np.zeros((0, 0)),
        [_topic((1.0, 0.0))],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )

    assert result.assignments == []
    assert result.sink_indices == []


def test_a_failed_embedding_lands_in_the_sink_rather_than_a_topic():
    """A zero row (a batch that failed open) is cosine-similar to nothing."""
    billing = _topic((1.0, 0.0))

    result = assign_topics(
        normalize_rows([[0.0, 0.0], [1.0, 0.0]]),
        [billing],
        fingerprint=FINGERPRINT,
        assignment_threshold=0.55,
        assignment_margin=0.05,
    )

    assert result.assignments[0].is_sink
    assert result.assignments[0].similarity == 0.0
    assert result.assignments[1].topic_id == billing.id


# --- Fingerprint ------------------------------------------------------------


def test_a_different_embedding_model_fails_the_run():
    stale = _topic((1.0, 0.0), model="openai/text-embedding-ada-002")

    with pytest.raises(EmbeddingFingerprintMismatchError) as error:
        assign_topics(
            _vectors((1.0, 0.0)),
            [stale],
            fingerprint=FINGERPRINT,
            assignment_threshold=0.55,
            assignment_margin=0.05,
        )

    # The message has to name both spaces, or the operator cannot act on it.
    assert "ada-002" in error.value.message
    assert MODEL in error.value.message


def test_a_different_embedding_dimension_fails_the_run():
    stale = _topic((1.0, 0.0), dimensions=1536)

    with pytest.raises(EmbeddingFingerprintMismatchError):
        require_matching_fingerprint([stale], FINGERPRINT)


def test_an_engine_that_reports_no_dimension_does_not_fail_on_dimension():
    """A missing accessor is not a mismatch; only a contradiction is."""
    topic = _topic((1.0, 0.0), dimensions=1536)

    require_matching_fingerprint([topic], EmbeddingFingerprint(model=MODEL, dimensions=0))


def test_a_centroid_whose_width_disagrees_with_the_questions_fails_the_run():
    """A stored vector contradicting its own recorded width is a mismatch, not a crash."""
    corrupt = _topic((1.0, 0.0, 0.0), dimensions=2)

    with pytest.raises(EmbeddingFingerprintMismatchError):
        stack_centroids([corrupt], 2)


def test_topics_are_never_silently_re_embedded():
    """The mismatch path must not call an embedding engine at all.

    The whole point of failing is that the operator decides; a module that quietly
    re-embedded would move every stored centroid and reset the score trend that
    stable topic ids exist to carry.
    """
    calls: list[list[str]] = []

    class _Engine:
        model = MODEL

        def get_vector_size(self) -> int:
            return 2

        async def embed_text(self, texts):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    engine = _Engine()
    assert engine_fingerprint(engine) == FINGERPRINT

    with pytest.raises(EmbeddingFingerprintMismatchError):
        assign_topics(
            _vectors((1.0, 0.0)),
            [_topic((1.0, 0.0), model="some/other-model")],
            fingerprint=engine_fingerprint(engine),
            assignment_threshold=0.55,
            assignment_margin=0.05,
        )

    assert calls == []


# --- Reuse of the dedup vectors ---------------------------------------------


def test_canonical_matrix_reuses_the_ask_level_vectors():
    """Assignment scores the vector dedup already paid for, never a fresh one."""
    normalized = _vectors((1.0, 0.0), (0.0, 1.0), _unit(1.0, 1.0))
    questions = [SimpleNamespace(canonical_index=2), SimpleNamespace(canonical_index=0)]

    matrix = canonical_matrix(questions, normalized)

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[0], normalized[2])
    assert np.allclose(matrix[1], normalized[0])


def test_canonical_matrix_of_no_questions_is_empty():
    assert canonical_matrix([], _vectors((1.0, 0.0))).shape == (0, 0)
