"""``score_rows_against_queries``: one list per query, cosine distance, order kept."""

import math
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.local_scoring import score_rows_against_queries


def test_scores_every_row_against_every_query_in_order():
    first, second = uuid4(), uuid4()
    rows = [(str(first), [1.0, 0.0]), (str(second), [1.0, 1.0])]

    scores = score_rows_against_queries(rows, [[1.0, 0.0], [0.0, 1.0]])

    assert [[r.id for r in query_scores] for query_scores in scores] == [[first, second]] * 2
    assert [r.score for r in scores[0]] == pytest.approx([0.0, 1 - 1 / math.sqrt(2)])
    assert [r.score for r in scores[1]] == pytest.approx([1.0, 1 - 1 / math.sqrt(2)])
    assert all(r.payload is None for query_scores in scores for r in query_scores)


def test_no_rows_or_no_queries_gives_empty_lists():
    assert score_rows_against_queries([], [[1.0, 0.0]]) == [[]]
    assert score_rows_against_queries([(str(uuid4()), [1.0, 0.0])], []) == []
