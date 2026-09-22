"""The mocked example runner's seeded mode: deterministic, non-zero embeddings."""

import math

from cognee.tests.utils.run_mocked import _hashed_unit_vector


def test_hashed_unit_vector_is_deterministic_and_normalized():
    a = _hashed_unit_vector("Audi is known for its modern designs", 300)
    b = _hashed_unit_vector("Audi is known for its modern designs", 300)

    assert a == b
    assert len(a) == 300
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
    assert any(x != 0.0 for x in a)


def test_hashed_unit_vector_differs_across_texts():
    a = _hashed_unit_vector("Audi", 64)
    b = _hashed_unit_vector("Apple", 64)

    assert a != b
    # Different directions, so cosine similarity is defined and not degenerate.
    cosine = sum(x * y for x, y in zip(a, b))
    assert abs(cosine) < 1.0
