"""``_tensor`` must behave like torch for every operation gliner2's glue calls.

Compared against real torch where it is installed (the ``gliner`` extra);
skipped otherwise.
"""

import numpy as np
import pytest

from cognee.tasks.graph.gliner_demo.onnx import _tensor as tt

torch = pytest.importorskip("torch")
rng = np.random.default_rng(0)


def _pair(array):
    return tt.tensor(array), torch.tensor(array)


def _same(ours, theirs):
    if isinstance(theirs, tuple):
        assert all(_same(a, b) for a, b in zip(ours, theirs))
        return True
    np.testing.assert_array_equal(np.asarray(ours), theirs.detach().cpu().numpy())
    assert np.asarray(ours).dtype == theirs.numpy().dtype
    return True


def test_stable_argsort_orders_ties_like_torch_both_directions():
    values = rng.integers(0, 4, size=(3, 17)).astype(np.float32)  # many ties
    ours, theirs = _pair(values)
    for descending in (False, True):
        _same(
            tt.argsort(ours, dim=-1, descending=descending, stable=True),
            torch.argsort(theirs, dim=-1, descending=descending, stable=True),
        )


def test_gather_expand_masked_fill_clamp_and_nonzero():
    values = rng.standard_normal((2, 5, 4)).astype(np.float32)
    index = rng.integers(0, 5, size=(2, 3, 4))
    ours, theirs = _pair(values)
    _same(ours.gather(1, tt.tensor(index)), theirs.gather(1, torch.tensor(index)))
    mask = values > 0
    _same(ours.masked_fill(tt.tensor(mask), -1e4), theirs.masked_fill(torch.tensor(mask), -1e4))
    ints = rng.integers(-5, 9, size=(3, 4))
    _same(tt.tensor(ints).clamp(0, 3), torch.tensor(ints).clamp(0, 3))
    _same(tt.tensor(mask).nonzero(), torch.tensor(mask).nonzero())
    row = rng.standard_normal((1, 4)).astype(np.float32)
    _same(tt.tensor(row).expand(3, -1), torch.tensor(row).expand(3, -1))


def test_reductions_and_dtypes_match():
    mask = rng.random((3, 6)) > 0.5
    _same(tt.tensor(mask).sum(-1), torch.tensor(mask).sum(-1))  # bool sums are int64
    _same(tt.tensor(mask).all(1), torch.tensor(mask).all(1))
    _same(tt.arange(5), torch.arange(5))
    _same(tt.tensor([1, 2]), torch.tensor([1, 2]))
    _same(tt.tensor([1.5]), torch.tensor([1.5]))


def test_element_access_returns_a_tensor_the_caller_can_keep_using():
    probabilities = tt.tensor([0.25, 0.75])
    item = probabilities[1]
    assert isinstance(item, tt.Tensor) and float(item.detach()) == 0.75
    assert [float(p) for p in probabilities] == [0.25, 0.75]


def test_sigmoid_matches_and_is_stable_at_mask_logits():
    values = np.array([-1e4, -30.0, 0.0, 12.5, 1e4], dtype=np.float32)
    with np.errstate(over="raise"):
        ours = tt.sigmoid(tt.tensor(values))
    np.testing.assert_allclose(
        np.asarray(ours), torch.sigmoid(torch.tensor(values)).numpy(), rtol=1e-6, atol=1e-7
    )
