"""The torch operations gliner2's inference glue uses, implemented on numpy.

gliner2's tokenization, batching and decoding code handles its tensors with
about 46 plain array operations (indexing, gather, reshape, stable argsort,
masked_fill, ...) and runs no network outside the three ONNX graphs. The
copied gliner2 modules in ``_gliner2`` import this module as ``torch``, so
their logic stays byte-identical to upstream apart from import lines, and the
extraction needs no torch install. Only what that code path calls is here,
with torch's semantics; anything else fails loudly with AttributeError.
"""

from __future__ import annotations

import contextlib
from collections import namedtuple

import numpy as np

# ---------------------------------------------------------------- dtypes / devices

long = int64 = np.dtype(np.int64)
int32 = np.dtype(np.int32)
float32 = float = np.dtype(np.float32)
float64 = double = np.dtype(np.float64)
float16 = half = np.dtype(np.float16)
bfloat16 = np.dtype(np.float32)  # never produced on this CPU path
bool = np.dtype(np.bool_)
uint8 = np.dtype(np.uint8)
dtype = np.dtype


class device(str):
    def __new__(cls, spec="cpu", index=None):
        return super().__new__(cls, str(spec))

    @property
    def type(self):
        return str(self).split(":")[0]


CPU = device("cpu")
finfo = np.finfo
iinfo = np.iinfo
_values_indices = namedtuple("values_indices", ["values", "indices"])


def _dt(value):
    return None if value is None else np.dtype(value)


def _axis(dim):
    return None if dim is None else int(dim)


# ---------------------------------------------------------------- tensor


class Tensor(np.ndarray):
    """A numpy array with the torch.Tensor methods the copied code calls."""

    def __array_finalize__(self, obj):
        pass

    # numpy returns bare scalars from element access and reductions; torch
    # returns 0-d tensors the caller keeps calling methods on.
    def __getitem__(self, index):
        if isinstance(index, Tensor):
            index = np.asarray(index)
        elif isinstance(index, tuple):
            index = tuple(np.asarray(i) if isinstance(i, Tensor) else i for i in index)
        return _t(super().__getitem__(index))

    def __setitem__(self, index, value):
        if isinstance(index, Tensor):
            index = np.asarray(index)
        elif isinstance(index, tuple):
            index = tuple(np.asarray(i) if isinstance(i, Tensor) else i for i in index)
        super().__setitem__(index, np.asarray(value) if isinstance(value, Tensor) else value)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __hash__(self):
        return id(self)

    # --- conversion
    @property
    def device(self):
        return CPU

    @property
    def requires_grad(self):
        return False

    def to(self, *args, **kwargs):
        target = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, np.dtype) or (isinstance(arg, type) and issubclass(arg, np.generic)):
                target = arg
            elif isinstance(arg, Tensor):
                target = arg.dtype
        return self if target is None or np.dtype(target) == self.dtype else _t(self.astype(target))

    def long(self):
        return self.to(long)

    def int(self):
        return self.to(int32)

    def float(self):
        return self.to(float32)

    def double(self):
        return self.to(float64)

    def bool(self):
        return self.to(bool)

    def type_as(self, other):
        return self.to(other.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _t(np.array(self, copy=True))

    def contiguous(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def dim(self):
        return self.ndim

    def numel(self):
        return int(np.asarray(self).size)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def element_size(self):
        return self.itemsize

    # --- shape
    def unsqueeze(self, dim):
        return _t(np.expand_dims(np.asarray(self), dim))

    def squeeze(self, dim=None):
        if dim is None:
            return _t(np.squeeze(np.asarray(self)))
        return _t(np.squeeze(np.asarray(self), dim)) if self.shape[dim] == 1 else self

    def view(self, *shape):
        return self.reshape(*shape)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _t(np.reshape(np.asarray(self), tuple(int(s) for s in shape)))

    def flatten(self, start_dim=0, end_dim=-1):
        nd = self.ndim
        start, end = start_dim % nd if nd else 0, end_dim % nd if nd else 0
        new = self.shape[:start] + (-1,) + self.shape[end + 1 :]
        return self.reshape(*new)

    def expand(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        lead = len(sizes) - self.ndim
        target = tuple(
            self.shape[i - lead] if (s == -1 and i >= lead) else int(s) for i, s in enumerate(sizes)
        )
        return _t(np.broadcast_to(np.asarray(self), target))

    def expand_as(self, other):
        return self.expand(*other.shape)

    def transpose(self, dim0, dim1):
        return _t(np.swapaxes(np.asarray(self), dim0, dim1))

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return _t(np.transpose(np.asarray(self), dims))

    def unbind(self, dim=0):
        return tuple(_t(x) for x in np.moveaxis(np.asarray(self), dim, 0))

    def split(self, size, dim=0):
        return tuple(
            _t(x) for x in np.split(np.asarray(self), range(size, self.shape[dim], size), axis=dim)
        )

    def repeat(self, *sizes):
        return _t(np.tile(np.asarray(self), sizes))

    # --- indexing
    def gather(self, dim, index):
        return _t(np.take_along_axis(np.asarray(self), np.asarray(index), axis=dim))

    def index_select(self, dim, index):
        return _t(np.take(np.asarray(self), np.asarray(index), axis=dim))

    def masked_fill(self, mask, value):
        return _t(np.where(np.asarray(mask), np.asarray(value, dtype=self.dtype), np.asarray(self)))

    def nonzero(self, as_tuple=False):
        if as_tuple:
            return tuple(_t(i) for i in np.nonzero(np.asarray(self)))
        return _t(np.argwhere(np.asarray(self)))

    # --- elementwise
    def clamp(self, min=None, max=None):
        out = np.asarray(self)
        if min is not None:
            out = np.maximum(out, np.asarray(min, dtype=self.dtype))
        if max is not None:
            out = np.minimum(out, np.asarray(max, dtype=self.dtype))
        return _t(out)

    def clamp_min(self, value):
        return self.clamp(min=value)

    def clamp_max(self, value):
        return self.clamp(max=value)

    def abs(self):
        return _t(np.abs(np.asarray(self)))

    def sign(self):
        return _t(np.sign(np.asarray(self)))

    def sigmoid(self):
        return sigmoid(self)

    def exp(self):
        return _t(np.exp(np.asarray(self)))

    def log(self):
        return _t(np.log(np.asarray(self)))

    def ge(self, other):
        return self >= other

    def gt(self, other):
        return self > other

    def le(self, other):
        return self <= other

    def lt(self, other):
        return self < other

    def eq(self, other):
        return self == other

    def ne(self, other):
        return self != other

    def add(self, other):
        return self + other

    def sub(self, other):
        return self - other

    def mul(self, other):
        return self * other

    def div(self, other, rounding_mode=None):
        return div(self, other, rounding_mode=rounding_mode)

    def logical_not(self):
        return ~self

    # --- reductions: torch's ``dim``/``keepdim``, and numpy's ``axis``/``keepdims``
    # because numpy functions call these methods internally. Bool sums are int64.
    def _reduce(self, fn, dim=None, keepdim=False, axis=None, keepdims=None, dtype=None, **_):
        arr = np.asarray(self)
        axis = _axis(dim if dim is not None else axis)
        keep = keepdim if keepdims is None else keepdims
        if fn is np.sum and dtype is None and arr.dtype == np.bool_:
            dtype = np.int64
        kwargs = {"dtype": dtype} if dtype is not None else {}
        return _t(fn(arr, axis=axis, keepdims=keep, **kwargs))

    def sum(self, dim=None, keepdim=False, **kwargs):
        return self._reduce(np.sum, dim, keepdim, **kwargs)

    def mean(self, dim=None, keepdim=False, **kwargs):
        return self._reduce(np.mean, dim, keepdim, **kwargs)

    def all(self, dim=None, keepdim=False, **kwargs):
        return self._reduce(np.all, dim, keepdim, **kwargs)

    def any(self, dim=None, keepdim=False, **kwargs):
        return self._reduce(np.any, dim, keepdim, **kwargs)

    def max(self, dim=None, keepdim=False, axis=None, **_):
        return _reduce_pair(self, dim if dim is not None else axis, keepdim, np.max, np.argmax)

    def min(self, dim=None, keepdim=False, axis=None, **_):
        return _reduce_pair(self, dim if dim is not None else axis, keepdim, np.min, np.argmin)

    def argmax(self, dim=None, keepdim=False, axis=None, **_):
        return _t(
            np.argmax(
                np.asarray(self), axis=_axis(dim if dim is not None else axis), keepdims=keepdim
            )
        )

    def cumsum(self, dim):
        return _t(np.cumsum(np.asarray(self), axis=dim))

    def argsort(self, dim=-1, descending=False, stable=False):
        return argsort(self, dim=dim, descending=descending, stable=stable)

    def sort(self, dim=-1, descending=False, stable=False):
        return sort(self, dim=dim, descending=descending, stable=stable)

    # --- factories off an existing tensor
    def new_zeros(self, *size, dtype=None, device=None):
        return zeros(*size, dtype=dtype or self.dtype)

    def new_ones(self, *size, dtype=None, device=None):
        return ones(*size, dtype=dtype or self.dtype)

    def new_full(self, size, fill_value, dtype=None, device=None):
        return full(size, fill_value, dtype=dtype or self.dtype)

    def new_empty(self, *size, dtype=None, device=None):
        return zeros(*size, dtype=dtype or self.dtype)

    def new_tensor(self, data, dtype=None, device=None):
        return tensor(data, dtype=dtype or self.dtype)


def _reduce_pair(x, dim, keepdim, value_fn, index_fn):
    arr = np.asarray(x)
    if dim is None:
        return _t(value_fn(arr))
    return _values_indices(
        _t(value_fn(arr, axis=dim, keepdims=keepdim)),
        _t(index_fn(arr, axis=dim, keepdims=keepdim).astype(np.int64)),
    )


def _t(value):
    """Wrap anything array-like (including numpy scalars) as a Tensor."""
    if isinstance(value, Tensor):
        return value
    return np.asarray(value).view(Tensor)


LongTensor = BoolTensor = FloatTensor = Tensor
Size = tuple


def is_tensor(value) -> bool:
    return isinstance(value, Tensor)


def from_numpy(array):
    return _t(array)


# ---------------------------------------------------------------- factories


def _size(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return tuple(int(s) for s in size)


def tensor(data, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        data = np.asarray(data)
    arr = np.array(data, dtype=_dt(dtype))
    if dtype is None:
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        elif arr.dtype.kind in "iu":
            arr = arr.astype(np.int64)
    return _t(arr)


as_tensor = tensor


def zeros(*size, dtype=None, device=None):
    return _t(np.zeros(_size(size), dtype=_dt(dtype) or np.float32))


def ones(*size, dtype=None, device=None):
    return _t(np.ones(_size(size), dtype=_dt(dtype) or np.float32))


def empty(*size, dtype=None, device=None):
    return zeros(*size, dtype=dtype)


def full(size, fill_value, dtype=None, device=None):
    if dtype is None:
        dtype = (
            np.bool_
            if isinstance(fill_value, (np.bool_,)) or type(fill_value).__name__ == "bool"
            else (np.int64 if isinstance(fill_value, (int, np.integer)) else np.float32)
        )
    return _t(
        np.full(
            _size((size,)) if not isinstance(size, (tuple, list)) else _size(size),
            fill_value,
            dtype=dtype,
        )
    )


def zeros_like(x, dtype=None, device=None):
    return _t(np.zeros_like(np.asarray(x), dtype=_dt(dtype)))


def ones_like(x, dtype=None, device=None):
    return _t(np.ones_like(np.asarray(x), dtype=_dt(dtype)))


def full_like(x, fill_value, dtype=None, device=None):
    return _t(np.full_like(np.asarray(x), fill_value, dtype=_dt(dtype)))


def empty_like(x, dtype=None, device=None):
    return zeros_like(x, dtype=dtype)


def arange(start, end=None, step=1, dtype=None, device=None):
    if end is None:
        start, end = 0, start
    out = np.arange(start, end, step)
    if dtype is not None:
        out = out.astype(dtype)
    elif out.dtype.kind in "iu":
        out = out.astype(np.int64)
    return _t(out)


# ---------------------------------------------------------------- functions


def cat(tensors, dim=0):
    return _t(np.concatenate([np.asarray(t) for t in tensors], axis=dim))


concat = cat


def stack(tensors, dim=0):
    return _t(np.stack([np.asarray(t) for t in tensors], axis=dim))


def where(condition, x=None, y=None):
    if x is None:
        return tuple(_t(i) for i in np.nonzero(np.asarray(condition)))
    return _t(np.where(np.asarray(condition), np.asarray(x), np.asarray(y)))


def sigmoid(x):
    arr = np.asarray(x)
    z = np.exp(-np.abs(arr.astype(np.float64)))  # stable for the -1e4 mask logits
    return _t(np.where(arr >= 0, 1.0 / (1.0 + z), z / (1.0 + z)).astype(arr.dtype))


def softmax(x, dim=-1):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.max(arr, axis=dim, keepdims=True)
    e = np.exp(arr)
    return _t((e / e.sum(axis=dim, keepdims=True)).astype(np.asarray(x).dtype))


def argsort(x, dim=-1, descending=False, stable=False):
    arr = np.asarray(x)
    if descending:
        # Negation preserves order for numbers; stable keeps ties in index order,
        # which is what torch's stable descending sort guarantees too.
        arr = -(arr.astype(np.int64) if arr.dtype == np.bool_ else arr)
    return _t(np.argsort(arr, axis=dim, kind="stable").astype(np.int64))


def sort(x, dim=-1, descending=False, stable=False):
    order = argsort(x, dim=dim, descending=descending, stable=stable)
    return _values_indices(
        _t(np.take_along_axis(np.asarray(x), np.asarray(order), axis=dim)), order
    )


def topk(x, k, dim=-1, largest=True, sorted=True):
    order = argsort(x, dim=dim, descending=largest, stable=True)
    index = np.take(np.asarray(order), np.arange(int(k)), axis=dim)
    return _values_indices(_t(np.take_along_axis(np.asarray(x), index, axis=dim)), _t(index))


def gather(x, dim, index):
    return _t(x).gather(dim, index)


def clamp(x, min=None, max=None):
    return _t(x).clamp(min=min, max=max)


def minimum(a, b):
    return _t(np.minimum(np.asarray(a), np.asarray(b)))


def maximum(a, b):
    return _t(np.maximum(np.asarray(a), np.asarray(b)))


def div(a, b, rounding_mode=None):
    a_, b_ = np.asarray(a), np.asarray(b)
    if rounding_mode == "floor":
        return _t(np.floor_divide(a_, b_))
    if rounding_mode == "trunc":
        return _t(np.trunc(a_ / b_).astype(np.result_type(a_, b_)))
    out = a_ / b_
    return _t(out.astype(np.float32) if out.dtype == np.float64 and a_.dtype != np.float64 else out)


def isfinite(x):
    return _t(np.isfinite(np.asarray(x)))


def isinf(x):
    return _t(np.isinf(np.asarray(x)))


def isnan(x):
    return _t(np.isnan(np.asarray(x)))


def logical_and(a, b):
    return _t(np.logical_and(np.asarray(a), np.asarray(b)))


def logical_or(a, b):
    return _t(np.logical_or(np.asarray(a), np.asarray(b)))


def logical_not(a):
    return _t(np.logical_not(np.asarray(a)))


def sign(x):
    return _t(np.sign(np.asarray(x)))


def abs(x):
    return _t(np.abs(np.asarray(x)))


def exp(x):
    return _t(np.exp(np.asarray(x)))


def log(x):
    return _t(np.log(np.asarray(x)))


def sum(x, dim=None, keepdim=False):
    return _t(x).sum(dim=dim, keepdim=keepdim)


def unique(x, sorted=True, return_inverse=False):
    out = np.unique(np.asarray(x), return_inverse=return_inverse)
    return tuple(_t(o) for o in out) if return_inverse else _t(out)


# ---------------------------------------------------------------- no-op contexts


class _NoGrad(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def inference_mode(mode=True):
    return _NoGrad()


def no_grad():
    return _NoGrad()


def set_grad_enabled(mode=True):
    return _NoGrad()


def is_grad_enabled():
    return False


class _Cuda:
    @staticmethod
    def is_available():
        return False


cuda = _Cuda()


# ---------------------------------------------------------------- nn stand-ins
#
# Copied modules define torch model classes next to the inference helpers
# (``class SparseRelationScorer(nn.Module)``). The classes only need to be
# importable; building one means the neural path was reached, which runs on
# ONNX here, so constructors raise.


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is a torch layer; the torch-free GLiNER path runs "
            "neural pieces on ONNX Runtime"
        )


class _Module:
    """Base for the stand-in model objects: device lookup and eval() only."""

    training = False

    def __init__(self, *args, **kwargs):
        pass

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def parameters(self):
        yield zeros(1)


class _NN:
    Module = _Module
    Parameter = _Unavailable
    ModuleList = ModuleDict = Sequential = _Unavailable
    Linear = LayerNorm = Embedding = Dropout = GELU = SiLU = ReLU = Identity = _Unavailable
    MultiheadAttention = LSTM = GRU = Conv1d = Bilinear = _Unavailable

    class init:
        pass

    class utils:
        class rnn:
            @staticmethod
            def pad_sequence(sequences, batch_first=False, padding_value=0.0):
                return pad_sequence(sequences, batch_first, padding_value)


nn = _NN()


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    sequences = [np.asarray(s) for s in sequences]
    longest = max(s.shape[0] for s in sequences)
    out = np.full(
        (len(sequences), longest, *sequences[0].shape[1:]), padding_value, dtype=sequences[0].dtype
    )
    for i, s in enumerate(sequences):
        out[i, : s.shape[0]] = s
    return _t(out if batch_first else np.swapaxes(out, 0, 1))


class _Functional:
    sigmoid = staticmethod(sigmoid)
    softmax = staticmethod(softmax)

    @staticmethod
    def pad(x, pad, value=0.0, mode="constant"):
        arr = np.asarray(x)
        widths = [(0, 0)] * arr.ndim
        for i in range(0, len(pad), 2):
            widths[arr.ndim - 1 - i // 2] = (pad[i], pad[i + 1])
        return _t(np.pad(arr, widths, constant_values=value))


functional = _Functional()


class DataLoader:
    """``torch.utils.data.DataLoader`` as inference uses it: ordered, in-process batches."""

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, **_):
        if shuffle or num_workers:
            raise NotImplementedError(
                "the torch-free DataLoader only yields ordered, in-process batches"
            )
        self.dataset, self.batch_size, self.collate_fn = list(dataset), int(batch_size), collate_fn

    def __iter__(self):
        for start in range(0, len(self.dataset), self.batch_size):
            chunk = self.dataset[start : start + self.batch_size]
            yield self.collate_fn(chunk) if self.collate_fn else chunk

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
