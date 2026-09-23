"""The GLiNER demo extractor's ONNX Runtime backend, without the real model.

The neural pieces are exercised against fake ONNX sessions here; real
torch-vs-ONNX parity on the downloaded model is checked by
``test_gliner_onnx_parity.py`` (opt-in, needs an export).
"""

import dataclasses
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import cognee.tasks.graph.gliner_demo.extractor as extractor_module
from cognee.modules.cognify.config import get_cognify_config
from cognee.tasks.graph.gliner_demo.onnx import runtime
from cognee.tasks.graph.gliner_demo.onnx.tree import flatten, unflatten

MODEL = "fastino/gliner2.5-base-v1"


@dataclasses.dataclass
class _Inner:
    scores: torch.Tensor
    label: str = "x"


@dataclasses.dataclass
class _Outer:
    inner: _Inner
    extra: torch.Tensor | None = None
    batch_size: int = 0


def test_tree_round_trips_dataclasses_and_fills_symbolic_fields():
    obj = _Outer(_Inner(torch.arange(3.0)), torch.ones(2), batch_size=4)
    spec, tensors = flatten(obj)
    spec["f"]["batch_size"] = {"sym": True}  # what torch.export leaves for shape-derived ints

    rebuilt = unflatten(json.loads(json.dumps(spec)), tensors, {"batch_size": 7})

    assert isinstance(rebuilt, _Outer) and isinstance(rebuilt.inner, _Inner)
    assert torch.equal(rebuilt.inner.scores, obj.inner.scores) and rebuilt.inner.label == "x"
    assert rebuilt.batch_size == 7


@pytest.fixture
def backend_env(monkeypatch):
    def set_backend(value):
        monkeypatch.setenv("GLINER_BACKEND", value)
        get_cognify_config.cache_clear()

    yield set_backend
    get_cognify_config.cache_clear()


def test_unknown_backend_is_an_error(backend_env):
    backend_env("tensorrt")
    with pytest.raises(ValueError, match="Unknown GLINER_BACKEND"):
        extractor_module.resolve_gliner_backend()


def test_onnx_backend_without_an_export_says_how_to_create_one(backend_env, monkeypatch, tmp_path):
    backend_env("onnx")
    monkeypatch.setenv(runtime.ONNX_DIR_ENV, str(tmp_path))
    fake = SimpleNamespace(
        AutoExtractor=SimpleNamespace(from_pretrained=MagicMock(return_value=SimpleNamespace()))
    )
    monkeypatch.setitem(sys.modules, "gliner2", fake)
    monkeypatch.setattr(extractor_module, "_extractors", {})
    monkeypatch.setattr(extractor_module, "hub_model_cached", lambda _m: (True, "cache"))

    with pytest.raises(runtime.OnnxModelNotExportedError, match="onnx.export --model"):
        extractor_module.load_extractor(MODEL)


def _export_dir(tmp_path, **meta_overrides):
    out = tmp_path / MODEL.replace("/", "--")
    out.mkdir(parents=True)
    for graph in runtime.GRAPHS:
        (out / graph).write_bytes(b"")
    meta = {"model": MODEL, "gliner2_version": "2.0.0"} | meta_overrides
    (out / "meta.json").write_text(json.dumps(meta))
    return out


def test_an_export_of_another_model_is_refused(tmp_path, monkeypatch):
    out = _export_dir(tmp_path, model="someone/else")
    with pytest.raises(RuntimeError, match="holds an export of someone/else"):
        runtime.attach_onnx_backend(SimpleNamespace(), MODEL, out)


def test_an_export_from_another_gliner2_version_is_refused(tmp_path, monkeypatch):
    out = _export_dir(tmp_path, gliner2_version="1.9.0")
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "2.0.0")
    with pytest.raises(RuntimeError, match="exported with gliner2 1.9.0"):
        runtime.attach_onnx_backend(SimpleNamespace(), MODEL, out)


class _FakeSession:
    """Records feeds; returns outputs shaped like the head's for the padded length."""

    def __init__(self, names, respond):
        self._names, self._respond, self.feeds = names, respond, []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self._names]

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        return self._respond(feeds)


def test_short_texts_are_padded_to_the_export_minimum_and_sliced_back():
    def respond(feeds):
        b, lp, _ = feeds["text_states"].shape
        q = feeds["query_states"].shape[1]
        return [
            np.zeros((b, q, lp + 1), np.float32),
            np.zeros((b, q, lp), np.float32),
            np.zeros((b, q), np.float32),
        ]

    session = _FakeSession(["text_states", "text_mask", "query_states", "query_mask"], respond)
    spec, _ = flatten((torch.zeros(1), torch.zeros(1), torch.zeros(1)))
    meta = {
        "boundary_head_output_spec": spec,
        "min_text_length": 32,
        "boundary_head_length_axes": [[[2, 1]], [[2, 0]], []],
    }
    head = runtime.OnnxBoundaryHead(session, meta, settings=None)

    start, inside, other = head(
        torch.ones(2, 5, 8),
        torch.ones(2, 5, dtype=torch.bool),
        torch.ones(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
    )

    fed = session.feeds[0]
    assert fed["text_states"].shape == (2, 32, 8)
    assert fed["text_mask"][:, :5].all() and not fed["text_mask"][:, 5:].any()
    assert start.shape == (2, 3, 6) and inside.shape == (2, 3, 5) and other.shape == (2, 3)


def test_texts_at_or_above_the_minimum_are_not_padded():
    session = _FakeSession(
        ["text_states", "text_mask", "query_states", "query_mask"],
        lambda f: [np.zeros((1, 1, f["text_states"].shape[1] + 1), np.float32)],
    )
    spec, _ = flatten((torch.zeros(1),))
    meta = {
        "boundary_head_output_spec": spec,
        "min_text_length": 32,
        "boundary_head_length_axes": [[[2, 1]]],
    }

    (start,) = runtime.OnnxBoundaryHead(session, meta, None)(
        torch.ones(1, 40, 8),
        torch.ones(1, 40, dtype=torch.bool),
        torch.ones(1, 1, 8),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert session.feeds[0]["text_states"].shape == (1, 40, 8) and start.shape == (1, 1, 41)


def test_relation_scorer_keeps_the_originals_early_returns():
    session = _FakeSession([], lambda f: pytest.fail("no graph call expected"))
    scorer = runtime.OnnxRelationScorer(session)
    pairs = MagicMock()
    pairs.__len__.return_value = 0

    assert scorer(torch.ones(1, 4, 8), torch.ones(1, 2, 8), None, pairs).shape == (0,)
    pairs.__len__.return_value = 3
    assert scorer(torch.ones(1, 4, 8), torch.ones(1, 0, 8), None, pairs).tolist() == [0.0, 0.0, 0.0]


def test_attribute_scoring_is_refused_rather_than_silently_run_elsewhere():
    head = runtime.OnnxBoundaryHead(
        _FakeSession([], None),
        {"boundary_head_output_spec": {}, "min_text_length": 32, "boundary_head_length_axes": []},
        None,
    )
    with pytest.raises(NotImplementedError, match="not exported"):
        head.score_explicit_spans()
