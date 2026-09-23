"""The GLiNER demo extractor's torch-free ONNX Runtime backend, without the real model.

The neural pieces are exercised against fake ONNX sessions here. Real
torch-vs-ONNX parity on the downloaded model is ``test_gliner_onnx_parity.py``
(opt-in, needs an export).
"""

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import cognee.tasks.graph.gliner_demo.extractor as extractor_module
from cognee.modules.cognify.config import get_cognify_config
from cognee.tasks.graph.gliner_demo.onnx import _tensor as tt
from cognee.tasks.graph.gliner_demo.onnx import runtime
from cognee.tasks.graph.gliner_demo.onnx.tree import unflatten

MODEL = "fastino/gliner2.5-base-v1"


@pytest.fixture
def backend_env(monkeypatch):
    def set_backend(value):
        monkeypatch.setenv("GLINER_BACKEND", value)
        get_cognify_config.cache_clear()

    yield set_backend
    get_cognify_config.cache_clear()


# ---------------------------------------------------------------- no torch


def test_the_onnx_path_imports_without_torch_transformers_or_gliner2():
    """Run in a fresh interpreter with those packages made unimportable."""
    script = textwrap.dedent(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("torch", "transformers", "gliner2"):
                    raise ImportError(name)
        sys.meta_path.insert(0, Block())
        import cognee.tasks.graph.gliner_demo.onnx.runtime
        import cognee.tasks.graph.gliner_demo.onnx._gliner2.models.boundary.engine
        import cognee.tasks.graph.gliner_demo.onnx._gliner2.processor
        loaded = [m for m in ("torch", "transformers", "gliner2") if m in sys.modules]
        print("LOADED:" + ",".join(loaded))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "LOADED:\n" in result.stdout or result.stdout.strip().endswith("LOADED:")


def test_onnx_backend_does_not_require_gliner2(backend_env, monkeypatch):
    backend_env("onnx")
    monkeypatch.setattr(extractor_module.importlib.util, "find_spec", lambda name: None)

    extractor_module.require_gliner2()  # no GlinerNotInstalledError
    assert extractor_module.gliner_runtime_installed()


def test_torch_backend_still_requires_gliner2(backend_env, monkeypatch):
    backend_env("torch")
    monkeypatch.setattr(extractor_module.importlib.util, "find_spec", lambda name: None)

    assert not extractor_module.gliner_runtime_installed()


def test_unknown_backend_is_an_error(backend_env):
    backend_env("tensorrt")
    with pytest.raises(ValueError, match="Unknown GLINER_BACKEND"):
        extractor_module.resolve_gliner_backend()


# ---------------------------------------------------------------- export checks


def _export_dir(tmp_path, **meta_overrides):
    out = tmp_path / MODEL.replace("/", "--")
    out.mkdir(parents=True)
    for graph in runtime.GRAPHS:
        (out / graph).write_bytes(b"")
    from cognee.tasks.graph.gliner_demo.onnx._gliner2 import __version__

    (out / "meta.json").write_text(
        json.dumps({"model": MODEL, "gliner2_version": __version__} | meta_overrides)
    )
    return out


def test_missing_export_says_how_to_create_one(tmp_path):
    with pytest.raises(runtime.OnnxModelNotExportedError, match="onnx.export --model"):
        runtime.load_onnx_extractor(MODEL, tmp_path / "nothing-here")


def test_an_export_of_another_model_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="holds an export of someone/else"):
        runtime.load_onnx_extractor(MODEL, _export_dir(tmp_path, model="someone/else"))


def test_an_export_from_another_gliner2_version_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="exported with gliner2 1.9.0"):
        runtime.load_onnx_extractor(MODEL, _export_dir(tmp_path, gliner2_version="1.9.0"))


# ---------------------------------------------------------------- ONNX stand-ins


class _FakeSession:
    def __init__(self, names, respond):
        self._names, self._respond, self.feeds = names, respond, []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self._names]

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        return self._respond(feeds)


def _head(respond, axes):
    session = _FakeSession(["text_states", "text_mask", "query_states", "query_mask"], respond)
    spec = {"tu": [{"t": i} for i in range(len(axes))]}
    meta = {
        "boundary_head_output_spec": spec,
        "min_text_length": 32,
        "boundary_head_length_axes": axes,
    }
    return runtime.OnnxBoundaryHead(session, meta, settings=None), session


def test_short_texts_are_padded_to_the_export_minimum_and_sliced_back():
    def respond(feeds):
        b, lp, _ = feeds["text_states"].shape
        q = feeds["query_states"].shape[1]
        return [
            np.zeros((b, q, lp + 1), np.float32),
            np.zeros((b, q, lp), np.float32),
            np.zeros((b, q), np.float32),
        ]

    head, session = _head(respond, [[[2, 1]], [[2, 0]], []])
    start, inside, other = head(
        tt.ones(2, 5, 8),
        tt.ones(2, 5, dtype=tt.bool),
        tt.ones(2, 3, 8),
        tt.ones(2, 3, dtype=tt.bool),
    )

    fed = session.feeds[0]
    assert fed["text_states"].shape == (2, 32, 8)
    assert fed["text_mask"][:, :5].all() and not fed["text_mask"][:, 5:].any()
    assert start.shape == (2, 3, 6) and inside.shape == (2, 3, 5) and other.shape == (2, 3)
    assert isinstance(start, tt.Tensor)


def test_texts_at_or_above_the_minimum_are_not_padded():
    head, session = _head(
        lambda f: [np.zeros((1, 1, f["text_states"].shape[1] + 1), np.float32)], [[[2, 1]]]
    )
    (start,) = head(
        tt.ones(1, 40, 8),
        tt.ones(1, 40, dtype=tt.bool),
        tt.ones(1, 1, 8),
        tt.ones(1, 1, dtype=tt.bool),
    )

    assert session.feeds[0]["text_states"].shape == (1, 40, 8) and start.shape == (1, 1, 41)


def test_relation_scorer_keeps_the_originals_early_returns():
    scorer = runtime.OnnxRelationScorer(
        _FakeSession([], lambda f: pytest.fail("no graph call expected"))
    )
    pairs = MagicMock()
    pairs.__len__.return_value = 0
    assert scorer(tt.ones(1, 4, 8), tt.ones(1, 2, 8), None, pairs).shape == (0,)
    pairs.__len__.return_value = 3
    assert scorer(tt.ones(1, 4, 8), tt.ones(1, 0, 8), None, pairs).tolist() == [0.0, 0.0, 0.0]


def test_attribute_scoring_and_records_are_refused_not_silently_run():
    head, _ = _head(lambda f: [], [])
    with pytest.raises(NotImplementedError, match="not exported"):
        head.score_explicit_spans()
    with pytest.raises(NotImplementedError, match="record extraction"):
        runtime._NoRecordDecoder()()


def test_export_specs_resolve_to_the_torch_free_copies():
    spec = {
        "dc": "gliner2.models.outputs:ExtractorOutput",
        "f": {"x": {"dc": "gliner2.models.outputs:CandidateTensorBatch"}},
    }
    assert runtime._vendored_spec(spec)["dc"].startswith(
        "cognee.tasks.graph.gliner_demo.onnx._gliner2."
    )
    assert runtime._vendored_spec(spec)["f"]["x"]["dc"].startswith(
        "cognee.tasks.graph.gliner_demo.onnx._gliner2."
    )


def test_unflatten_fills_symbolic_fields():
    from dataclasses import dataclass

    @dataclass
    class Out:
        scores: object
        batch_size: int = 0

    spec = {"dc": f"{__name__}:Out", "f": {"scores": {"t": 0}, "batch_size": {"sym": True}}}
    globals()["Out"] = Out
    rebuilt = unflatten(spec, [tt.zeros(2)], {"batch_size": 7})
    assert rebuilt.batch_size == 7 and rebuilt.scores.shape == (2,)
