"""Run a gliner2 boundary extractor's neural pieces on ONNX Runtime.

``attach_onnx_backend`` swaps the encoder, boundary head and relation scorer of a
loaded extractor for ONNX Runtime sessions built by ``export.py``. gliner2's
tokenization, batching, windowing and decoding run unchanged, so extraction
output matches the torch backend up to float rounding.

The torch encoder weights (~740 MB of the model) are released once swapped; the
remaining torch modules are kept only as the Python glue gliner2 needs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .tree import unflatten

ONNX_DIR_ENV = "GLINER_ONNX_DIR"
GRAPHS = ("encoder.onnx", "boundary_head.onnx", "relation_scorer.onnx")


class OnnxModelNotExportedError(RuntimeError):
    """The ONNX backend was selected but no export exists for this model."""

    def __init__(self, model_name: str, onnx_dir: Path):
        super().__init__(
            f"GLINER_BACKEND=onnx needs an ONNX export of {model_name} in {onnx_dir}. "
            "Create it once with: python -m cognee.tasks.graph.gliner_demo.onnx.export "
            f"--model {model_name}"
        )


def default_onnx_dir(model_name: str) -> Path:
    base = os.getenv(ONNX_DIR_ENV)
    root = Path(base).expanduser() if base else Path.home() / ".cognee" / "models" / "gliner-onnx"
    return root / model_name.replace("/", "--")


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(session, *arrays) -> list[np.ndarray]:
    names = [i.name for i in session.get_inputs()]
    return session.run(None, {n: np.ascontiguousarray(a) for n, a in zip(names, arrays)})


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class _Anchored(torch.nn.Module):
    """gliner2 reads the model's device from ``next(self.parameters())``; with the
    encoder's weights released a tiny parameter keeps that working."""

    def __init__(self):
        super().__init__()
        self._device_anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


class OnnxEncoder(_Anchored):
    def __init__(self, session, config):
        super().__init__()
        self.session = session
        self.config = config

    def forward(self, input_ids, attention_mask, **_):
        (states,) = _run(self.session, _np(input_ids), _np(attention_mask))
        return SimpleNamespace(last_hidden_state=torch.from_numpy(states))


class OnnxBoundaryHead(_Anchored):
    """The boundary head's inference forward; padding covers texts shorter than the export minimum."""

    def __init__(self, session, meta, settings):
        super().__init__()
        self.session = session
        self.spec = meta["boundary_head_output_spec"]
        self.length_axes = meta["boundary_head_length_axes"]
        self.min_length = meta["min_text_length"]
        self.settings = settings

    def forward(
        self,
        token_states,
        text_mask,
        query_states,
        query_mask,
        targets=None,
        *,
        return_candidates=True,
        **_,
    ):
        if targets is not None or not return_candidates:
            raise NotImplementedError("the ONNX boundary head only runs inference with candidates")
        b, length, h = token_states.shape
        pad = max(self.min_length - length, 0)
        states, mask = _np(token_states), _np(text_mask)
        if pad:
            # Masked positions: the head is length-invariant under padding (every
            # batch is already padded to its longest text); outputs are sliced back.
            states = np.concatenate((states, np.zeros((b, pad, h), states.dtype)), 1)
            mask = np.concatenate((mask, np.zeros((b, pad), mask.dtype)), 1)
        outs = _run(self.session, states, mask, _np(query_states), _np(query_mask))
        tensors = []
        for array, axes in zip(outs, self.length_axes):
            if pad:
                for axis, offset in axes:
                    array = np.take(array, np.arange(length + offset), axis=axis)
            tensors.append(torch.from_numpy(np.ascontiguousarray(array)))
        return unflatten(self.spec, tensors, {"batch_size": b})

    def score_explicit_spans(self, *args, **kwargs):
        raise NotImplementedError(
            "entity attributes and choice fields are not exported to ONNX; cognee's "
            "entity + relation schema does not use them"
        )


class OnnxRelationScorer(_Anchored):
    def __init__(self, session):
        super().__init__()
        self.session = session

    def forward(self, boundary_states, relation_query_states, entity_candidates, relation_pairs):
        # The original's early returns, kept outside the graph.
        if len(relation_pairs) == 0:
            return boundary_states.new_zeros(0)
        if (
            min(boundary_states.shape[0], relation_query_states.shape[0]) <= 0
            or relation_query_states.shape[1] <= 0
        ):
            return boundary_states.new_zeros(len(relation_pairs))
        p = relation_pairs
        mask = (
            p.pair_mask
            if p.pair_mask is not None
            else torch.ones_like(p.batch_index, dtype=torch.bool)
        )
        (scores,) = _run(
            self.session,
            _np(boundary_states),
            _np(relation_query_states),
            _np(p.batch_index),
            _np(p.relation_index),
            _np(p.head_start),
            _np(p.head_end),
            _np(p.tail_start),
            _np(p.tail_end),
            _np(mask),
        )
        return torch.from_numpy(scores)


def attach_onnx_backend(extractor, model_name: str, onnx_dir: Path | None = None):
    """Swap ``extractor``'s neural pieces for ONNX Runtime sessions, in place."""
    from importlib.metadata import version

    onnx_dir = onnx_dir or default_onnx_dir(model_name)
    meta_path = onnx_dir / "meta.json"
    if not meta_path.is_file() or not all((onnx_dir / g).is_file() for g in GRAPHS):
        raise OnnxModelNotExportedError(model_name, onnx_dir)
    meta = json.loads(meta_path.read_text())
    if meta["model"] != model_name:
        raise RuntimeError(f"{onnx_dir} holds an export of {meta['model']}, not {model_name}")
    installed = version("gliner2")
    if meta["gliner2_version"] != installed:
        raise RuntimeError(
            f"{onnx_dir} was exported with gliner2 {meta['gliner2_version']} but {installed} is "
            "installed; re-export: python -m cognee.tasks.graph.gliner_demo.onnx.export "
            f"--model {model_name}"
        )
    config = extractor.encoder.config
    settings = extractor.boundary_head.settings
    extractor.encoder = OnnxEncoder(_session(onnx_dir / "encoder.onnx"), config)
    extractor.boundary_head = OnnxBoundaryHead(
        _session(onnx_dir / "boundary_head.onnx"), meta, settings
    )
    extractor.relation_scorer = OnnxRelationScorer(_session(onnx_dir / "relation_scorer.onnx"))
    return extractor
