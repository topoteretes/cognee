"""Run the GLiNER demo extractor on ONNX Runtime, with no torch installed.

``load_onnx_extractor`` builds gliner2's boundary extractor from the copied,
torch-free gliner2 modules in ``_gliner2`` (tensors are numpy, see
``_tensor``), the checkpoint's ``config.json`` and ``tokenizer.json``, and the
three ONNX graphs ``export.py`` writes: the DeBERTa encoder, the boundary head
and the relation scorer. Tokenization, batching, windowing and decoding are
gliner2's own code, so the output matches the torch backend up to float
rounding. Exporting needs torch once; running needs onnxruntime, tokenizers
and numpy, which cognee already depends on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import _tensor
from ._tensor import _t
from .tree import unflatten

ONNX_DIR_ENV = "GLINER_ONNX_DIR"
GRAPHS = ("encoder.onnx", "boundary_head.onnx", "relation_scorer.onnx")
_VENDORED = "cognee.tasks.graph.gliner_demo.onnx._gliner2"


class OnnxModelNotExportedError(RuntimeError):
    """The ONNX backend was selected but no export exists for this model."""

    def __init__(self, model_name: str, onnx_dir: Path):
        super().__init__(
            f"GLINER_BACKEND=onnx needs an ONNX export of {model_name} in {onnx_dir}. "
            "Create it once, on a machine with the `gliner` extra (torch), with: "
            f"python -m cognee.tasks.graph.gliner_demo.onnx.export --model {model_name}"
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
    return session.run(
        None, {n: np.ascontiguousarray(np.asarray(a)) for n, a in zip(names, arrays)}
    )


def _vendored_spec(spec):
    """Export specs name gliner2's classes; the torch-free path uses the copies."""
    if isinstance(spec, dict):
        return {
            k: (v.replace("gliner2.", f"{_VENDORED}.", 1) if k == "dc" else _vendored_spec(v))
            for k, v in spec.items()
        }
    if isinstance(spec, list):
        return [_vendored_spec(v) for v in spec]
    return spec


class OnnxEncoder:
    def __init__(self, session, config):
        self.session = session
        self.config = config

    def __call__(self, input_ids, attention_mask, **_):
        (states,) = _run(self.session, input_ids, attention_mask)
        return SimpleNamespace(last_hidden_state=_t(states))


class OnnxBoundaryHead:
    """The head's inference forward; texts below the export minimum are padded."""

    def __init__(self, session, meta, settings):
        self.session = session
        self.spec = _vendored_spec(meta["boundary_head_output_spec"])
        self.length_axes = meta["boundary_head_length_axes"]
        self.min_length = meta["min_text_length"]
        self.settings = settings

    def __call__(
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
        states, mask = np.asarray(token_states), np.asarray(text_mask)
        b, length, h = states.shape
        pad = max(self.min_length - length, 0)
        if pad:
            # Masked positions: the head is invariant to padding (every batch is
            # already padded to its longest text); outputs are sliced back.
            states = np.concatenate((states, np.zeros((b, pad, h), states.dtype)), 1)
            mask = np.concatenate((mask, np.zeros((b, pad), mask.dtype)), 1)
        outs = _run(self.session, states, mask, query_states, query_mask)
        tensors = []
        for array, axes in zip(outs, self.length_axes):
            if pad:
                for axis, offset in axes:
                    array = np.take(array, np.arange(length + offset), axis=axis)
            tensors.append(_t(np.ascontiguousarray(array)))
        return unflatten(self.spec, tensors, {"batch_size": b})

    def score_explicit_spans(self, *args, **kwargs):
        raise NotImplementedError(
            "entity attributes and choice fields are not exported to ONNX; cognee's "
            "entity + relation schema does not use them"
        )


class OnnxRelationScorer:
    def __init__(self, session):
        self.session = session

    def __call__(self, boundary_states, relation_query_states, entity_candidates, relation_pairs):
        # The original's early returns, kept outside the graph.
        if len(relation_pairs) == 0:
            return _tensor.zeros(0)
        if (
            min(boundary_states.shape[0], relation_query_states.shape[0]) <= 0
            or relation_query_states.shape[1] <= 0
        ):
            return _tensor.zeros(len(relation_pairs))
        p = relation_pairs
        mask = (
            p.pair_mask
            if p.pair_mask is not None
            else np.ones(np.asarray(p.batch_index).shape, bool)
        )
        (scores,) = _run(
            self.session,
            boundary_states,
            relation_query_states,
            p.batch_index,
            p.relation_index,
            p.head_start,
            p.head_end,
            p.tail_start,
            p.tail_end,
            mask,
        )
        return _t(scores)


class _NoRecordDecoder:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("record extraction is not supported by the ONNX GLiNER path")


def _checkpoint_file(model_name: str, name: str) -> str:
    if os.path.isdir(model_name):
        return str(Path(model_name) / name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_name, name)


def load_onnx_extractor(model_name: str, onnx_dir: Path | None = None):
    """Build the torch-free boundary extractor for ``model_name`` from its ONNX export."""
    from ._gliner2 import __version__ as vendored_version
    from ._gliner2.configuration import BoundaryHeadSettings, ExtractorConfig
    from ._gliner2.models.boundary.engine import BoundaryExtractor
    from ._gliner2.models.boundary.relations import (
        RelationProposalSettings,
        TypedRelationPairGenerator,
    )
    from ._gliner2.processor import SchemaTransformer
    from ._hf import AutoTokenizer

    onnx_dir = onnx_dir or default_onnx_dir(model_name)
    meta_path = onnx_dir / "meta.json"
    if not meta_path.is_file() or not all((onnx_dir / g).is_file() for g in GRAPHS):
        raise OnnxModelNotExportedError(model_name, onnx_dir)
    meta = json.loads(meta_path.read_text())
    if meta["model"] != model_name:
        raise RuntimeError(f"{onnx_dir} holds an export of {meta['model']}, not {model_name}")
    if meta["gliner2_version"] != vendored_version:
        raise RuntimeError(
            f"{onnx_dir} was exported with gliner2 {meta['gliner2_version']}; this cognee runs "
            f"the gliner2 {vendored_version} inference code. Re-export with gliner2 "
            f"{vendored_version}: python -m cognee.tasks.graph.gliner_demo.onnx.export --model {model_name}"
        )

    config = ExtractorConfig.from_pretrained(_checkpoint_file(model_name, "config.json"))
    if config.architecture != "boundary":
        raise ValueError(
            f"the ONNX GLiNER path supports the boundary architecture, not {config.architecture!r}"
        )
    settings = BoundaryHeadSettings(**config.boundary_head)
    encoder_config = SimpleNamespace(
        **json.loads(Path(_checkpoint_file(model_name, "encoder_config/config.json")).read_text())
    )

    extractor = BoundaryExtractor.__new__(BoundaryExtractor)
    extractor.config = config
    extractor.processor = SchemaTransformer(
        tokenizer=AutoTokenizer.from_pretrained(model_name), token_pooling=config.token_pooling
    )
    extractor.hidden_size = encoder_config.hidden_size
    extractor.boundary_settings = settings
    extractor.enable_records = settings.enable_records
    extractor.enable_relations = settings.enable_relations
    extractor.encoder = OnnxEncoder(_session(onnx_dir / "encoder.onnx"), encoder_config)
    extractor.boundary_head = OnnxBoundaryHead(
        _session(onnx_dir / "boundary_head.onnx"), meta, settings
    )
    extractor.record_decoder = _NoRecordDecoder()
    if settings.enable_relations:
        extractor.relation_pair_generator = TypedRelationPairGenerator(
            RelationProposalSettings(
                heads_per_relation=settings.relation_heads_per_type,
                tails_per_relation=settings.relation_tails_per_type,
                pair_cap=settings.relation_pair_cap,
                argument_threshold=settings.relation_argument_proposal_threshold,
            )
        )
        extractor.relation_scorer = OnnxRelationScorer(_session(onnx_dir / "relation_scorer.onnx"))
    extractor.name_or_path = model_name
    return extractor
