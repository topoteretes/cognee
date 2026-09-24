"""Run the GLiNER demo extractor on ONNX Runtime, with no torch installed.

``load_onnx_extractor`` builds gliner2's boundary extractor from the copied,
torch-free gliner2 modules in ``_gliner2`` (tensors are numpy, see
``_tensor``) and three ONNX graphs shipped with cognee in ``graphs/<model>``:
the DeBERTa encoder, the boundary head and the relation scorer. The graphs
carry no weights (about 2 MB together). Their weights are the checkpoint's own
``model.safetensors``, downloaded from Hugging Face at the revision the graphs
were exported against — the file the torch backend downloads too — and read
with numpy. Tokenization, batching, windowing and decoding are gliner2's own
code, so the output matches the torch backend up to float rounding. Running
needs onnxruntime, transformers (tokenizer and config only, no torch),
huggingface_hub and numpy, all core dependencies; only exporting new graphs
needs torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import _tensor, safetensors_numpy
from ._tensor import _t
from .tree import unflatten

GRAPHS = ("encoder", "boundary_head", "relation_scorer")
_GRAPHS_ROOT = Path(__file__).resolve().parent / "graphs"
_VENDORED = "cognee.tasks.graph.gliner_demo.onnx._gliner2"


class OnnxModelNotExportedError(RuntimeError):
    """The ONNX backend was selected for a model cognee ships no graphs for."""

    def __init__(self, model_name: str):
        shipped = sorted(p.parent.name.replace("--", "/") for p in _GRAPHS_ROOT.glob("*/meta.json"))
        super().__init__(
            f"GLINER_BACKEND=onnx has no ONNX graphs for {model_name}; cognee ships graphs for "
            f"{', '.join(shipped) or 'no model'}. Graphs for another model are created, with the "
            f"`gliner` extra (torch), by: python -m cognee.tasks.graph.gliner_demo.onnx.export --model {model_name}"
        )


def graphs_dir(model_name: str) -> Path:
    return _GRAPHS_ROOT / model_name.replace("/", "--")


def weighted_session(template: Path, recipe: dict, checkpoint: dict):
    """An ONNX Runtime session for a weight-free graph, weights taken from ``checkpoint``.

    Returns the session and the arrays backing its weights, which must outlive it.
    """
    import onnxruntime as ort

    names, values, arrays = [], [], []
    for initializer, (tensor, transposed, shape, dtype) in recipe.items():
        if tensor not in checkpoint:
            raise RuntimeError(
                f"model.safetensors has no tensor {tensor} (needed by {template.name})"
            )
        array = checkpoint[tensor].T if transposed else checkpoint[tensor]
        if list(array.shape) != list(shape) or str(array.dtype) != dtype:
            raise RuntimeError(
                f"{tensor} in model.safetensors is {array.dtype}{list(array.shape)}; "
                f"{template.name} expects {dtype}{list(shape)}"
            )
        array = np.ascontiguousarray(array)
        arrays.append(array)
        names.append(initializer)
        values.append(ort.OrtValue.ortvalue_from_numpy(array))
    options = ort.SessionOptions()
    options.add_external_initializers(names, values)
    session = ort.InferenceSession(str(template), options, providers=["CPUExecutionProvider"])
    return session, (arrays, values)


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


def _checkpoint_file(model_name: str, name: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_name, name, revision=revision)


def load_onnx_extractor(model_name: str):
    """Build the torch-free boundary extractor for ``model_name`` from its shipped graphs."""
    from ._gliner2 import __version__ as vendored_version
    from ._gliner2.configuration import BoundaryHeadSettings, ExtractorConfig
    from ._gliner2.models.base import load_extractor_tokenizer
    from ._gliner2.models.boundary.engine import BoundaryExtractor
    from ._gliner2.models.boundary.relations import (
        RelationProposalSettings,
        TypedRelationPairGenerator,
    )
    from ._gliner2.processor import SchemaTransformer

    directory = graphs_dir(model_name)
    meta_path = directory / "meta.json"
    if not meta_path.is_file() or not all((directory / f"{g}.onnx").is_file() for g in GRAPHS):
        raise OnnxModelNotExportedError(model_name)
    meta = json.loads(meta_path.read_text())
    if meta["model"] != model_name:
        raise RuntimeError(f"{directory} holds graphs for {meta['model']}, not {model_name}")
    if meta["gliner2_version"] != vendored_version:
        raise RuntimeError(
            f"{directory} was exported with gliner2 {meta['gliner2_version']}; this cognee runs "
            f"the gliner2 {vendored_version} inference code. Re-export with gliner2 "
            f"{vendored_version}: python -m cognee.tasks.graph.gliner_demo.onnx.export --model {model_name}"
        )
    revision = meta["revision"]

    config = ExtractorConfig.from_pretrained(_checkpoint_file(model_name, "config.json", revision))
    if config.architecture != "boundary":
        raise ValueError(
            f"the ONNX GLiNER path supports the boundary architecture, not {config.architecture!r}"
        )
    settings = BoundaryHeadSettings(**config.boundary_head)
    encoder_config = SimpleNamespace(
        **json.loads(
            Path(_checkpoint_file(model_name, "encoder_config/config.json", revision)).read_text()
        )
    )
    checkpoint = safetensors_numpy.load(_checkpoint_file(model_name, "model.safetensors", revision))
    sessions = {
        graph: weighted_session(directory / f"{graph}.onnx", meta["weights"][graph], checkpoint)
        for graph in GRAPHS
    }

    extractor = BoundaryExtractor.__new__(BoundaryExtractor)
    extractor.config = config
    # gliner2's own loader, as the torch path uses it: this checkpoint's legacy
    # special-token metadata needs its compatibility retry.
    tokenizer_dir = str(
        Path(_checkpoint_file(model_name, "tokenizer_config.json", revision)).parent
    )
    _checkpoint_file(model_name, "tokenizer.json", revision)
    extractor.processor = SchemaTransformer(
        tokenizer=load_extractor_tokenizer(tokenizer_dir), token_pooling=config.token_pooling
    )
    extractor.hidden_size = encoder_config.hidden_size
    extractor.boundary_settings = settings
    extractor.enable_records = settings.enable_records
    extractor.enable_relations = settings.enable_relations
    extractor.encoder = OnnxEncoder(sessions["encoder"][0], encoder_config)
    extractor.boundary_head = OnnxBoundaryHead(sessions["boundary_head"][0], meta, settings)
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
        extractor.relation_scorer = OnnxRelationScorer(sessions["relation_scorer"][0])
    extractor.name_or_path = model_name
    # The weight arrays back the sessions' initializers; keep them alive with it.
    extractor._onnx_weights = [keep for _, keep in sessions.values()]
    return extractor
