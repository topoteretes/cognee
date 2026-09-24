"""Export a gliner2 boundary extractor's neural pieces to ONNX (needs torch).

    python -m cognee.tasks.graph.gliner_demo.onnx.export [--model NAME] [--out DIR]

Three graphs are written, the only parts of an extraction that run a network:
the DeBERTa encoder (~99% of inference time), the boundary head (candidate
proposal and scoring) and the sparse relation scorer. Everything else —
tokenization, batching, windowing, decoding — stays gliner2's own Python.

Shapes stay symbolic. Two things in gliner2 pin a dimension to its traced value
under export, and both are rewritten here, export-only and numerically
identical (``_patched`` / ``_TensorRelationScorer``, verified below):
``shift_right_with_eos`` indexes with a Python int of the text length, and the
relation scorer turns the text length and pair count into Python numbers.
The head also branches on ``min(32, L + 1)``, so it is exported for text
lengths >= 32 and the runtime pads shorter inputs with masked positions.

The export verifies itself: every recorded example call is replayed through
ONNX Runtime and compared with torch; any mismatch aborts before files are kept.

What is kept is graphs without weights. Every weight in the three graphs is a
tensor of the checkpoint's own ``model.safetensors`` at a pinned Hugging Face
revision, as-is or transposed, so the runtime reads the weights from the file
users download from Hugging Face anyway (the torch backend downloads the same
file) and the shipped graphs are about 2 MB. The weight-free graphs are
checked bit-identical to the full ones before they are written.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .tree import flatten

MIN_TEXT_LENGTH = 32
FLOAT_TOLERANCE = 1e-3
OPSET = 18

SAMPLE_TEXTS = [
    (
        "Ada Lovelace worked with Charles Babbage on the Analytical Engine in London. "
        "She published notes in 1843 describing an algorithm for the machine, and she is often "
        "called the first computer programmer. Babbage was a professor at the University of "
        "Cambridge, where he designed the Difference Engine before the Analytical Engine."
    ),
    (
        "In 2016 DeepMind, a subsidiary of Google based in London, built AlphaGo, a program that "
        "defeated Lee Sedol in Seoul. Demis Hassabis co-founded DeepMind in 2010 and later led "
        "research on protein folding with AlphaFold, which the company released in 2020 for "
        "scientists at universities and laboratories around the world."
    ),
    (
        "Microsoft invested in OpenAI in San Francisco, and OpenAI released ChatGPT in November "
        "2022. Sam Altman leads OpenAI, while Satya Nadella is the chief executive of Microsoft in "
        "Redmond. The partnership gave Azure customers access to large language models."
    ),
]
SAMPLE_ENTITIES = {
    "person": "a named human",
    "organization": "a company or institution",
    "location": "a city or country",
    "date": "a year or calendar date",
    "product": "a named software, machine or model",
}
SAMPLE_RELATIONS = {
    "works_for": "person employed by an organization",
    "founded": "person founded organization",
    "located_in": "entity located in a place",
    "created": "organization or person created product",
}


# ---------------------------------------------------------------- export-only rewrites


def _shift_right_with_eos(text_states, text_lengths, eos_state):
    """``encoding.shift_right_with_eos`` without indexing by a Python int of L."""
    b, l, h = text_states.shape
    eos = eos_state.to(text_states.dtype).view(1, 1, h).expand(b, 1, h)
    out = torch.cat((text_states, eos), dim=1)
    pos = text_lengths.clamp(max=l).view(b, 1, 1).expand(b, 1, h)
    return out.scatter(1, pos, eos)


@contextlib.contextmanager
def _patched():
    """Install the rewrites in every gliner2 module that imported the originals."""
    from gliner2.models.boundary import encoding

    replacements = [(encoding, "shift_right_with_eos", _shift_right_with_eos)]
    rebound = []
    for module, name, new in replacements:
        old = getattr(module, name)
        for mod in list(sys.modules.values()):
            if (
                getattr(mod, "__name__", "").startswith("gliner2")
                and getattr(mod, name, None) is old
            ):
                rebound.append((mod, name, old))
                setattr(mod, name, new)
    try:
        yield
    finally:
        for mod, name, old in rebound:
            setattr(mod, name, old)


@contextlib.contextmanager
def _topk_sorts():
    """Legacy exporter only: drop ``stable=True`` (it rejects it); ONNX TopK orders
    equal values by lower index, which is the stable-sort guarantee."""
    saved = (torch.sort, torch.argsort, torch.Tensor.sort, torch.Tensor.argsort)

    def strip(fn):
        def inner(*a, **k):
            k.pop("stable", None)
            return fn(*a, **k)

        return inner

    torch.sort, torch.argsort = strip(saved[0]), strip(saved[1])
    torch.Tensor.sort, torch.Tensor.argsort = strip(saved[2]), strip(saved[3])
    try:
        yield
    finally:
        torch.sort, torch.argsort, torch.Tensor.sort, torch.Tensor.argsort = saved


def _sort_stable_translation():
    """``aten.sort.stable`` for the torch.export exporter: a full-length TopK."""
    from onnxscript import opset18 as op

    def aten_sort_stable(self, stable: bool = False, dim: int = -1, descending: bool = False):
        k = op.Shape(self, start=dim, end=dim + 1) if dim != -1 else op.Shape(self, start=-1)
        values, indices = op.TopK(self, k, axis=dim, largest=descending, sorted=True)
        return values, indices

    return {torch.ops.aten.sort.stable: aten_sort_stable}


# ---------------------------------------------------------------- graph wrappers


class _Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class _Head(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, text_states, text_mask, query_states, query_mask):
        out = self.head(text_states, text_mask, query_states, query_mask, return_candidates=True)
        return tuple(flatten(out)[1])


class _TensorRelationScorer(torch.nn.Module):
    """``SparseRelationScorer.forward`` for P >= 1 pairs with tensor-only shape use.

    Same weights and maths as the original, which turns the text length and the
    pair count into Python numbers (pinning them under export). The runtime keeps
    the original's P == 0 / no-relation early returns.
    """

    def __init__(self, scorer):
        super().__init__()
        self.m = scorer

    def forward(
        self,
        boundary_states,
        relation_query_states,
        batch_index,
        relation_index,
        head_start,
        head_end,
        tail_start,
        tail_end,
        pair_mask,
    ):
        m = self.m
        batch_count = torch.minimum(
            torch.tensor(boundary_states.shape[0]), torch.tensor(relation_query_states.shape[0])
        )
        b = torch.minimum(batch_index.clamp(min=0), batch_count - 1)
        relation_count = relation_query_states.shape[1]
        rel_idx = torch.minimum(relation_index.clamp(min=0), torch.tensor(relation_count) - 1)
        pair_valid = (
            (batch_index >= 0)
            & (batch_index < batch_count)
            & (relation_index >= 0)
            & (relation_index < relation_count)
            & pair_mask
        )
        length = torch.tensor(boundary_states.shape[1])

        def gather(pos):
            return boundary_states[b, torch.minimum(pos.clamp(min=0), (length - 1).clamp(min=0))]

        rel = relation_query_states[b, rel_idx]
        delta = (tail_start - head_start).to(boundary_states.dtype)
        feats = torch.cat(
            [
                gather(head_start),
                gather(head_end - 1),
                gather(tail_start),
                gather(tail_end - 1),
                rel,
                torch.sign(delta).unsqueeze(-1),
                (delta.abs() / length.clamp(min=1).to(boundary_states.dtype)).unsqueeze(-1),
            ],
            dim=-1,
        )
        score = m.mlp(feats).squeeze(-1)
        if m.use_biaffine_content:
            prefix = torch.cat(
                (
                    boundary_states.new_zeros(boundary_states.shape[0], 1, m.hidden_size),
                    boundary_states.float().cumsum(1).to(boundary_states.dtype),
                ),
                dim=1,
            )

            def pool(start, end):
                span = (
                    prefix[b, torch.minimum(end.clamp(min=0), length)]
                    - prefix[b, torch.minimum(start.clamp(min=0), length)]
                )
                return span / (end - start).clamp_min(1).unsqueeze(-1).to(span.dtype)

            head_content = m.head_content_projection(pool(head_start, head_end))
            tail_content = m.tail_content_projection(pool(tail_start, tail_end))
            gate = torch.sigmoid(m.relation_content_gate(rel))
            score = (
                score
                + (head_content * gate * tail_content).sum(-1) / (m.hidden_size**0.5)
                + m.content_linear(torch.cat((head_content, tail_content, rel), dim=-1)).squeeze(-1)
            )
        return score.masked_fill(~pair_valid, 0.0)


# ---------------------------------------------------------------- example calls


def _record_calls(extractor) -> dict[str, list]:
    """Run a real extraction and keep every call's inputs and torch outputs."""
    calls = {"encoder": [], "boundary_head": [], "relation_scorer": []}

    def recorder(name, fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            calls[name].append((args, kwargs, out))
            return out

        return wrapped

    originals = {
        "encoder": extractor.encoder.forward,
        "boundary_head": extractor.boundary_head.forward,
        "relation_scorer": extractor.relation_scorer.forward,
    }
    for name, fn in originals.items():
        getattr(extractor, name).forward = recorder(name, fn)
    try:
        schema = extractor.create_schema().entities(SAMPLE_ENTITIES).relations(SAMPLE_RELATIONS)
        extractor.batch_extract(SAMPLE_TEXTS, schema, batch_size=len(SAMPLE_TEXTS), threshold=0.3)
    finally:
        for name, fn in originals.items():
            getattr(extractor, name).forward = fn
    return calls


def _pick(calls, name, ok):
    for call in calls[name]:
        if ok(*call):
            return call
    raise RuntimeError(f"no usable example call recorded for {name}")


# ---------------------------------------------------------------- export + verify


def _relation_inputs(args):
    states, rel_states, _candidates, pairs = args[:4]
    mask = (
        pairs.pair_mask
        if pairs.pair_mask is not None
        else torch.ones_like(pairs.batch_index, dtype=torch.bool)
    )
    return (
        states,
        rel_states,
        pairs.batch_index,
        pairs.relation_index,
        pairs.head_start,
        pairs.head_end,
        pairs.tail_start,
        pairs.tail_end,
        mask,
    )


def _verify(path: Path, inputs, expected, session=None) -> float:
    import onnxruntime as ort

    session = session or ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {i.name: x.numpy() for i, x in zip(session.get_inputs(), inputs)}
    got = session.run(None, feeds)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    worst = 0.0
    for index, (g, e) in enumerate(zip(got, expected)):
        e = e.detach().numpy()
        if g.shape != e.shape:
            raise RuntimeError(f"{path.name} out{index}: shape {g.shape} != torch {e.shape}")
        if e.dtype.kind in "biu":
            if (g != e).any():
                raise RuntimeError(
                    f"{path.name} out{index}: {(g != e).sum()} index/mask mismatches"
                )
            continue
        finite = np.isfinite(e)
        if (np.isfinite(g) != finite).any():
            raise RuntimeError(f"{path.name} out{index}: finite/inf pattern differs from torch")
        if finite.any():
            worst = max(worst, float(np.abs(g[finite] - e[finite]).max()))
    if worst > FLOAT_TOLERANCE:
        raise RuntimeError(f"{path.name}: max abs diff {worst:.2e} exceeds {FLOAT_TOLERANCE}")
    return worst


def _length_axes(head, example) -> list[list[list[int]]]:
    """Which output axes follow the text length L, and their offset (L or L + 1).

    Found by running the head at L and L + 5; the runtime slices padded outputs
    back to the caller's length along exactly these axes.
    """
    text_states, text_mask, query_states, query_mask = example
    extra = 5
    padded = (
        torch.cat(
            (text_states, text_states.new_zeros(text_states.shape[0], extra, text_states.shape[2])),
            1,
        ),
        torch.cat((text_mask, text_mask.new_zeros(text_mask.shape[0], extra)), 1),
        query_states,
        query_mask,
    )
    a, b = _Head(head)(*example), _Head(head)(*padded)
    length = text_states.shape[1]
    axes = []
    for ta, tb in zip(a, b):
        entry = []
        for axis, (sa, sb) in enumerate(zip(ta.shape, tb.shape)):
            if sb - sa == extra:
                entry.append([axis, sa - length])
        axes.append(entry)
    return axes


# Weights not found in the checkpoint stay inside the graph. Only small
# constants (shape vectors, a reshaped EOS state) may; anything bigger means a
# weight was fused or rewritten and the checkpoint cannot supply it.
MAX_EMBEDDED_BYTES = 64 * 1024


def _strip_weights(graph_path: Path, checkpoint: dict, out_path: Path) -> dict:
    """Write ``graph_path`` without the weights ``checkpoint`` holds; return the recipe.

    The recipe maps each stripped initializer to ``[checkpoint tensor, transposed,
    shape, dtype]``; the runtime checks shape and dtype before using a tensor.
    """
    import hashlib

    import onnx
    from onnx import TensorProto, numpy_helper

    def key(array):
        # The array's own shape: np.ascontiguousarray turns a 0-d scalar into a
        # 1-element array, which would match a [1]-shaped checkpoint tensor the
        # graph cannot take in place of its scalar.
        return (
            hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest(),
            str(array.dtype),
            tuple(array.shape),
        )

    as_is = {key(v): name for name, v in checkpoint.items()}
    transposed = {key(v.T): name for name, v in checkpoint.items() if v.ndim == 2}
    model = onnx.load(str(graph_path))
    recipe, embedded = {}, 0
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        k = key(array)
        if k in as_is:
            recipe[init.name] = [as_is[k], False, list(array.shape), str(array.dtype)]
        elif array.ndim == 2 and k in transposed:
            recipe[init.name] = [transposed[k], True, list(array.shape), str(array.dtype)]
        else:
            embedded += array.nbytes
            continue
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        entry = init.external_data.add()
        entry.key, entry.value = "location", "supplied-from-model.safetensors"
    if embedded > MAX_EMBEDDED_BYTES:
        raise RuntimeError(
            f"{graph_path.name}: {embedded} bytes of weights are not tensors of the checkpoint; "
            "the weight-free graph would not reproduce it"
        )
    onnx.save(model, str(out_path))
    return recipe


def export(model_name: str, out_dir: Path) -> Path:
    from importlib.metadata import version

    from gliner2 import AutoExtractor
    from huggingface_hub import hf_hub_download, model_info
    from torch.export import Dim

    from . import safetensors_numpy
    from .runtime import weighted_session

    # Pin the checkpoint: the shipped graphs name its tensors, so the runtime must
    # read exactly this revision's model.safetensors.
    revision = model_info(model_name).sha
    extractor = AutoExtractor.from_pretrained(model_name, revision=revision)
    extractor.eval()
    torch.set_grad_enabled(False)
    calls = _record_calls(extractor)

    _, enc_kwargs, enc_out = _pick(calls, "encoder", lambda a, k, o: k["input_ids"].shape[0] > 1)
    head_args, _, head_out = _pick(
        calls,
        "boundary_head",
        lambda a, k, o: (
            a[0].shape[0] > 1 and a[0].shape[1] >= MIN_TEXT_LENGTH and a[2].shape[1] > 1
        ),
    )
    rel_args, _, _ = _pick(
        calls, "relation_scorer", lambda a, k, o: len(a[3]) > 1 and a[1].shape[1] > 1
    )

    tmp = Path(tempfile.mkdtemp(prefix="gliner-onnx-"))
    try:
        enc_in = (enc_kwargs["input_ids"], enc_kwargs["attention_mask"])
        with _topk_sorts():
            torch.onnx.export(
                _Encoder(extractor.encoder),
                enc_in,
                str(tmp / "encoder.onnx"),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={n: {0: "b", 1: "l"} for n in ("input_ids", "attention_mask")},
                opset_version=OPSET,
                dynamo=False,
                do_constant_folding=True,
            )

        b, l, q = (
            Dim("b", min=2, max=256),
            Dim("l", min=MIN_TEXT_LENGTH, max=16384),
            Dim("q", min=2, max=1024),
        )
        head_in = tuple(head_args[:4])
        head_spec, head_tensors = flatten(head_out)
        with _patched():
            program = torch.onnx.export(
                _Head(extractor.boundary_head),
                head_in,
                dynamo=True,
                dynamic_shapes=({0: b, 1: l}, {0: b, 1: l}, {0: b, 1: q}, {0: b, 1: q}),
                input_names=["text_states", "text_mask", "query_states", "query_mask"],
                output_names=[f"out{i}" for i in range(len(head_tensors))],
                custom_translation_table=_sort_stable_translation(),
                optimize=True,
            )
        program.save(str(tmp / "boundary_head.onnx"))

        rl, r, p = (
            Dim("l", min=2, max=16384),
            Dim("r", min=2, max=1024),
            Dim("p", min=2, max=1 << 20),
        )
        rel_in = _relation_inputs(rel_args)
        program = torch.onnx.export(
            _TensorRelationScorer(extractor.relation_scorer),
            rel_in,
            dynamo=True,
            # One sample per call: the engine scores relations sample by sample.
            dynamic_shapes=({1: rl}, {1: r}, *({0: p},) * 7),
            input_names=[
                "boundary_states",
                "relation_query_states",
                "batch_index",
                "relation_index",
                "head_start",
                "head_end",
                "tail_start",
                "tail_end",
                "pair_mask",
            ],
            output_names=["scores"],
            optimize=True,
        )
        program.save(str(tmp / "relation_scorer.onnx"))

        report = {
            "encoder": [
                _verify(tmp / "encoder.onnx", enc_in, enc_out.last_hidden_state)
                for _, kw, o in [(None, enc_kwargs, enc_out)]
            ],
            "boundary_head": [
                _verify(tmp / "boundary_head.onnx", tuple(a[:4]), flatten(o)[1])
                for a, _, o in calls["boundary_head"]
                if a[0].shape[1] >= MIN_TEXT_LENGTH
            ],
            "relation_scorer": [
                _verify(tmp / "relation_scorer.onnx", _relation_inputs(a), o)
                for a, _, o in calls["relation_scorer"]
                if len(a[3])
            ],
        }
        checkpoint = safetensors_numpy.load(
            hf_hub_download(model_name, "model.safetensors", revision=revision)
        )
        templates = tmp / "templates"
        templates.mkdir()
        recipes = {}
        for graph in ("encoder", "boundary_head", "relation_scorer"):
            recipes[graph] = _strip_weights(
                tmp / f"{graph}.onnx", checkpoint, templates / f"{graph}.onnx"
            )

        def full_and_template(graph):
            import onnxruntime as ort

            full = ort.InferenceSession(
                str(tmp / f"{graph}.onnx"), providers=["CPUExecutionProvider"]
            )
            template, _ = weighted_session(templates / f"{graph}.onnx", recipes[graph], checkpoint)
            return full, template

        def identical(graph, inputs):
            full, template = full_and_template(graph)
            feeds = {i.name: x.numpy() for i, x in zip(full.get_inputs(), inputs)}
            return all(
                np.array_equal(a, b)
                for a, b in zip(full.run(None, feeds), template.run(None, feeds))
            )

        checks = (
            [("encoder", enc_in)]
            + [
                ("boundary_head", tuple(a[:4]))
                for a, _, _ in calls["boundary_head"]
                if a[0].shape[1] >= MIN_TEXT_LENGTH
            ]
            + [
                ("relation_scorer", _relation_inputs(a))
                for a, _, _ in calls["relation_scorer"]
                if len(a[3])
            ]
        )
        for graph, inputs in checks:
            if not identical(graph, inputs):
                raise RuntimeError(
                    f"{graph}: the weight-free graph does not reproduce the full export"
                )

        meta = {
            "model": model_name,
            "revision": revision,
            "gliner2_version": version("gliner2"),
            "architecture": "boundary",
            "opset": OPSET,
            "min_text_length": MIN_TEXT_LENGTH,
            "boundary_head_output_spec": head_spec,
            "boundary_head_length_axes": _length_axes(extractor.boundary_head, head_in),
            "verification_max_abs_diff": {k: max(v) if v else None for k, v in report.items()},
        }
        meta["weights"] = recipes
        (templates / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in templates.iterdir():
            shutil.move(str(item), str(out_dir / item.name))
        return out_dir
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    from ..extractor import DEFAULT_MODEL
    from .runtime import graphs_dir

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = export(args.model, args.out or graphs_dir(args.model))
    meta = json.loads((out / "meta.json").read_text())
    print(f"exported {args.model} at revision {meta['revision']} to {out}")
    print("verification (max abs diff vs torch):", meta["verification_max_abs_diff"])


if __name__ == "__main__":
    main()
