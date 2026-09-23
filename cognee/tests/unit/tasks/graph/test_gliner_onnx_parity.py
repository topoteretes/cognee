"""Torch vs ONNX Runtime parity on the real GLiNER demo model (opt-in).

Needs the model and an export; run with:

    python -m cognee.tasks.graph.gliner_demo.onnx.export
    GLINER_ONNX_PARITY=1 pytest cognee/tests/unit/tasks/graph/test_gliner_onnx_parity.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("GLINER_ONNX_PARITY") != "1",
    reason="set GLINER_ONNX_PARITY=1 (needs the model and an ONNX export)",
)

TEXTS = [
    "Ada Lovelace programmed.",
    "Google acquired DeepMind in London in 2014. Demis Hassabis co-founded DeepMind in 2010.",
    " ".join(["Lee Sedol lost to AlphaGo, a program built by DeepMind, in Seoul in 2016."] * 60),
]


def _extract(backend):
    from cognee.modules.cognify.config import get_cognify_config
    from cognee.tasks.graph.gliner_demo import extractor as E
    from cognee.tasks.graph.gliner_demo.schema import GlinerSchema

    os.environ["GLINER_BACKEND"] = backend
    get_cognify_config.cache_clear()
    schema = GlinerSchema(
        entity_types=(
            ("person", "a named human"),
            ("organization", "a company"),
            ("location", "a place"),
            ("date", "a year"),
        ),
        relation_types=(
            ("founded", "person founded organization"),
            ("acquired", "organization acquired organization"),
        ),
    )
    return E.extract_batch(E.load_extractor(), TEXTS, schema)


def _mentions(results):
    found = {}
    for i, result in enumerate(results):
        for label, mentions in (result.get("entities") or {}).items():
            for m in mentions:
                found[(i, label, m["text"], m.get("start"), m.get("end"))] = m.get("confidence")
    return found


def test_onnx_extraction_matches_torch():
    try:
        torch_found = _mentions(_extract("torch"))
        onnx_found = _mentions(_extract("onnx"))
    finally:
        from cognee.modules.cognify.config import get_cognify_config

        os.environ.pop("GLINER_BACKEND", None)
        get_cognify_config.cache_clear()
    assert torch_found, "no entities extracted; the parity check would be vacuous"
    assert set(onnx_found) == set(torch_found)
    assert max(abs(onnx_found[k] - torch_found[k]) for k in torch_found) < 1e-3
