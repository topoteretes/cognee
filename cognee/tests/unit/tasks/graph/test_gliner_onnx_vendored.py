"""The copied gliner2 code must be exactly what ``vendor_gliner2.py`` produces.

Hand edits to ``_gliner2`` would silently fork upstream. Regenerating into a
temporary directory and comparing byte for byte catches them. Needs gliner2
(the ``gliner`` extra) at the version the copy was made from; skipped otherwise.
"""

from pathlib import Path

import pytest

gliner2 = pytest.importorskip("gliner2")

from cognee.tasks.graph.gliner_demo.onnx import vendor_gliner2  # noqa: E402
from cognee.tasks.graph.gliner_demo.onnx._gliner2 import (  # noqa: E402
    __version__ as vendored_version,
)


def test_the_copy_is_regenerable_byte_for_byte(tmp_path):
    from importlib.metadata import version

    if version("gliner2") != vendored_version:
        pytest.skip(
            f"installed gliner2 {version('gliner2')} differs from the copied {vendored_version}"
        )
    out = vendor_gliner2.vendor(tmp_path / "_gliner2")
    committed = vendor_gliner2.DEFAULT_OUT

    generated = {p.relative_to(out): p.read_bytes() for p in out.rglob("*.py")}
    existing = {p.relative_to(committed): p.read_bytes() for p in committed.rglob("*.py")}
    assert set(generated) == set(existing)
    differing = [str(path) for path in generated if generated[path] != existing[path]]
    assert not differing, f"hand-edited copies: {differing}"
