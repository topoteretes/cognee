"""Drift guard between PipelineContext and the COGX import ctx adaptation.

The COGX import path adapts the pipeline context before handing it to
``add_data_points`` (``_provenance_ctx`` in ``cognee/modules/migration/loader.py``).
A hand-copied field list there silently drops any parameter later added to
``PipelineContext`` — exactly how ``ctx.pipeline_run_id`` was lost, crashing
every preserve-mode import with AttributeError because ``add_data_points``
reads it unconditionally.

These tests introspect ``dataclasses.fields(PipelineContext)``, so they fail
for ANY field that does not survive the adaptation — adding a new parameter
to PipelineContext without carrying it through the cogx import path breaks
them by construction. Pure: no databases, no LLM calls, no network.
"""

import dataclasses
from types import SimpleNamespace
from uuid import uuid4

from cognee.modules.migration.loader import _provenance_ctx
from cognee.modules.pipelines.models.PipelineContext import PipelineContext


def _full_context() -> PipelineContext:
    """A PipelineContext with every field set to a distinct sentinel.

    ``data_item`` is DataItem-shaped (has ``data_id``, no ``id``) so
    ``_provenance_ctx`` takes its substitution path instead of returning the
    context unchanged. Sentinels are per-field strings, so a swapped or
    defaulted field cannot pass equality by accident.
    """
    ctx = PipelineContext()
    for context_field in dataclasses.fields(PipelineContext):
        setattr(ctx, context_field.name, f"sentinel:{context_field.name}")
    ctx.data_item = SimpleNamespace(data_id=uuid4())
    return ctx


def test_every_pipeline_context_field_survives_adaptation():
    ctx = _full_context()

    adapted = _provenance_ctx(ctx)

    assert adapted is not ctx, "expected the data_item substitution path, not the pass-through"
    for context_field in dataclasses.fields(PipelineContext):
        assert hasattr(adapted, context_field.name), (
            f"PipelineContext.{context_field.name} does not survive the COGX import "
            "ctx adaptation (_provenance_ctx in cognee/modules/migration/loader.py). "
            "If you added a parameter to PipelineContext, the cogx import path must "
            "carry it: task consumers such as add_data_points read ctx attributes "
            "unconditionally (this is how pipeline_run_id regressed)."
        )
        if context_field.name == "data_item":
            continue
        assert getattr(adapted, context_field.name) == getattr(ctx, context_field.name), (
            f"PipelineContext.{context_field.name} changed value across _provenance_ctx"
        )


def test_data_item_id_substituted_from_data_id():
    ctx = _full_context()

    adapted = _provenance_ctx(ctx)

    assert adapted.data_item.id == ctx.data_item.data_id


def test_passthrough_when_data_item_already_has_id():
    ctx = PipelineContext(data_item=SimpleNamespace(id=uuid4()))

    assert _provenance_ctx(ctx) is ctx


def test_none_ctx_stays_none():
    assert _provenance_ctx(None) is None
