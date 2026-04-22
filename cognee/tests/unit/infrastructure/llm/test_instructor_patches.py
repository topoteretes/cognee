"""Regression guards for the instructor ``openai_schema`` memoization patch.

The patch replaces ``instructor.processing.function_calls.openai_schema``
with a cached wrapper. Two things can silently break it:

1. An instructor version bump that changes where ``openai_schema`` lives
   or what signature it takes — the patch applies but is a no-op.
2. A regression that removes the memoization — same-input calls start
   returning distinct wrapper classes again, and the leak returns.

These tests pin both.
"""

import importlib


def test_patch_wraps_original_openai_schema():
    """After apply(), the module-level openai_schema is our cached wrapper
    and ``__wrapped__`` points back to the real instructor function."""
    from instructor.processing import function_calls as fc

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm import (
        _instructor_patches,
    )

    # Force a fresh apply by resetting the module-level guard.
    _instructor_patches._APPLIED = False
    importlib.reload(fc)
    original = fc.openai_schema

    _instructor_patches.apply()

    patched = fc.openai_schema
    assert patched is not original, "apply() did not replace openai_schema"
    assert getattr(patched, "__wrapped__", None) is original, (
        "cached wrapper must expose the original via __wrapped__ so future "
        "instructor version bumps are detectable"
    )


def test_patch_memoizes_same_class():
    """Calling the patched openai_schema twice with the same class must
    return the exact same wrapper — this is the whole point of the patch."""
    from pydantic import BaseModel

    from instructor.processing import function_calls as fc

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm import (
        _instructor_patches,
    )

    _instructor_patches._APPLIED = False
    importlib.reload(fc)
    _instructor_patches.apply()

    class Example(BaseModel):
        x: int

    w1 = fc.openai_schema(Example)
    w2 = fc.openai_schema(Example)
    assert w1 is w2, "patched openai_schema must memoize by input class"


def test_apply_is_idempotent():
    """Calling apply() twice must not double-wrap the function."""
    from instructor.processing import function_calls as fc

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm import (
        _instructor_patches,
    )

    _instructor_patches._APPLIED = False
    importlib.reload(fc)
    _instructor_patches.apply()
    first_patched = fc.openai_schema

    _instructor_patches.apply()
    second_patched = fc.openai_schema

    assert first_patched is second_patched, (
        "apply() must be idempotent — a second call should not wrap the "
        "already-cached wrapper"
    )
