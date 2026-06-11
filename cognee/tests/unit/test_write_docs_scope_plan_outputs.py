from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "write_docs_scope_plan_outputs.py"
SPEC = importlib.util.spec_from_file_location("write_docs_scope_plan_outputs", MODULE_PATH)
write_docs_scope_plan_outputs = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(write_docs_scope_plan_outputs)


def test_parse_docs_needed_true():
    scope_plan = """# Documentation Scope Plan

## Docs Needed
`true`

## Reason
The public API changed.
"""

    assert write_docs_scope_plan_outputs.parse_docs_needed(scope_plan) is True


def test_parse_docs_needed_false():
    scope_plan = """# Documentation Scope Plan

## Docs Needed
false

## Reason
Only tests changed.
"""

    assert write_docs_scope_plan_outputs.parse_docs_needed(scope_plan) is False


def test_parse_docs_needed_requires_section():
    with pytest.raises(ValueError, match="Docs Needed"):
        write_docs_scope_plan_outputs.parse_docs_needed("# Documentation Scope Plan\n")


def test_parse_docs_needed_rejects_unknown_value():
    scope_plan = """# Documentation Scope Plan

## Docs Needed
maybe

## Reason
Ambiguous.
"""

    with pytest.raises(ValueError, match="true.*false"):
        write_docs_scope_plan_outputs.parse_docs_needed(scope_plan)
