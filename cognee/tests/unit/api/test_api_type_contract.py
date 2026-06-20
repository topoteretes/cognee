"""Ratchet guard against FastAPI router⇄core type-contract regressions.

A router/core signature audit found ~91 type inconsistencies (untyped
``response_model``s, ``Form()`` in JSON body models, tuple field defaults, …).
The cleanup lands incrementally; this test stops the surface from GROWING:
the detector (``api_type_contract.py``) flags the unambiguous offenders, and
anything not already in ``api_type_contract_baseline.json`` fails the build.

To clear a baseline entry: fix the endpoint/model, then delete its line from the
baseline JSON. To (legitimately) accept a new one: it must be justified in
review and added explicitly. New violations are rejected by default.
"""

import json
from pathlib import Path

from cognee.tests.unit.api.api_type_contract import detect_all

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BASELINE_PATH = Path(__file__).with_name("api_type_contract_baseline.json")


def _load_baseline() -> set[str]:
    return set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))


def test_no_new_api_type_contract_violations():
    """Fail if any router introduces a type-contract violation not in the baseline."""
    current = {v.key() for v in detect_all(_REPO_ROOT)}
    baseline = _load_baseline()

    new_violations = sorted(current - baseline)
    assert not new_violations, (
        "New FastAPI router⇄core type-contract violation(s) detected.\n"
        "Fix the endpoint/model (preferred), or, if genuinely acceptable, add the "
        "key to cognee/tests/unit/api/api_type_contract_baseline.json with review "
        "justification:\n  " + "\n  ".join(new_violations)
    )


def test_baseline_has_no_stale_entries_warning(capsys):
    """Non-failing hygiene check: surface baseline entries already fixed so the
    allowlist can be pruned. Intentionally never fails (fixes land in other PRs)."""
    current = {v.key() for v in detect_all(_REPO_ROOT)}
    stale = sorted(_load_baseline() - current)
    if stale:
        print("\n[api-type-contract] baseline entries now FIXED — prune them:")
        for key in stale:
            print("  -", key)
    # Always passes; this is a nudge, not a gate.
    assert True


def test_detector_finds_known_patterns():
    """Sanity: the detector actually fires on each rule, so the gate isn't a no-op."""
    from cognee.tests.unit.api.api_type_contract import detect_in_source

    src = (
        "from pydantic import BaseModel\n"
        "from fastapi import Form\n"
        "class Body(BaseModel):\n"
        "    a: str = (Form(...),)\n"  # tuple default + Form in body
        "router = object()\n"
        "@router.post('/x', response_model=None)\n"
        "async def h():\n"
        "    return {}\n"
    )
    rules = {v.rule for v in detect_in_source(src, "fake.py")}
    assert "tuple_default_on_field" in rules
    assert "form_in_body_model" in rules
    assert "weak_response_model" in rules
