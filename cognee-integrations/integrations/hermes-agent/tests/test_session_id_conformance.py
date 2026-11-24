"""Conformance test: hermes-agent session-id sanitization matches the shared spec.

Loads the shared case table in integrations/conformance/session_id_cases.json
(the same table the claude-code, codex and openclaw tests use) and checks the
hermes-agent sanitizer against it. If any implementation drifts from the shared
rule, its test fails.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cases():
    root = next(p for p in Path(__file__).resolve().parents if p.name == "integrations")
    return json.loads((root / "conformance" / "session_id_cases.json").read_text(encoding="utf-8"))


def test_sanitizer_matches_shared_table():
    from cognee_integration_hermes.provider import _safe_session_component

    mismatches = []
    for case in _load_cases():
        result = _safe_session_component(case["input"])
        if result != case["expected"]:
            mismatches.append(f"{case['input']!r} -> {result!r}, expected {case['expected']!r}")
    assert not mismatches, "session-id sanitization drift:\n" + "\n".join(mismatches)
