"""Conformance test: hermes-agent dataset-name sanitization matches the shared spec.

Loads the shared case table in integrations/conformance/dataset_name_cases.json
(the same table the claude-code, codex, antigravity and openclaw tests use) and
checks the hermes-agent sanitizer against it. Also checks that an explicit
switch to an invalid name is refused, and that code-graph dataset names never
carry a dot (cognee rejects them).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cases():
    root = next(p for p in Path(__file__).resolve().parents if p.name == "integrations")
    return json.loads(
        (root / "conformance" / "dataset_name_cases.json").read_text(encoding="utf-8")
    )


def test_sanitizer_matches_shared_table():
    from cognee_integration_hermes.provider import _safe_dataset_name

    mismatches = []
    for case in _load_cases():
        result = _safe_dataset_name(case["input"], case["fallback"])
        if result != case["expected"]:
            mismatches.append(f"{case['input']!r} -> {result!r}, expected {case['expected']!r}")
    assert not mismatches, "dataset-name sanitization drift:\n" + "\n".join(mismatches)


def test_switch_refuses_an_invalid_name_with_a_suggestion():
    from cognee_integration_hermes.provider import CogneeMemoryProvider

    provider = CogneeMemoryProvider()
    result = json.loads(provider._handle_switch_dataset({"action": "switch", "dataset": "my.data"}))
    assert "cognee rejects spaces and dots" in result["error"]
    assert "'my_data'" in result["error"]


def test_code_graph_dataset_name_has_no_dot(tmp_path):
    from cognee_integration_hermes.code_graph import default_code_dataset

    repo = tmp_path / "Foo.JS"
    repo.mkdir()
    name = default_code_dataset(str(repo))
    assert name.startswith("codebase-foo-js-")
    assert "." not in name
