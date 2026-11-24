"""Conformance: hermes-agent capture policy matches the shared table.

``integrations/conformance/capture_policy_cases.json`` was generated from the
hook plugins' ``_capture_policy.py``. Each case's ``env`` becomes the settings
mapping the hermes port reads. Tool names are matched case-insensitively, as in
openclaw, so every case also runs with the lowercased name.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import capture_policy as cp  # noqa: E402


def _load_cases():
    root = next(p for p in Path(__file__).resolve().parents if p.name == "integrations")
    return json.loads((root / "conformance" / "capture_policy_cases.json").read_text("utf-8"))


def _decide(tool, params, env):
    try:
        return cp.allow_tool(tool, params, env)
    except Exception:
        return "error"


def test_tool_decisions_match_shared_table():
    mismatches = []
    for case in _load_cases()["tool_cases"]:
        for tool in {case["tool"], case["tool"].lower()}:
            result = _decide(tool, case["params"], case["env"])
            if result != case["expected"]:
                mismatches.append(
                    f"{case['note']} ({tool}): {result!r}, expected {case['expected']!r}"
                )
    assert not mismatches, "capture tool policy drift:\n" + "\n".join(mismatches)


def test_invalid_patterns_are_reported():
    for case in _load_cases()["tool_cases"]:
        if case["expected"] == "error":
            assert cp.capture_pattern_error(case["env"]), case["note"]


def test_redaction_matches_shared_table():
    mismatches = []
    for case in _load_cases()["redact_cases"]:
        if case.get("error"):
            try:
                cp.redact(case["input"], case["env"])
            except Exception:
                continue
            mismatches.append(f"{case['note']}: expected an error")
            continue
        result = cp.redact(case["input"], case["env"])
        if result != case["expected"]:
            mismatches.append(f"{case['note']}: {result!r}, expected {case['expected']!r}")
    assert not mismatches, "redaction drift:\n" + "\n".join(mismatches)


def test_config_values_stand_in_for_the_environment(monkeypatch):
    for name in cp.CONFIG_KEYS.values():
        monkeypatch.delenv(name, raising=False)
    env = cp.policy_from_config(
        {"capture_deny_tools": ["terminal(git:*)"], "capture_redact": False}
    )
    assert env["COGNEE_CAPTURE_DENY_TOOLS"] == '["terminal(git:*)"]'
    assert cp.allow_tool("terminal", {"command": "git status"}, env) is False
    assert cp.allow_tool("terminal", {"command": "ls"}, env) is True
    assert cp.redact("token=abc", env) == "token=abc"


def test_the_environment_fills_in_what_config_leaves_unset(monkeypatch):
    monkeypatch.setenv("COGNEE_CAPTURE", "false")
    env = cp.policy_from_config({})
    assert cp.allow_tool("read_file", {"path": "main.py"}, env) is False


def test_self_reference_is_recognised():
    assert cp.is_self_reference("cognee_recall", {})
    assert cp.is_self_reference("terminal", {"command": "hermes cognee status"})
    assert not cp.is_self_reference("terminal", {"command": "ls"})
