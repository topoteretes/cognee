"""Unit tests for the explore/change/verify split of a coding unit (SDK-1335)."""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.session.trace_phases import (
    CHANGE,
    EXPLORE,
    VERIFY,
    PhaseBlock,
    assign_phases,
    classify_trace,
    command_phase,
    max_parts_for,
    merge_blocks,
    python_script_phase,
    split_into_parts,
    split_shell_command,
)


def _trace(tool, command=None):
    return SimpleNamespace(
        origin_function=tool, method_params={"command": command} if command else {}
    )


def _traces(spec):
    """``"E28 C2 V1"`` -> Read, Edit and ``pytest`` Bash traces in that order."""
    tools = {"E": ("Read", None), "C": ("Edit", None), "V": ("Bash", "pytest -q")}
    traces = []
    for item in spec.split():
        tool, command = tools[item[0]]
        traces += [_trace(tool, command) for _ in range(int(item[1:]))]
    return traces


# -- Acceptance examples ----------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "phase"),
    [
        ("sed -i 's/a/b/' src/x.py && pytest", CHANGE),
        ("cd /repo && pytest -q", VERIFY),
        ("cat x | grep y", EXPLORE),
    ],
)
def test_acceptance_commands(command, phase):
    assert command_phase(command) == phase


def test_four_trace_unit_is_one_part():
    parts = split_into_parts(_traces("E2 C1 V1"))
    assert [(p.phase, p.size) for p in parts] == [(EXPLORE, 4)]


def test_six_blocks_of_thirty_stay_six_parts():
    parts = split_into_parts(_traces("E30 C30 V30 E30 C30 V30"))
    assert [(p.phase, p.size) for p in parts] == [
        (EXPLORE, 30),
        (CHANGE, 30),
        (VERIFY, 30),
        (EXPLORE, 30),
        (CHANGE, 30),
        (VERIFY, 30),
    ]


def test_ninety_trace_merge_example():
    parts = split_into_parts(_traces("E28 C2 E3 C25 V1 C5 V26"))
    assert [(p.phase, p.size) for p in parts] == [(EXPLORE, 33), (CHANGE, 31), (VERIFY, 26)]
    assert [(p.start, p.end) for p in parts] == [(0, 33), (33, 64), (64, 90)]


# -- Classification -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "phase"),
    [
        ("pytest | grep FAIL", VERIFY),
        ("ls -la; cat 'a && b'", EXPLORE),
        ("git status && git diff", EXPLORE),
        ("git add . && git commit -m 'x'", CHANGE),
        ("git branch", EXPLORE),
        ("git branch -D foo", CHANGE),
        ("git stash", CHANGE),
        ("git stash list", EXPLORE),
        ("git log --oneline -5 | head", EXPLORE),
        ("python -m pytest", VERIFY),
        ("python script.py", VERIFY),
        ("python -c 'import requests; print(1)'", VERIFY),
        ("python -c 'import json; print(json.dumps({}))'", EXPLORE),
        ("python - <<'EOF'\nimport os\nprint(os.getcwd())\nEOF", EXPLORE),
        ("python - <<'EOF'\nopen('x', 'w').write('a')\nEOF", CHANGE),
        ("python - <<'EOF'\nimport subprocess\nsubprocess.run(['ls'])\nEOF", VERIFY),
        ("cognee-cli recall 'x'", VERIFY),
        ("FOO=1 uv run pytest", VERIFY),
        ("echo hi > out.txt", CHANGE),
        ("echo hi > /tmp/x", None),
        ("echo hi > $SCRATCH/x", None),
        ("rm /tmp/foo", None),
        ("rm foo", CHANGE),
        ("tee out.log", CHANGE),
        ("sed 's/a/b/' x.py", EXPLORE),
        ("export X=1", None),
        ("cd /x", None),
        ("", None),
    ],
)
def test_command_phases(command, phase):
    assert command_phase(command) == phase


def test_split_respects_quotes_and_heredocs():
    segments = split_shell_command(
        "cat 'a && b' | grep x; python - <<'EOF'\nprint('1 | 2')\nEOF\nls"
    )
    assert segments == [
        ("cat 'a && b'", False),
        ("grep x", True),
        ("python - <<'EOF'\nprint('1 | 2')\nEOF", False),
        ("ls", False),
    ]


def test_python_script_phase():
    assert python_script_phase("import os, sys\nprint(1)") == EXPLORE
    assert python_script_phase("from pathlib import Path\nPath('x').write_text('y')") == CHANGE
    assert python_script_phase("import numpy\n") == VERIFY


def test_tools_and_inheritance():
    assert classify_trace(_trace("Grep")) == EXPLORE
    assert classify_trace(_trace("Write")) == CHANGE
    assert classify_trace(_trace("TodoWrite")) is None
    assert classify_trace(_trace("mcp__cognee__search")) is None
    traces = [
        _trace("TodoWrite"),
        _trace("Edit"),
        _trace("Task"),
        _trace("Bash", "cd x"),
        _trace("Read"),
    ]
    assert assign_phases(traces) == [EXPLORE, CHANGE, CHANGE, CHANGE, EXPLORE]


# -- Merging ---------------------------------------------------------------------------


def test_max_parts_bounds():
    assert max_parts_for(120) == 4
    assert max_parts_for(121) == 5
    assert max_parts_for(180) == 6


def test_merge_takes_the_smaller_neighbour_and_the_larger_sides_phase():
    blocks = [PhaseBlock(EXPLORE, 0, 10), PhaseBlock(CHANGE, 10, 12), PhaseBlock(VERIFY, 12, 18)]
    merged = merge_blocks(blocks, 18)
    # C2 merges into V6 (the smaller neighbour) and takes its phase.
    assert [(b.phase, b.start, b.end) for b in merged] == [(EXPLORE, 0, 10), (VERIFY, 10, 18)]


def test_merge_ties_prefer_the_previous_neighbour_and_its_phase():
    blocks = [PhaseBlock(EXPLORE, 0, 6), PhaseBlock(CHANGE, 6, 8), PhaseBlock(VERIFY, 8, 14)]
    merged = merge_blocks(blocks, 14)
    assert [(b.phase, b.start, b.end) for b in merged] == [(EXPLORE, 0, 8), (VERIFY, 8, 14)]


def test_merge_caps_the_number_of_parts():
    blocks = [PhaseBlock([EXPLORE, CHANGE, VERIFY][i % 3], i * 10, i * 10 + 10) for i in range(6)]
    merged = merge_blocks(blocks, 60)
    assert len(merged) == 4
    assert all(b.size >= 5 for b in merged)
    assert merged[0].start == 0 and merged[-1].end == 60
