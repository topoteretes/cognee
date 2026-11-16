"""Conformance: hermes-agent recall trimming matches the shared table.

``integrations/conformance/recall_text_cases.json`` was generated from the hook
plugins' ``_recall_text.py``; ``cognee_integration_hermes.recall_text`` is a port
of it and must produce the same context, guidance and stats. Also covers
``cap_total``, the port of openclaw's ``capTotal``.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import recall_text as rt  # noqa: E402

_KINDS = {"trim", "cap_passage", "prune_entities", "split_passages"}


def _load_cases():
    root = next(p for p in Path(__file__).resolve().parents if p.name == "integrations")
    return json.loads((root / "conformance" / "recall_text_cases.json").read_text(encoding="utf-8"))


def _run(case):
    kind = case["kind"]
    if kind == "trim":
        context, guidance, stats = rt.trim_recall_text(
            case["input_text"],
            case["prompt"],
            case["session_id"],
            passage_chars=case["passage_chars"],
            context_chars=case["context_chars"],
        )
        return {"context": context, "guidance": guidance, "stats": stats}
    if kind == "cap_passage":
        return {"text": rt.cap_passage(case["input_text"], case["passage_chars"])}
    if kind == "prune_entities":
        text, removed = rt.prune_entities(case["input_text"])
        return {"text": text, "removed": removed}
    passages = rt.split_passages(case["input_text"])
    return {"passages": passages, "session_ids": [rt.passage_session_id(p) for p in passages]}


def test_the_table_covers_every_kind():
    assert {case["kind"] for case in _load_cases()} == _KINDS


def test_recall_text_matches_shared_table(monkeypatch):
    # A null option in the table means the reference default (2000 / 12000),
    # which is also this module's default; the environment must not move it.
    monkeypatch.delenv("COGNEE_RECALL_PASSAGE_CHARS", raising=False)
    monkeypatch.delenv("COGNEE_RECALL_CONTEXT_CHARS", raising=False)
    mismatches = []
    for case in _load_cases():
        result = _run(case)
        if result != case["expected"]:
            mismatches.append(f"{case['kind']}:{case['name']}: {result!r} != {case['expected']!r}")
    assert not mismatches, "recall-text trimming drift:\n" + "\n".join(mismatches)


def test_cap_total_keeps_parts_in_priority_order():
    assert rt.cap_total(["aaa", "bbb"], 0) == ["aaa", "bbb"]
    assert rt.cap_total(["aaa", "bbb"], 6) == ["aaa", "bbb"]
    head = "line one\nline two\n" + "x" * 40
    capped = rt.cap_total(["g" * 10, head, "dropped"], 10 + 30)
    assert capped[0] == "g" * 10
    assert capped[1] == "line one" + rt.TRUNCATION_MARK
    assert capped[2] == ""


def test_cap_total_never_cuts_an_atomic_part():
    capped = rt.cap_total(["a" * 5, "<code_graph>\nfact\n</code_graph>"], 20, lambda i: i == 1)
    assert capped == ["a" * 5, ""]
