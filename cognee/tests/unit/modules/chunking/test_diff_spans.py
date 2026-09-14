"""Tests for the paragraph-anchored diff engine behind incremental updates.

``_diff_spans`` promises: disjoint, ordered (old_start, old_end, new_start,
new_end) character spans with the text OUTSIDE the spans byte-identical
between old and new. Everything downstream (region arithmetic, the no-loss
gate) builds on that contract, so it is verified here directly, by
differential fuzzing against the previous line-level implementation, across
arbitrary change shapes, and on the popular-content inputs that made a raw
line diff quadratic.
"""

import random
import time
from difflib import SequenceMatcher

from cognee.modules.chunking import incremental_chunking as ic
from cognee.modules.chunking.incremental_chunking import (
    _diff_spans,
    _shrink_span,
    compute_incremental_plan,
    validate_no_loss,
)


def assert_span_contract(old_text: str, new_text: str, spans) -> None:
    """The exact properties the downstream region arithmetic relies on."""
    prev_old, prev_new = 0, 0
    for old_start, old_end, new_start, new_end in spans:
        assert prev_old <= old_start <= old_end, "old span out of order"
        assert prev_new <= new_start <= new_end, "new span out of order"
        assert old_text[prev_old:old_start] == new_text[prev_new:new_start], (
            "between-span text differs"
        )
        prev_old, prev_new = old_end, new_end
    assert old_text[prev_old:] == new_text[prev_new:], "tail text differs"

    rebuilt, cursor = [], 0
    for old_start, old_end, new_start, new_end in spans:
        rebuilt.append(old_text[cursor:old_start])
        rebuilt.append(new_text[new_start:new_end])
        cursor = old_end
    rebuilt.append(old_text[cursor:])
    assert "".join(rebuilt) == new_text, "spans do not reconstruct the new text"


def _reference_diff_spans(old_text: str, new_text: str):
    """The previous engine (raw line-level SequenceMatcher): the test oracle."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    def offsets(lines):
        out = [0]
        for line in lines:
            out.append(out[-1] + len(line))
        return out

    old_offsets, new_offsets = offsets(old_lines), offsets(new_lines)
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        spans.append(
            _shrink_span(
                old_text,
                new_text,
                old_offsets[i1],
                old_offsets[i2],
                new_offsets[j1],
                new_offsets[j2],
            )
        )
    return spans


def _prose(rng, paragraphs, vocabulary=500, tag="tok"):
    out = []
    for _ in range(paragraphs):
        words = " ".join(f"{tag}{rng.randint(0, vocabulary)}" for _ in range(rng.randint(4, 14)))
        out.append(words + ".\n")
        if rng.random() < 0.6:
            out.append("\n")
    return "".join(out)


def _tile(text, size=200):
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


def _plan_affected(old_text, new_text):
    chunks = _tile(old_text)
    plan = compute_incremental_plan(old_text, chunks, new_text)
    validate_no_loss(chunks, plan, [[r.replacement_text] for r in plan.regions], new_text)
    return set(plan.affected_indices), len(chunks)


# --------------------------------------------------------------------------- #
# Differential fuzz: new engine vs the previous one as oracle
# --------------------------------------------------------------------------- #


def test_differential_fuzz_against_line_level_oracle():
    rng = random.Random(2024)
    for _ in range(400):
        old = _prose(rng, rng.randint(1, 40))
        new = old
        for _ in range(rng.randint(1, 5)):
            pos = rng.randint(0, len(new))
            op = rng.choice(["ins", "del", "sub", "para"])
            if op == "ins":
                new = new[:pos] + f"INS{rng.randint(0, 99)} " + new[pos:]
            elif op == "del":
                new = new[:pos] + new[min(len(new), pos + rng.randint(1, 40)) :]
            elif op == "sub":
                new = new[:pos] + "XX" + new[min(len(new), pos + rng.randint(1, 25)) :]
            else:
                new = new[:pos] + _prose(rng, 1) + new[pos:]

        spans = _diff_spans(old, new)
        assert_span_contract(old, new, spans)
        oracle_spans = _reference_diff_spans(old, new)
        assert_span_contract(old, new, oracle_spans)
        # Precision parity: the new engine may not be systematically coarser
        # than the old one (span placement may differ on ambiguous alignments).
        assert len(spans) <= len(oracle_spans) + 2
        _plan_affected(old, new)  # full plan + no-loss on every trial


def test_differential_fuzz_duplicate_heavy():
    """Small alphabet -> maximal alignment ambiguity and duplicate content."""
    rng = random.Random(1312)
    parts = ["alpha beta gamma.\n", "delta epsilon.\n", "\n", "zeta eta theta.\n"]
    for _ in range(400):
        old = "".join(rng.choice(parts) for _ in range(rng.randint(2, 80)))
        new = "".join(rng.choice(parts) for _ in range(rng.randint(2, 80)))
        assert_span_contract(old, new, _diff_spans(old, new))


def test_plan_precision_matches_oracle_on_local_edits():
    """For ordinary local edits both engines must flag the SAME chunks."""
    rng = random.Random(77)
    for _ in range(150):
        old = _prose(rng, rng.randint(6, 40))
        pos = rng.randint(0, max(0, len(old) - 30))
        new = old[:pos] + "CHANGED TEXT " + old[pos + rng.randint(0, 20) :]

        chunks = _tile(old)
        plan_new = compute_incremental_plan(old, chunks, new)
        validate_no_loss(chunks, plan_new, [[r.replacement_text] for r in plan_new.regions], new)

        original = ic._diff_spans
        ic._diff_spans = _reference_diff_spans
        try:
            plan_oracle = compute_incremental_plan(old, chunks, new)
        finally:
            ic._diff_spans = original
        assert set(plan_new.affected_indices) == set(plan_oracle.affected_indices)


# --------------------------------------------------------------------------- #
# Arbitrary change shapes — "there are no rules"
# --------------------------------------------------------------------------- #


def test_single_change_touches_one_chunk():
    rng = random.Random(1)
    old = _prose(rng, 300)
    new = old[: len(old) // 2] + "EDITED " + old[len(old) // 2 :]
    affected, total = _plan_affected(old, new)
    assert len(affected) <= 2 and total > 50


def test_first_and_last_characters_edited():
    rng = random.Random(2)
    old = _prose(rng, 300)
    new = "X" + old[1:-1] + "Y"
    affected, total = _plan_affected(old, new)
    assert 0 in affected and (total - 1) in affected
    assert len(affected) <= 4


def test_half_the_document_rewritten():
    rng = random.Random(3)
    old = _prose(rng, 400)
    quarter = len(old) // 4
    new = old[:quarter] + _prose(rng, 200, tag="new") + old[3 * quarter :]
    affected, total = _plan_affected(old, new)
    kept = total - len(affected)
    assert kept > total // 3  # the untouched halves survive


def test_everything_rewritten_except_one_chunk():
    rng = random.Random(4)
    old = _prose(rng, 400)
    chunks = _tile(old)
    survivor_index = len(chunks) // 2
    survivor = chunks[survivor_index]
    new = _prose(rng, 400, tag="aa") + survivor + _prose(rng, 400, tag="bb")

    plan = compute_incremental_plan(old, chunks, new)
    validate_no_loss(chunks, plan, [[r.replacement_text] for r in plan.regions], new)
    assert survivor_index not in set(plan.affected_indices)


def test_completely_different_document():
    rng = random.Random(5)
    old = _prose(rng, 300)
    new = _prose(rng, 300, tag="zz")
    affected, total = _plan_affected(old, new)
    assert len(affected) == total  # everything legitimately changed


def test_every_other_paragraph_rewritten():
    rng = random.Random(6)
    paragraphs = [_prose(rng, 1) for _ in range(200)]
    old = "".join(paragraphs)
    new = "".join(p if i % 2 == 0 else _prose(rng, 1, tag="swap") for i, p in enumerate(paragraphs))
    _plan_affected(old, new)  # contract + no-loss are the assertions here


def test_trim_is_not_load_bearing(monkeypatch):
    """Disabling the prefix/suffix fast path must not change any plan."""
    rng = random.Random(8)
    scenarios = []
    old = _prose(rng, 200)
    scenarios.append((old, old[: len(old) // 3] + "EDIT " + old[len(old) // 3 :]))
    scenarios.append((old, "X" + old[1:-1] + "Y"))
    scenarios.append((old, _prose(rng, 200, tag="zz")))

    baselines = [_plan_affected(o, n) for o, n in scenarios]
    monkeypatch.setattr(ic, "_line_trim", lambda a, b: (0, 0))
    assert [_plan_affected(o, n) for o, n in scenarios] == baselines


# --------------------------------------------------------------------------- #
# Popular content — the inputs that made a raw line diff quadratic
# --------------------------------------------------------------------------- #


def test_log_file_single_edit_is_fast_and_correct():
    old = "the same line repeated forever\n" * 20_000
    new = old[: len(old) // 2] + "CHANGED\n" + old[len(old) // 2 :]
    started = time.monotonic()
    spans = _diff_spans(old, new)
    elapsed = time.monotonic() - started
    assert_span_contract(old, new, spans)
    assert elapsed < 10  # the previous engine needed ~25s here, quadratic in size


def test_log_file_distant_edits_stay_near_linear():
    old = "the same line repeated forever\n" * 20_000
    quarter = len(old) // 4
    new = (
        old[:quarter]
        + "CHANGED-1\n"
        + old[quarter : 3 * quarter]
        + "CHANGED-2\n"
        + old[3 * quarter :]
    )
    started = time.monotonic()
    spans = _diff_spans(old, new)
    elapsed = time.monotonic() - started
    assert_span_contract(old, new, spans)
    assert elapsed < 10


def test_blank_line_heavy_prose_is_fast():
    """A novel's blank lines were the original 13.8s case at 3MB scale."""
    rng = random.Random(9)
    old = "".join(_prose(rng, 1) + "\n\n\n" for _ in range(3_000))
    new = old[:100] + "EDIT-A " + old[100 : len(old) // 2] + "\nEDIT-B\n" + old[len(old) // 2 :]
    started = time.monotonic()
    spans = _diff_spans(old, new)
    elapsed = time.monotonic() - started
    assert_span_contract(old, new, spans)
    assert elapsed < 10


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_identical_texts_yield_no_spans():
    rng = random.Random(10)
    text = _prose(rng, 50)
    assert _diff_spans(text, text) == []


def test_delete_everything():
    rng = random.Random(11)
    old = _prose(rng, 50)
    spans = _diff_spans(old, "")
    assert_span_contract(old, "", spans)


def test_insert_into_empty():
    rng = random.Random(12)
    new = _prose(rng, 50)
    spans = _diff_spans("", new)
    assert_span_contract("", new, spans)


def test_no_trailing_newline():
    old = "alpha beta\ngamma delta"
    new = "alpha beta\ngamma delta EXTRA"
    spans = _diff_spans(old, new)
    assert_span_contract(old, new, spans)


def test_single_line_document_no_newlines():
    old = "one very long single line with many words and no newline at all"
    new = old.replace("many", "MANY MORE")
    spans = _diff_spans(old, new)
    assert_span_contract(old, new, spans)


def test_crlf_line_endings():
    old = "first line\r\nsecond line\r\n\r\nthird line\r\n"
    new = "first line\r\nsecond CHANGED line\r\n\r\nthird line\r\n"
    spans = _diff_spans(old, new)
    assert_span_contract(old, new, spans)


def test_unicode_multibyte_content():
    old = "Привет мир 🌍 這是中文段落。\n\nSecond paragraph naïve café.\n"
    new = "Привет мир 🌍 這是中文段落。\n\nSecond paragraph naïve café UPDATED 🚀.\n"
    spans = _diff_spans(old, new)
    assert_span_contract(old, new, spans)


def test_blank_only_document():
    old = "\n\n\n\n"
    new = "\n\ncontent appeared\n\n"
    spans = _diff_spans(old, new)
    assert_span_contract(old, new, spans)


def test_deterministic():
    rng = random.Random(13)
    old = _prose(rng, 80)
    new = old[:500] + "CHANGE " + old[500:]
    assert _diff_spans(old, new) == _diff_spans(old, new)
