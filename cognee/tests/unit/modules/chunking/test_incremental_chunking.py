"""Unit tests for the multi-region incremental-update planning algorithm."""

import random

import pytest

from cognee.modules.chunking.incremental_chunking import (
    IncrementalPlanError,
    compute_incremental_plan,
    validate_no_loss,
)


def _units(n_words: int, prefix: str = "w"):
    """Word units with their separators; every 7th separator is a newline
    (the diff is line-anchored, so the synthetic text must have lines)."""
    return [f"{prefix}{i:04d}" + ("\n" if i % 7 == 6 else " ") for i in range(n_words)]


def _make(n_words: int, prefix: str = "w", per_chunk: int = 10):
    units = _units(n_words, prefix)
    text = "".join(units)
    chunks = ["".join(units[i : i + per_chunk]) for i in range(0, len(units), per_chunk)]
    return text, chunks


def _region_texts(plan):
    return [[region.replacement_text] for region in plan.regions]


def test_identical_text_is_a_noop():
    text, chunks = _make(100)
    plan = compute_incremental_plan(text, chunks, text)
    assert plan.regions == []


def test_single_mid_edit_yields_one_region():
    text, chunks = _make(100)  # 10 chunks of 10 words
    start = text.index("w0045")
    end = text.index("w0055")
    new_text = text[:start] + "changed words here\n" + text[end:]

    plan = compute_incremental_plan(text, chunks, new_text)
    assert len(plan.regions) == 1
    assert plan.regions[0].affected_indices == [4, 5]
    validate_no_loss(chunks, plan, _region_texts(plan), new_text)


def test_three_disjoint_edits_yield_three_regions_and_keep_the_middle():
    """The Alice scenario in miniature: prepend + mid-insert + append."""
    text, chunks = _make(140)  # 14 chunks
    middle = text.index("w0070")
    new_text = (
        "HEAD LINE\n"
        + text[:middle]
        + "INSERTED BLOCK\nMORE INSERTED\n"
        + text[middle:]
        + "TAIL LINE\n"
    )

    plan = compute_incremental_plan(text, chunks, new_text)
    assert len(plan.regions) == 3, [r.affected_indices for r in plan.regions]
    affected = set(plan.affected_indices)
    # Edits touch only the first, one middle, and the last chunk — everything
    # else is kept without ever being re-chunked.
    assert len(affected) <= 4
    assert 0 in affected and (len(chunks) - 1) in affected
    kept = set(range(len(chunks))) - affected
    assert len(kept) >= 10
    validate_no_loss(chunks, plan, _region_texts(plan), new_text)


def test_adjacent_hunks_merge_into_one_region():
    text, chunks = _make(100)
    # Two edits inside the same chunk collapse into one region.
    a, b = text.index("w0042"), text.index("w0047")
    new_text = text[:a] + "X " + text[a:b] + "Y " + text[b:]
    plan = compute_incremental_plan(text, chunks, new_text)
    assert len(plan.regions) == 1
    validate_no_loss(chunks, plan, _region_texts(plan), new_text)


def test_multi_edit_reassembly_never_loses_content_under_fuzz():
    random.seed(23)
    text, chunks = _make(210, per_chunk=13)
    for trial in range(200):
        edit_count = random.randint(1, 3)
        positions = sorted(random.sample(range(len(text)), edit_count), reverse=True)
        mutated = text
        for pos in positions:
            kind = random.choice(["ins", "del", "sub"])
            if kind == "ins":
                mutated = mutated[:pos] + f" zz{trial}\n" + mutated[pos:]
            elif kind == "del":
                mutated = mutated[:pos] + mutated[pos + random.randrange(1, 60) :]
            else:
                mutated = mutated[:pos] + "#" + mutated[pos + 1 :]
        plan = compute_incremental_plan(text, chunks, mutated)
        validate_no_loss(chunks, plan, _region_texts(plan), mutated)


def test_non_tiling_chunks_raise():
    text, _ = _make(50)
    with pytest.raises(IncrementalPlanError):
        compute_incremental_plan(text, ["not-in-text"], text + "x\n")


def test_edit_at_document_edges():
    text, chunks = _make(60)
    plan = compute_incremental_plan(text, chunks, "HEAD\n" + text)
    assert plan.regions[0].affected_indices[0] == 0
    validate_no_loss(chunks, plan, _region_texts(plan), "HEAD\n" + text)

    plan = compute_incremental_plan(text, chunks, text + "TAIL\n")
    assert plan.regions[-1].affected_indices[-1] == len(chunks) - 1
    validate_no_loss(chunks, plan, _region_texts(plan), text + "TAIL\n")
