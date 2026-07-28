"""Unit tests for the incremental-update chunk planning algorithm."""

import random

import pytest

from cognee.modules.chunking.incremental_chunking import (
    IncrementalPlanError,
    compute_incremental_plan,
    validate_no_loss,
)


def _make_text(n_words: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i:04d}" for i in range(n_words))


def _tile(text: str, words_per_chunk: int):
    """Split text into chunks of N words, preserving the exact text."""
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        piece = " ".join(words[i : i + words_per_chunk])
        if i + words_per_chunk < len(words):
            piece += " "
        chunks.append(piece)
    return chunks


def test_identical_text_is_a_noop():
    text = _make_text(100)
    chunks = _tile(text, 10)
    plan = compute_incremental_plan(text, chunks, text)
    assert plan.affected_indices == []
    assert plan.replacement_region == ""


def test_mid_document_insertion_affects_only_straddled_chunks():
    """Spec scenario: an edit across two chunks replaces only that region."""
    text = _make_text(100)
    chunks = _tile(text, 10)  # 10 chunks of 10 words
    # Replace the span from the middle of chunk 4 to the middle of chunk 5 and
    # add 25 words of new content.
    start = text.index("w0045")
    end = text.index("w0055")
    new_text = text[:start] + _make_text(25, prefix="new") + " " + text[end:]

    plan = compute_incremental_plan(text, chunks, new_text)
    assert plan.affected_indices == [4, 5]
    assert plan.unchanged_prefix_count == 4
    assert plan.unchanged_suffix_count == 4
    # The region is exactly the new text between the affected chunk boundaries.
    kept_prefix = "".join(chunks[:4])
    kept_suffix = "".join(chunks[6:])
    assert kept_prefix + plan.replacement_region + kept_suffix == new_text
    # Any tiling of the region passes the no-loss check…
    validate_no_loss(chunks, plan, [plan.replacement_region], new_text)
    # …and a lossy split does not.
    with pytest.raises(IncrementalPlanError):
        validate_no_loss(chunks, plan, [plan.replacement_region[:-1]], new_text)


def test_region_reassembly_never_loses_content_under_fuzz():
    random.seed(11)
    text = _make_text(200)
    chunks = _tile(text, 13)
    for trial in range(200):
        pos = random.randrange(len(text))
        kind = random.choice(["ins", "del", "sub"])
        if kind == "ins":
            mutated = text[:pos] + f" zz{trial} " + text[pos:]
        elif kind == "del":
            mutated = text[:pos] + text[pos + random.randrange(1, 100) :]
        else:
            mutated = text[:pos] + "#" + text[pos + 1 :]
        plan = compute_incremental_plan(text, chunks, mutated)
        validate_no_loss(chunks, plan, [plan.replacement_region], mutated)


def test_non_tiling_chunks_raise():
    text = _make_text(50)
    with pytest.raises(IncrementalPlanError):
        compute_incremental_plan(text, ["not-in-text"], text + " x")


def test_edit_at_document_edges():
    text = _make_text(60)
    chunks = _tile(text, 10)
    prepended = "HEAD " + text
    plan = compute_incremental_plan(text, chunks, prepended)
    assert plan.affected_indices[0] == 0
    appended = text + " TAIL"
    plan = compute_incremental_plan(text, chunks, appended)
    assert plan.affected_indices[-1] == len(chunks) - 1
