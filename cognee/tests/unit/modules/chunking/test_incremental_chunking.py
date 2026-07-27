"""Unit tests for the incremental-update chunk planning algorithm."""

import random
from math import ceil

import pytest

from cognee.modules.chunking.incremental_chunking import (
    IncrementalPlanError,
    balanced_token_split,
    compute_incremental_plan,
)

WORD = 1  # one token per word keeps expectations easy to compute


def _word_size(_word: str) -> int:
    return WORD


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
    plan = compute_incremental_plan(text, chunks, text, 10, word_size=_word_size)
    assert plan.affected_indices == []
    assert plan.new_chunk_texts == []


def test_mid_document_insertion_affects_only_straddled_chunks():
    """Spec scenario: an insertion across two chunks re-splits only that region."""
    text = _make_text(100)
    chunks = _tile(text, 10)  # 10 chunks of 10 tokens
    # Replace the span from the middle of chunk 4 to the middle of chunk 5 and
    # add 25 tokens (~2.5 chunk-budgets) of new content.
    start = text.index("w0045")
    end = text.index("w0055")
    new_text = text[:start] + _make_text(25, prefix="new") + " " + text[end:]

    plan = compute_incremental_plan(text, chunks, new_text, 10, word_size=_word_size)
    assert plan.affected_indices == [4, 5]
    assert plan.unchanged_prefix_count == 4
    assert plan.unchanged_suffix_count == 4
    region_tokens = len(plan.replacement_region.split())
    assert len(plan.new_chunk_texts) == ceil(region_tokens / 10)
    assert all(len(c.split()) <= 10 for c in plan.new_chunk_texts)


def test_reassembly_never_loses_content_under_fuzz():
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
        plan = compute_incremental_plan(text, chunks, mutated, 13, word_size=_word_size)
        reassembled = (
            "".join(chunks[: plan.unchanged_prefix_count])
            + "".join(plan.new_chunk_texts)
            + "".join(chunks[len(chunks) - plan.unchanged_suffix_count :])
        )
        assert reassembled == mutated


def test_balanced_split_respects_budget_and_balance():
    text = _make_text(95)
    pieces = balanced_token_split(text, 10, word_size=_word_size)
    assert "".join(pieces) == text
    sizes = [len(p.split()) for p in pieces]
    assert all(size <= 10 for size in sizes)
    assert len(pieces) == ceil(95 / 10)


def test_non_tiling_chunks_raise():
    text = _make_text(50)
    with pytest.raises(IncrementalPlanError):
        compute_incremental_plan(text, ["not-in-text"], text + " x", 10, word_size=_word_size)


def test_edit_at_document_edges():
    text = _make_text(60)
    chunks = _tile(text, 10)
    prepended = "HEAD " + text
    plan = compute_incremental_plan(text, chunks, prepended, 10, word_size=_word_size)
    assert plan.affected_indices[0] == 0
    appended = text + " TAIL"
    plan = compute_incremental_plan(text, chunks, appended, 10, word_size=_word_size)
    assert plan.affected_indices[-1] == len(chunks) - 1
