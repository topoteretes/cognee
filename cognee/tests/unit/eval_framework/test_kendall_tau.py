"""Unit tests for the BEAM Kendall's tau-b metric helpers."""

import pytest

from cognee.eval_framework.beam.eval.metrics.kendall_tau import (
    _build_rank_vectors,
    _kendall_tau_b,
)


class TestKendallTauB:
    """Direct checks of the pure-Python tau-b implementation."""

    def test_perfect_agreement(self):
        assert _kendall_tau_b([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)

    def test_perfect_inversion(self):
        assert _kendall_tau_b([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)

    def test_ties_shrink_the_denominator(self):
        # One pair ties on x only: 5 concordant pairs out of 6, so tau-b is
        # 5 / sqrt(5 * 6) rather than 5 / 6.
        assert _kendall_tau_b([0, 1, 5, 5], [0, 1, 2, 3]) == pytest.approx(5 / (30**0.5))

    def test_all_tied_on_one_side_is_undefined(self):
        # A constant vector leaves no untied pairs on that side, so tau-b has no
        # usable denominator and the implementation reports 0.0.
        assert _kendall_tau_b([1, 1, 1], [0, 1, 2]) == 0.0

    @pytest.mark.parametrize(
        "x, y",
        [
            ([], []),
            ([0], [0]),
            ([0, 1], [0]),
        ],
    )
    def test_degenerate_inputs_return_zero(self, x, y):
        assert _kendall_tau_b(x, y) == 0.0


class TestBuildRankVectors:
    """The union-based rank construction shared with upstream BEAM."""

    def test_exact_match_ranks_are_identical(self):
        ref_ranks, sys_ranks = _build_rank_vectors(3, 3, {0: 0, 1: 1, 2: 2})
        assert ref_ranks == [0, 1, 2]
        assert sys_ranks == [0, 1, 2]

    def test_reordered_system_events_keep_reference_order(self):
        # The system lists the same three events, but reversed.
        ref_ranks, sys_ranks = _build_rank_vectors(3, 3, {0: 2, 1: 1, 2: 0})
        assert ref_ranks == [0, 1, 2]
        assert sys_ranks == [2, 1, 0]
        assert _kendall_tau_b(ref_ranks, sys_ranks) == pytest.approx(-1.0)

    def test_system_only_events_tie_in_the_reference_ranks(self):
        # Two hallucinated events are absent from the reference, so both must
        # collapse onto the shared tie rank instead of receiving ascending
        # reference ranks. Regression guard: ranking them 2 and 3 here would
        # make a hallucinated ordering look perfectly concordant.
        ref_ranks, sys_ranks = _build_rank_vectors(2, 4, {0: 0, 1: 1, 2: -1, 3: -1})

        tie_rank = 5  # len(union) + 1, with union = [ref0, ref1, sys2, sys3]
        assert ref_ranks == [0, 1, tie_rank, tie_rank]
        assert sys_ranks == [0, 1, 2, 3]

        # Concordance is now partial rather than perfect.
        tau_b = _kendall_tau_b(ref_ranks, sys_ranks)
        assert tau_b == pytest.approx(5 / (30**0.5))
        assert tau_b < 1.0

    def test_single_system_only_event_needs_no_tie(self):
        # A lone system-only event still takes the tie rank, but with no second
        # member sharing it there is no tie, so concordance stays perfect. This
        # is why the bug only becomes visible from two hallucinated events on.
        ref_ranks, sys_ranks = _build_rank_vectors(2, 3, {0: 0, 1: 1, 2: -1})
        tie_rank = 4  # len(union) + 1, with union = [ref0, ref1, sys2]
        assert ref_ranks == [0, 1, tie_rank]
        assert sys_ranks == [0, 1, 2]
        assert _kendall_tau_b(ref_ranks, sys_ranks) == pytest.approx(1.0)

    def test_missing_reference_events_tie_in_the_system_ranks(self):
        # The system recovered only the middle reference event; the other two
        # are absent from the system sequence and tie there.
        ref_ranks, sys_ranks = _build_rank_vectors(3, 1, {0: 1})
        tie_rank = 4  # len(union) + 1, with union = [ref0, ref1, ref2]
        assert ref_ranks == [0, 1, 2]
        assert sys_ranks == [tie_rank, 0, tie_rank]

    def test_empty_alignment_ties_every_reference_event(self):
        ref_ranks, sys_ranks = _build_rank_vectors(3, 0, {})
        assert ref_ranks == [0, 1, 2]
        assert sys_ranks == [4, 4, 4]

    def test_out_of_range_reference_index_is_treated_as_unmatched(self):
        # A malformed alignment from the LLM must not index past the reference
        # list; index 7 is dropped and the event is ranked as system-only.
        ref_ranks, sys_ranks = _build_rank_vectors(2, 2, {0: 0, 1: 7})
        assert ref_ranks == [0, 1, 4]
        assert sys_ranks == [0, 4, 1]
