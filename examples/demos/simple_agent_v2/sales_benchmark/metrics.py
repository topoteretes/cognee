"""Metrics collection and reporting for the sales benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import litellm

from .models import ConversationResult
from .leads import LEADS

# Global reference to the currently active collector.
# The monkey-patch delegates to whoever is active.
_active_collector: Optional[MetricsCollector] = None
_original_acompletion = None
_patched = False


def _install_global_patch() -> None:
    """Patch litellm.acompletion once. Must be called before any LLM client is created."""
    global _original_acompletion, _patched
    if _patched:
        return
    _original_acompletion = litellm.acompletion

    async def patched_acompletion(*args, **kwargs):
        response = await _original_acompletion(*args, **kwargs)
        if _active_collector is not None and _active_collector._current is not None:
            _active_collector._current.llm_calls += 1
            usage = getattr(response, "usage", None)
            if usage:
                _active_collector._current.prompt_tokens += (
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                _active_collector._current.completion_tokens += (
                    getattr(usage, "completion_tokens", 0) or 0
                )
        return response

    litellm.acompletion = patched_acompletion
    _patched = True


@dataclass
class LeadMetrics:
    """Metrics for a single lead conversation."""

    lead_id: str
    persona_tag: str
    mode: str
    outcome: str = ""
    rounds: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    r1_feature_correct: bool = False


@dataclass
class MetricsCollector:
    """Collects per-lead metrics by monkey-patching litellm.acompletion.

    Instructor wraps litellm.acompletion internally, so litellm.success_callback
    never fires. Instead, we patch acompletion once (globally) and route metrics
    to whichever collector is currently active.
    """

    _current: Optional[LeadMetrics] = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    results: list[LeadMetrics] = field(default_factory=list)

    def start_lead(self, lead_id: str, persona_tag: str, mode: str) -> None:
        self._current = LeadMetrics(lead_id=lead_id, persona_tag=persona_tag, mode=mode)
        self._start_time = time.monotonic()

    def end_lead(self, conv_result: ConversationResult) -> None:
        if self._current is None:
            return
        self._current.outcome = conv_result.outcome
        self._current.rounds = conv_result.rounds
        self._current.wall_time_s = round(time.monotonic() - self._start_time, 2)
        # Round-1 feature accuracy: did the agent pick the right feature on first pitch?
        if conv_result.features_pitched:
            lead_profile = next(
                (l for l in LEADS if l.lead_id == conv_result.lead_id), None
            )
            if lead_profile:
                self._current.r1_feature_correct = (
                    conv_result.features_pitched[0] == lead_profile.must_have_feature
                )
        self.results.append(self._current)
        self._current = None

    def activate(self) -> None:
        """Set this collector as the active one receiving metrics."""
        global _active_collector
        _install_global_patch()
        _active_collector = self

    def deactivate(self) -> None:
        global _active_collector
        if _active_collector is self:
            _active_collector = None


def print_comparison(
    nomem: MetricsCollector,
    ctx: MetricsCollector,
    mem: MetricsCollector,
) -> None:
    """Print a 3-way comparison table."""
    header = (
        f"{'Lead':<6} {'Persona':<28} {'Mode':<10} {'Result':<12} "
        f"{'R1':>3} {'Rounds':>6} {'Calls':>6} {'Tokens':>8} {'Time':>7}"
    )
    sep = "-" * len(header)

    print()
    print("BENCHMARK RESULTS: NO-MEMORY vs CONTEXT vs COGNEE GRAPH")
    print(sep)
    print(header)
    print(sep)

    nomem_by_id = {r.lead_id: r for r in nomem.results}
    ctx_by_id = {r.lead_id: r for r in ctx.results}
    mem_by_id = {r.lead_id: r for r in mem.results}
    all_ids = list(
        dict.fromkeys(
            r.lead_id for r in nomem.results + ctx.results + mem.results
        )
    )

    for lead_id in all_ids:
        for mode, lookup in [
            ("no-mem", nomem_by_id),
            ("context", ctx_by_id),
            ("graph", mem_by_id),
        ]:
            r = lookup.get(lead_id)
            if r is None:
                continue
            total_tokens = r.prompt_tokens + r.completion_tokens
            r1_mark = "✓" if r.r1_feature_correct else "✗"
            print(
                f"{r.lead_id:<6} {r.persona_tag:<28} {mode:<10} {r.outcome:<12} "
                f"{r1_mark:>3} {r.rounds:>6} {r.llm_calls:>6} {total_tokens:>8,} {r.wall_time_s:>6.1f}s"
            )

    print(sep)

    def _agg(results: list[LeadMetrics]):
        if not results:
            return {}
        total = len(results)
        wins = sum(1 for r in results if r.outcome == "CLOSED_WON")
        avg_rounds = sum(r.rounds for r in results) / total
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
        total_calls = sum(r.llm_calls for r in results)
        total_time = sum(r.wall_time_s for r in results)
        r1_correct = sum(1 for r in results if r.r1_feature_correct)
        won_results = [r for r in results if r.outcome == "CLOSED_WON"]
        avg_rounds_to_win = (
            sum(r.rounds for r in won_results) / len(won_results) if won_results else None
        )
        first_r1 = None
        for i, r in enumerate(results):
            if r.rounds == 1 and r.outcome == "CLOSED_WON":
                first_r1 = i + 1
                break
        return {
            "win_rate": wins / total * 100,
            "avg_rounds": avg_rounds,
            "avg_rounds_to_win": avg_rounds_to_win,
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "total_time": total_time,
            "r1_accuracy": r1_correct / total * 100,
            "first_r1_close": first_r1,
        }

    nm = _agg(nomem.results)
    cx = _agg(ctx.results)
    mm = _agg(mem.results)

    if not nm or not cx or not mm:
        return

    def _delta(a, b):
        if a == 0:
            return "n/a"
        pct = (b - a) / a * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.0f}%"

    print()
    print("AGGREGATE COMPARISON")
    print(sep)
    print(
        f"{'Metric':<20} {'No-Memory':>12} {'Context':>12} {'Graph':>12}"
    )
    print(sep)
    print(
        f"{'Win rate':<20} {nm['win_rate']:>11.0f}% {cx['win_rate']:>11.0f}% "
        f"{mm['win_rate']:>11.0f}%"
    )
    print(
        f"{'R1 feature accuracy':<20} {nm['r1_accuracy']:>11.0f}% {cx['r1_accuracy']:>11.0f}% "
        f"{mm['r1_accuracy']:>11.0f}%"
    )
    print(
        f"{'Avg rounds':<20} {nm['avg_rounds']:>12.1f} {cx['avg_rounds']:>12.1f} "
        f"{mm['avg_rounds']:>12.1f}"
    )
    nm_rw = f"{nm['avg_rounds_to_win']:.1f}" if nm["avg_rounds_to_win"] else "n/a"
    cx_rw = f"{cx['avg_rounds_to_win']:.1f}" if cx["avg_rounds_to_win"] else "n/a"
    mm_rw = f"{mm['avg_rounds_to_win']:.1f}" if mm["avg_rounds_to_win"] else "n/a"
    print(
        f"{'Avg rounds to win':<20} {nm_rw:>12} {cx_rw:>12} {mm_rw:>12}"
    )
    print(
        f"{'Total LLM calls':<20} {nm['total_calls']:>12,} {cx['total_calls']:>12,} "
        f"{mm['total_calls']:>12,}"
    )
    print(
        f"{'Total tokens':<20} {nm['total_tokens']:>12,} {cx['total_tokens']:>12,} "
        f"{mm['total_tokens']:>12,}"
    )
    print(
        f"{'Total time':<20} {nm['total_time']:>11.1f}s {cx['total_time']:>11.1f}s "
        f"{mm['total_time']:>11.1f}s"
    )

    nm_r1 = f"Lead #{nm['first_r1_close']}" if nm["first_r1_close"] else "never"
    cx_r1 = f"Lead #{cx['first_r1_close']}" if cx["first_r1_close"] else "never"
    mm_r1 = f"Lead #{mm['first_r1_close']}" if mm["first_r1_close"] else "never"
    print(f"{'First R1 close':<20} {nm_r1:>12} {cx_r1:>12} {mm_r1:>12}")
    print(sep)

    # Delta table
    print()
    print("DELTAS vs NO-MEMORY BASELINE")
    print(sep)
    print(
        f"{'Metric':<20} {'Context Δ':>12} {'Graph Δ':>12}"
    )
    print(sep)
    print(
        f"{'Win rate':<20} {_delta(nm['win_rate'], cx['win_rate']):>12} "
        f"{_delta(nm['win_rate'], mm['win_rate']):>12}"
    )
    print(
        f"{'R1 feature accuracy':<20} {_delta(nm['r1_accuracy'], cx['r1_accuracy']):>12} "
        f"{_delta(nm['r1_accuracy'], mm['r1_accuracy']):>12}"
    )
    print(
        f"{'Avg rounds':<20} {_delta(nm['avg_rounds'], cx['avg_rounds']):>12} "
        f"{_delta(nm['avg_rounds'], mm['avg_rounds']):>12}"
    )
    if nm["avg_rounds_to_win"] and cx["avg_rounds_to_win"] and mm["avg_rounds_to_win"]:
        print(
            f"{'Avg rounds to win':<20} {_delta(nm['avg_rounds_to_win'], cx['avg_rounds_to_win']):>12} "
            f"{_delta(nm['avg_rounds_to_win'], mm['avg_rounds_to_win']):>12}"
        )
    print(
        f"{'Total LLM calls':<20} {_delta(nm['total_calls'], cx['total_calls']):>12} "
        f"{_delta(nm['total_calls'], mm['total_calls']):>12}"
    )
    print(
        f"{'Total tokens':<20} {_delta(nm['total_tokens'], cx['total_tokens']):>12} "
        f"{_delta(nm['total_tokens'], mm['total_tokens']):>12}"
    )
    print(
        f"{'Total time':<20} {_delta(nm['total_time'], cx['total_time']):>12} "
        f"{_delta(nm['total_time'], mm['total_time']):>12}"
    )
    print(sep)
