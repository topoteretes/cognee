"""Kendall's tau-b metric for BEAM event_ordering questions.

Computes Kendall's tau-b rank correlation between the predicted and reference
event orderings, combined with F1 for event coverage.
Final score = tau_b_normalized × event_f1.

Only applies to event_ordering questions; returns None for all other types.

Reference: https://github.com/mohammadtavakoli78/BEAM/blob/main/src/evaluation/compute_metrics.py
"""

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _get_llm_client():
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
        get_llm_client,
    )

    return get_llm_client()


def _kendall_tau_b(x: List[int], y: List[int]) -> float:
    """Compute Kendall's tau-b rank correlation coefficient.

    Pure-Python implementation to avoid scipy dependency.
    Returns a value in [-1, 1], or 0.0 if computation is not possible.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0

    for i, j in combinations(range(n), 2):
        dx = x[i] - x[j]
        dy = y[i] - y[j]

        if dx == 0 and dy == 0:
            tied_x += 1
            tied_y += 1
        elif dx == 0:
            tied_x += 1
        elif dy == 0:
            tied_y += 1
        elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
            concordant += 1
        else:
            discordant += 1

    n_pairs = n * (n - 1) // 2
    denom_x = n_pairs - tied_x
    denom_y = n_pairs - tied_y

    if denom_x == 0 or denom_y == 0:
        return 0.0

    tau_b = (concordant - discordant) / ((denom_x * denom_y) ** 0.5)
    return tau_b


_ALIGN_SYSTEM_PROMPT = """You are comparing two lists of events to find matches.
For each event in the SYSTEM list, determine if it matches an event in the REFERENCE list
(same event described differently). Return a JSON object mapping system indices to reference
indices. Use -1 if no match exists. Each reference index can only be used once.

REFERENCE events:
{reference}

SYSTEM events:
{system}

Return ONLY a JSON object like: {{"0": 2, "1": -1, "2": 0, ...}}"""


_EXTRACT_EVENTS_PROMPT = """Extract the ordered list of events or steps from this text.
Return ONLY a JSON array of strings, each describing one event/step in the order they appear.

Text:
{text}

Return ONLY a JSON array like: ["first event", "second event", ...]"""


def _parse_json_list(raw: str) -> List[str]:
    """Parse a JSON list from LLM response."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item) for item in result]
    except json.JSONDecodeError:
        pass

    # Fallback: find array in text
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        try:
            result = json.loads(arr_match.group())
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

    # Last resort: split by newlines, filter numbered items
    lines = [re.sub(r"^\d+[\.\)]\s*", "", line.strip()) for line in text.split("\n")]
    return [line for line in lines if line and len(line) > 5]


def _parse_alignment(raw: str, n_system: int) -> Dict[int, int]:
    """Parse alignment mapping from LLM response."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {int(k): int(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group())
            return {int(k): int(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


class KendallTauMetric:
    """Kendall's tau-b metric for BEAM event_ordering questions.

    Returns None for non-event_ordering questions.
    For event_ordering: extracts events, aligns them, computes tau-b × F1.
    """

    def __init__(self):
        self.score: Optional[float] = None
        self.reason: Optional[str] = None

    def measure(self, test_case) -> Optional[float]:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, self.a_measure(test_case)).result()
        else:
            result = asyncio.run(self.a_measure(test_case))

        return result

    async def a_measure(self, test_case) -> Optional[float]:
        metadata = getattr(test_case, "additional_metadata", {}) or {}
        question_type = metadata.get("question_type", "")

        if question_type != "event_ordering":
            self.score = None
            self.reason = "Not applicable (not event_ordering)"
            return self.score

        actual_output = test_case.actual_output
        rubric = metadata.get("rubric", [])

        if not rubric or not actual_output:
            self.score = 0.0
            self.reason = "Missing rubric or actual output"
            return self.score

        llm_client = _get_llm_client()

        try:
            # The rubric for event_ordering IS the reference ordered list
            reference_events = rubric

            # Extract events from actual output
            extract_response = await llm_client.acreate_structured_output(
                text_input=_EXTRACT_EVENTS_PROMPT.format(text=actual_output),
                system_prompt="You extract events from text. Return only valid JSON.",
                response_model=str,
            )
            system_events = _parse_json_list(str(extract_response))

            if not system_events:
                self.score = 0.0
                self.reason = "Could not extract events from response"
                return self.score

            # Align system events to reference events via LLM
            ref_formatted = "\n".join(
                f"{i}: {e}" for i, e in enumerate(reference_events)
            )
            sys_formatted = "\n".join(
                f"{i}: {e}" for i, e in enumerate(system_events)
            )

            align_response = await llm_client.acreate_structured_output(
                text_input=_ALIGN_SYSTEM_PROMPT.format(
                    reference=ref_formatted, system=sys_formatted
                ),
                system_prompt="You align event lists. Return only valid JSON.",
                response_model=str,
            )
            alignment = _parse_alignment(str(align_response), len(system_events))

            # Compute F1 based on matched events
            matched_ref_indices = {
                v for v in alignment.values() if v >= 0 and v < len(reference_events)
            }
            tp = len(matched_ref_indices)
            fp = len(system_events) - tp
            fn = len(reference_events) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Build rank vectors for matched events
            # Union of all events (reference order first, then unmatched system)
            union = list(range(len(reference_events)))
            for i in range(len(system_events)):
                if alignment.get(i, -1) < 0:
                    union.append(len(reference_events) + i)

            tie_rank = len(union) + 1

            # Reference ranks: natural order 0, 1, 2, ...
            ref_ranks = list(range(len(union)))

            # System ranks: based on alignment
            sys_rank_map = {}
            sys_pos = 0
            for i in range(len(system_events)):
                ref_idx = alignment.get(i, -1)
                if ref_idx >= 0 and ref_idx < len(reference_events):
                    sys_rank_map[ref_idx] = sys_pos
                else:
                    sys_rank_map[len(reference_events) + i] = sys_pos
                sys_pos += 1

            sys_ranks = [sys_rank_map.get(u, tie_rank) for u in union]

            tau_b = _kendall_tau_b(ref_ranks, sys_ranks)
            tau_b_norm = (tau_b + 1) / 2  # Normalize from [-1,1] to [0,1]

            self.score = tau_b_norm * f1
            self.reason = (
                f"Kendall tau-b={tau_b:.3f} (norm={tau_b_norm:.3f}), "
                f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}), "
                f"final={self.score:.3f}, "
                f"matched {tp}/{len(reference_events)} reference events"
            )
            return self.score

        except Exception as e:
            logger.error(f"KendallTauMetric failed: {e}")
            self.score = 0.0
            self.reason = f"ERROR: {e}"
            return self.score
