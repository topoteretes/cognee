"""BEAM-10M benchmark adapter — loads 10M-token long-context conversations.

Dataset: https://huggingface.co/datasets/Mohammadta/BEAM-10M
Paper: "Beyond a Million Tokens: Benchmarking Long-Term Memory in LLMs"

Each conversation contains 10 plans, each plan has ~1M tokens of multi-session
chat batches. Together they form a ~10M token conversation with 20 probing
questions across 10 skill categories.
"""

import ast
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter
from cognee.shared.logging_utils import get_logger

logger = get_logger()

ALL_PLANS = [f"plan-{index}" for index in range(1, 11)]

_ANSWER_FIELD_NAMES = [
    "answer",
    "ideal_response",
    "ideal_answer",
    "ideal_summary",
]


def _extract_answer(question_dict: dict) -> str:
    """Extract the ground-truth answer from a BEAM probing question."""
    for field in _ANSWER_FIELD_NAMES:
        if field in question_dict and question_dict[field]:
            val = question_dict[field]
            return val if isinstance(val, str) else str(val)
    return ""


def _load_beam_10m_dataset():
    try:
        import datasets as _datasets_lib

        load_dataset = _datasets_lib.load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for BEAM-10M. Install it with: pip install datasets"
        )

    return load_dataset("Mohammadta/BEAM-10M", split="10M")


def load_beam_10m_row(conversation_index: int) -> dict[str, Any]:
    ds = _load_beam_10m_dataset()
    if conversation_index >= len(ds):
        raise IndexError(
            f"conversation_index={conversation_index} out of range "
            f"(BEAM-10M has {len(ds)} conversations)"
        )
    return ds[conversation_index]


def get_beam_10m_plan_names(
    chat_items: list[dict[str, Any]],
    plans: Optional[List[str]] = None,
) -> list[str]:
    if plans is not None:
        return plans

    if not chat_items:
        return []

    plan_names = [plan_name for plan_name in chat_items[0].keys() if plan_name.startswith("plan-")]
    return sorted(plan_names, key=lambda value: int(value.split("-")[1]))


def collect_beam_10m_plan_batches(
    chat_items: list[dict[str, Any]],
    *,
    plans: Optional[List[str]] = None,
    max_batches_per_plan: Optional[int] = None,
) -> dict[str, list[dict[str, Any]]]:
    plan_names = get_beam_10m_plan_names(chat_items, plans=plans)
    plan_batches: dict[str, list[dict[str, Any]]] = {plan_name: [] for plan_name in plan_names}

    for chat_item in chat_items:
        for plan_name in plan_names:
            batches = chat_item.get(plan_name, []) or []
            if max_batches_per_plan is not None and len(batches) > max_batches_per_plan:
                batches = batches[:max_batches_per_plan]
            plan_batches[plan_name].extend(batches)

    return plan_batches


def parse_beam_10m_probing_questions(
    row: dict[str, Any],
    *,
    limit: Optional[int] = None,
    instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    filter_instances_fn=None,
    conversation_index: int = 0,
) -> list[dict[str, Any]]:
    probing_raw = row.get("probing_questions", "")
    if isinstance(probing_raw, str):
        try:
            probing_data = ast.literal_eval(probing_raw)
        except (ValueError, SyntaxError):
            try:
                probing_data = json.loads(probing_raw)
            except json.JSONDecodeError:
                logger.error("Failed to parse probing_questions field")
                probing_data = {}
    else:
        probing_data = probing_raw if isinstance(probing_raw, dict) else {}

    question_answer_pairs = []

    for question_type, questions in probing_data.items():
        if not isinstance(questions, list):
            continue
        for q in questions:
            if not isinstance(q, dict) or "question" not in q:
                continue

            answer_text = _extract_answer(q)
            rubric = q.get("rubric", [])
            if isinstance(rubric, str):
                rubric = [rubric]

            qa_pair: Dict[str, Any] = {
                "question": q["question"],
                "answer": answer_text,
                "question_type": question_type,
                "rubric": rubric,
                "difficulty": q.get("difficulty", "unknown"),
                "conversation_id": row.get("conversation_id", str(conversation_index)),
            }

            question_answer_pairs.append(qa_pair)

    if instance_filter is not None:
        if filter_instances_fn is None:
            raise ValueError("filter_instances_fn must be provided when instance_filter is used")
        question_answer_pairs = filter_instances_fn(
            question_answer_pairs, instance_filter, id_key="question"
        )

    if limit is not None and limit < len(question_answer_pairs):
        question_answer_pairs = question_answer_pairs[:limit]

    return question_answer_pairs


def _flatten_10m_chat(
    chat_items: list[dict[str, Any]],
    plans: Optional[List[str]] = None,
    max_batches_per_plan: Optional[int] = None,
) -> str:
    """Flatten the nested BEAM-10M chat structure into a single text corpus.

    BEAM-10M has: chat = list of items, each item has plan-1..plan-10,
    each plan is a list of batches, each batch has turns (list of turn groups).

    Args:
        chat_items: The chat field from the dataset row.
        plans: Which plans to include (e.g., ["plan-1"]). None = all plans.
    """
    lines = []
    session_counter = 0

    plan_batches_by_name = collect_beam_10m_plan_batches(
        chat_items,
        plans=plans,
        max_batches_per_plan=max_batches_per_plan,
    )

    for plan_name, plan_batches in plan_batches_by_name.items():
        lines.append(f"=== {plan_name.upper()} ===")

        for batch in plan_batches:
            session_counter += 1
            batch_num = batch.get("batch_number", session_counter)
            lines.append(f"--- Session {batch_num} ({plan_name}) ---")

            turns = batch.get("turns", [])
            if turns is None:
                continue

            for turn_group in turns:
                if isinstance(turn_group, list):
                    for msg in turn_group:
                        role = msg.get("role", "unknown").capitalize()
                        content = msg.get("content", "")
                        time_anchor = msg.get("time_anchor")
                        prefix = f"[{time_anchor}] " if time_anchor else ""
                        lines.append(f"{prefix}{role}: {content}")
                elif isinstance(turn_group, dict):
                    role = turn_group.get("role", "unknown").capitalize()
                    content = turn_group.get("content", "")
                    time_anchor = turn_group.get("time_anchor")
                    prefix = f"[{time_anchor}] " if time_anchor else ""
                    lines.append(f"{prefix}{role}: {content}")

            lines.append("")

    return "\n".join(lines)


class BEAM10MAdapter(BaseBenchmarkAdapter):
    """Adapter for the BEAM-10M long-context conversation benchmark.

    Args:
        conversation_index: Which conversation to load (0-9). Default 0.
        plans: Which plans to include (e.g., ["plan-1"]). None = all 10 plans.
        max_batches_per_plan: Max session batches per plan. None = all.
    """

    def __init__(
        self,
        conversation_index: int = 0,
        plans: Optional[List[str]] = None,
        max_batches_per_plan: Optional[int] = None,
    ):
        self.conversation_index = conversation_index
        self.plans = plans
        self.max_batches_per_plan = max_batches_per_plan

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load a single BEAM-10M conversation as corpus + probing questions."""
        logger.info(
            f"Loading BEAM-10M dataset, conversation_index={self.conversation_index}, "
            f"plans={self.plans or 'ALL'}"
        )

        row = load_beam_10m_row(self.conversation_index)

        # Build corpus
        chat_items = row["chat"]
        corpus_text = _flatten_10m_chat(
            chat_items,
            plans=self.plans,
            max_batches_per_plan=self.max_batches_per_plan,
        )
        corpus_list = [corpus_text]

        # Parse probing questions
        question_answer_pairs = parse_beam_10m_probing_questions(
            row,
            limit=limit,
            instance_filter=instance_filter,
            filter_instances_fn=self._filter_instances,
            conversation_index=self.conversation_index,
        )

        logger.info(
            f"Loaded BEAM-10M conversation {self.conversation_index}: "
            f"{len(corpus_text):,} chars corpus, "
            f"{len(question_answer_pairs)} probing questions"
        )

        return corpus_list, question_answer_pairs
