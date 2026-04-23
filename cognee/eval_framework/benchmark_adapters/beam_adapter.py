"""BEAM benchmark adapter — loads synthetic long-context conversations.

Dataset: https://huggingface.co/datasets/Mohammadta/BEAM
Paper: "Beyond a Million Tokens: Benchmarking Long-Term Memory in LLMs"

Each conversation contains multi-session batches with 20 probing questions
across 10 skill categories (information_extraction, temporal_reasoning,
multi_session_reasoning, contradiction_resolution, event_ordering,
knowledge_update, summarization, abstention, preference_following,
instruction_following).
"""

import ast
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Map BEAM question types to a normalized key for the answer field
# (the dataset uses different field names for the ground-truth answer)
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


def _load_beam_dataset(split: str):
    try:
        import datasets as _datasets_lib

        load_dataset = _datasets_lib.load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for BEAM. Install it with: pip install datasets"
        )

    return load_dataset("Mohammadta/BEAM", split=split)


def load_beam_row(split: str, conversation_index: int) -> dict[str, Any]:
    ds = _load_beam_dataset(split)
    if conversation_index >= len(ds):
        raise IndexError(
            f"conversation_index={conversation_index} out of range "
            f"(split '{split}' has {len(ds)} conversations)"
        )
    return ds[conversation_index]


def truncate_beam_chat_batches(chat_batches: list, max_batches: Optional[int]) -> list:
    if max_batches is not None and len(chat_batches) > max_batches:
        logger.info(
            "Truncating conversation from %s batches to %s (max_batches)",
            len(chat_batches),
            max_batches,
        )
        return chat_batches[:max_batches]
    return chat_batches


def parse_beam_probing_questions(
    row: dict[str, Any],
    chat_batches: list,
    *,
    limit: Optional[int] = None,
    load_golden_context: bool = False,
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

            source_ids = q.get("source_chat_ids")
            if source_ids and load_golden_context:
                golden = BEAMAdapter._extract_golden_context(chat_batches, source_ids)
                if golden:
                    qa_pair["golden_context"] = golden

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


def _flatten_chat(chat_batches: list) -> str:
    """Flatten the nested chat structure into a single text corpus.

    The chat field is a list of batches, each batch is a list of messages.
    We produce a readable conversation transcript.
    """
    lines = []
    for batch_idx, batch in enumerate(chat_batches):
        lines.append(f"--- Session {batch_idx + 1} ---")
        for msg in batch:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            time_anchor = msg.get("time_anchor")
            prefix = f"[{time_anchor}] " if time_anchor else ""
            lines.append(f"{prefix}{role}: {content}")
        lines.append("")
    return "\n".join(lines)


class BEAMAdapter(BaseBenchmarkAdapter):
    """Adapter for the BEAM long-context conversation benchmark.

    Loads conversations from the HuggingFace dataset and extracts
    probing questions with rubric-based ground truth.

    Args:
        split: Dataset split to use ("100K", "500K", "1M"). Default "100K".
        conversation_index: Which conversation to load (0-indexed). Default 0.
    """

    def __init__(
        self,
        split: str = "100K",
        conversation_index: int = 0,
        max_batches: Optional[int] = None,
    ):
        self.split = split
        self.conversation_index = conversation_index
        self.max_batches = max_batches

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load a single BEAM conversation as corpus + probing questions.

        Returns:
            corpus_list: List with one entry — the full conversation transcript.
            question_answer_pairs: List of dicts with question, answer, rubric,
                question_type, and optional source_chat_ids.
        """
        logger.info(
            f"Loading BEAM dataset split={self.split}, conversation_index={self.conversation_index}"
        )

        row = load_beam_row(self.split, self.conversation_index)

        # Build corpus — optionally truncate to first N batches for faster local runs
        chat_batches = truncate_beam_chat_batches(row["chat"], self.max_batches)
        corpus_text = _flatten_chat(chat_batches)
        corpus_list = [corpus_text]

        question_answer_pairs = parse_beam_probing_questions(
            row,
            chat_batches,
            limit=limit,
            load_golden_context=load_golden_context,
            instance_filter=instance_filter,
            filter_instances_fn=self._filter_instances,
            conversation_index=self.conversation_index,
        )

        logger.info(
            f"Loaded BEAM conversation {self.conversation_index}: "
            f"{len(corpus_text)} chars corpus, "
            f"{len(question_answer_pairs)} probing questions"
        )

        return corpus_list, question_answer_pairs

    @staticmethod
    def _extract_golden_context(chat_batches: list, source_ids: Any) -> str:
        """Extract messages referenced by source_chat_ids as golden context."""
        # source_chat_ids can be a list of ints, or a dict with nested lists
        ids_to_find: set = set()

        if isinstance(source_ids, list):
            for item in source_ids:
                if isinstance(item, int):
                    ids_to_find.add(item)
        elif isinstance(source_ids, dict):
            for v in source_ids.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, int):
                            ids_to_find.add(item)
                elif isinstance(v, int):
                    ids_to_find.add(v)

        if not ids_to_find:
            return ""

        # Build id → message content map
        context_parts = []
        for batch in chat_batches:
            for msg in batch:
                msg_id = msg.get("id")
                if msg_id in ids_to_find:
                    role = msg.get("role", "unknown").capitalize()
                    content = msg.get("content", "")
                    context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)
