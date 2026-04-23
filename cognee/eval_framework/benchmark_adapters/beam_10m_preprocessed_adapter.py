"""BEAM-10M adapter that emits bounded, self-anchored preprocessed documents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter
from cognee.eval_framework.benchmark_adapters.beam_10m_adapter import (
    collect_beam_10m_plan_batches,
    load_beam_10m_row,
    parse_beam_10m_probing_questions,
)
from cognee.modules.chunking.conversation_preprocessing import (
    build_preprocessed_fragments_from_beam_10m_plan_batches,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class BEAM10MPreprocessedAdapter(BaseBenchmarkAdapter):
    """Load one BEAM-10M conversation as bounded documents grouped by plan."""

    def __init__(
        self,
        conversation_index: int = 0,
        plans: Optional[List[str]] = None,
        max_batches_per_plan: Optional[int] = None,
        preprocessed_max_chunk_size: int = 800,
    ):
        self.conversation_index = conversation_index
        self.plans = plans
        self.max_batches_per_plan = max_batches_per_plan
        self.preprocessed_max_chunk_size = preprocessed_max_chunk_size

    def load_plan_corpus(
        self,
        *,
        limit: Optional[int] = None,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[Tuple[str, List[str]]], List[Dict[str, Any]]]:
        logger.info(
            "Loading preprocessed BEAM-10M conversation_index=%s, plans=%s",
            self.conversation_index,
            self.plans or "ALL",
        )
        if load_golden_context:
            logger.warning("BEAM-10M preprocessed adapter does not support golden contexts")

        row = load_beam_10m_row(self.conversation_index)
        plan_batches = collect_beam_10m_plan_batches(
            row["chat"],
            plans=self.plans,
            max_batches_per_plan=self.max_batches_per_plan,
        )
        question_answer_pairs = parse_beam_10m_probing_questions(
            row,
            limit=limit,
            instance_filter=instance_filter,
            filter_instances_fn=self._filter_instances,
            conversation_index=self.conversation_index,
        )

        plan_documents: List[Tuple[str, List[str]]] = []
        for plan_name, batches in plan_batches.items():
            fragments = build_preprocessed_fragments_from_beam_10m_plan_batches(
                batches,
                self.preprocessed_max_chunk_size,
                plan=plan_name,
            )
            documents = [fragment.text for fragment in fragments]
            plan_documents.append((plan_name, documents))

        logger.info(
            "Loaded preprocessed BEAM-10M conversation %s: %s plans, %s probing questions",
            self.conversation_index,
            len(plan_documents),
            len(question_answer_pairs),
        )
        return plan_documents, question_answer_pairs

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        plan_documents, question_answer_pairs = self.load_plan_corpus(
            limit=limit,
            load_golden_context=load_golden_context,
            instance_filter=instance_filter,
        )
        corpus_list = [document for _, documents in plan_documents for document in documents]
        return corpus_list, question_answer_pairs
