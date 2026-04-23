"""BEAM adapter that emits bounded, self-anchored preprocessed documents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter
from cognee.eval_framework.benchmark_adapters.beam_adapter import (
    load_beam_row,
    parse_beam_probing_questions,
    truncate_beam_chat_batches,
)
from cognee.modules.chunking.conversation_preprocessing import (
    build_preprocessed_fragments_from_beam_batches,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class BEAMPreprocessedAdapter(BaseBenchmarkAdapter):
    """Load one BEAM conversation as many bounded preprocessed documents."""

    def __init__(
        self,
        split: str = "100K",
        conversation_index: int = 0,
        max_batches: Optional[int] = None,
        preprocessed_max_chunk_size: int = 800,
    ):
        self.split = split
        self.conversation_index = conversation_index
        self.max_batches = max_batches
        self.preprocessed_max_chunk_size = preprocessed_max_chunk_size

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        logger.info(
            "Loading preprocessed BEAM dataset split=%s, conversation_index=%s",
            self.split,
            self.conversation_index,
        )

        row = load_beam_row(self.split, self.conversation_index)
        chat_batches = truncate_beam_chat_batches(row["chat"], self.max_batches)
        question_answer_pairs = parse_beam_probing_questions(
            row,
            chat_batches,
            limit=limit,
            load_golden_context=load_golden_context,
            instance_filter=instance_filter,
            filter_instances_fn=self._filter_instances,
            conversation_index=self.conversation_index,
        )

        fragments = build_preprocessed_fragments_from_beam_batches(
            chat_batches,
            self.preprocessed_max_chunk_size,
        )
        corpus_list = [fragment.text for fragment in fragments]

        logger.info(
            "Loaded preprocessed BEAM conversation %s: %s documents, %s probing questions",
            self.conversation_index,
            len(corpus_list),
            len(question_answer_pairs),
        )

        return corpus_list, question_answer_pairs
