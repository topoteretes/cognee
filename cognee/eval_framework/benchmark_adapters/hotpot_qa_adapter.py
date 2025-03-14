import requests
import os
import json
import random
from typing import Optional, Any, List, Union, Tuple
from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class HotpotQAAdapter(BaseBenchmarkAdapter):
    dataset_info = {
        "filename": "hotpot_benchmark.json",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        # train: "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json" delete file after changing the url
        # distractor test: "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" delete file after changing the url
    }

    def __init__(self):
        super().__init__()
        self.metadata_field_name = "level"

    def _is_valid_supporting_fact(self, sentences: List[str], sentence_idx: Any) -> bool:
        """Validates if a supporting fact index is valid for the given sentences."""
        return sentences and isinstance(sentence_idx, int) and 0 <= sentence_idx < len(sentences)

    def _get_golden_context(self, item: dict[str, Any]) -> str:
        """Extracts and formats the golden context from supporting facts."""
        # Create a mapping of title to sentences for easy lookup
        context_dict = {title: sentences for (title, sentences) in item["context"]}

        # Get all supporting facts in order
        golden_contexts = []
        for title, sentence_idx in item["supporting_facts"]:
            sentences = context_dict.get(title, [])
            if not self._is_valid_supporting_fact(sentences, sentence_idx):
                continue
            golden_contexts.append(f"{title}: {sentences[sentence_idx]}")

        return "\n".join(golden_contexts)

    def _get_raw_corpus(self) -> List[dict[str, Any]]:
        """Loads the raw corpus data from file or URL and returns it as a list of dictionaries."""
        filename = self.dataset_info["filename"]

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                raw_corpus = json.load(f)
        else:
            response = requests.get(self.dataset_info["url"])
            response.raise_for_status()
            raw_corpus = response.json()

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(raw_corpus, f, ensure_ascii=False, indent=4)

        return raw_corpus

    def _get_corpus_entries(self, item: dict[str, Any]) -> List[str]:
        """Extracts corpus entries from the context of an item."""
        return [" ".join(sentences) for title, sentences in item["context"]]

    def _get_question_answer_pair(
        self,
        item: dict[str, Any],
        load_golden_context: bool = False,
    ) -> dict[str, Any]:
        """Extracts a question-answer pair from an item."""
        qa_pair = {
            "question": item["question"],
            "answer": item["answer"].lower(),
            self.metadata_field_name: item[self.metadata_field_name],
        }

        if load_golden_context:
            qa_pair["golden_context"] = self._get_golden_context(item)

        return qa_pair

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """Loads and processes the HotpotQA corpus, optionally with filtering and golden context."""
        raw_corpus = self._get_raw_corpus()

        if instance_filter is not None:
            raw_corpus = self._filter_instances(raw_corpus, instance_filter, id_key="_id")

        if limit is not None and 0 < limit < len(raw_corpus):
            random.seed(seed)
            raw_corpus = random.sample(raw_corpus, limit)

        corpus_list = []
        question_answer_pairs = []

        for item in raw_corpus:
            corpus_list.extend(self._get_corpus_entries(item))
            question_answer_pairs.append(self._get_question_answer_pair(item, load_golden_context))

        return corpus_list, question_answer_pairs
