import requests
import os
import json
import random
from typing import Optional, Any, List, Tuple
from evals.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


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

    def _process_item(
        self,
        item: dict[str, Any],
        corpus_list: List[str],
        question_answer_pairs: List[dict[str, Any]],
        load_golden_context: bool = False,
    ) -> None:
        """Processes a single item and adds it to the corpus and QA pairs."""
        for title, sentences in item["context"]:
            corpus_list.append(" ".join(sentences))

        qa_pair = {
            "question": item["question"],
            "answer": item["answer"].lower(),
            self.metadata_field_name: item[self.metadata_field_name],
        }

        if load_golden_context:
            qa_pair["golden_context"] = self._get_golden_context(item)

        question_answer_pairs.append(qa_pair)

    def load_corpus(
        self, limit: Optional[int] = None, seed: int = 42, load_golden_context: bool = False
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """Loads and processes the HotpotQA corpus, optionally with golden context."""
        filename = self.dataset_info["filename"]

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                corpus_json = json.load(f)
        else:
            response = requests.get(self.dataset_info["url"])
            response.raise_for_status()
            corpus_json = response.json()

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(corpus_json, f, ensure_ascii=False, indent=4)

        if limit is not None and 0 < limit < len(corpus_json):
            random.seed(seed)
            corpus_json = random.sample(corpus_json, limit)

        corpus_list = []
        question_answer_pairs = []

        for item in corpus_json:
            self._process_item(item, corpus_list, question_answer_pairs, load_golden_context)

        return corpus_list, question_answer_pairs
