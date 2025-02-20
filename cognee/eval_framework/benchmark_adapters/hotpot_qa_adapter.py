import requests
import os
import json
import random
from typing import Optional, Any
from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class HotpotQAAdapter(BaseBenchmarkAdapter):
    dataset_info = {
        "filename": "hotpot_benchmark.json",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        # train: "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json" delete file after changing the url
        # distractor test: "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" delete file after changing the url
    }

    def load_corpus(
        self, limit: Optional[int] = None, seed: int = 42
    ) -> tuple[list[str], list[dict[str, Any]]]:
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
            for title, sentences in item["context"]:
                corpus_list.append(" ".join(sentences))

            question_answer_pairs.append(
                {
                    "question": item["question"],
                    "answer": item["answer"].lower(),
                    "level": item["level"],
                }
            )

        return corpus_list, question_answer_pairs
