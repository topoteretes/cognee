import os
import json
import random
from typing import Optional, Any, List, Union, Tuple
import zipfile

import gdown

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class MusiqueQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the Musique QA dataset with local file loading and optional download."""

    dataset_info = {
        "filename": "data/musique_ans_v1.0_dev.jsonl",
        "download_url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing",
        "zip_filename": "musique_v1.0.zip",
    }

    def _get_golden_context(self, item: dict[str, Any]) -> str:
        """Extracts golden context from question decomposition and supporting paragraphs."""
        golden_context = []
        paragraphs = item.get("paragraphs", [])

        # Process each decomposition step
        for step in item.get("question_decomposition", []):
            # Add the supporting paragraph if available
            support_idx = step.get("paragraph_support_idx")
            if isinstance(support_idx, int) and 0 <= support_idx < len(paragraphs):
                para = paragraphs[support_idx]
                golden_context.append(f"{para['title']}: {para['paragraph_text']}")

            # Add the step's question and answer
            golden_context.append(f"Q: {step['question']}")
            golden_context.append(f"A: {step['answer']}")
            golden_context.append("")  # Empty line between steps

        return "\n".join(golden_context)

    def _get_raw_corpus(self, auto_download: bool = True) -> List[dict[str, Any]]:
        """Loads the raw corpus data from file or downloads it if needed."""
        target_filename = self.dataset_info["filename"]

        if not os.path.exists(target_filename):
            if auto_download:
                self._musique_download_file()
            else:
                raise FileNotFoundError(
                    f"Expected dataset file not found: {target_filename}\n"
                    "Set auto_download=True or manually place the file."
                )

        with open(target_filename, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        return data

    def _get_corpus_entries(self, item: dict[str, Any]) -> List[str]:
        """Extracts corpus entries from the paragraphs of an item."""
        return [paragraph["paragraph_text"] for paragraph in item.get("paragraphs", [])]

    def _get_question_answer_pair(
        self,
        item: dict[str, Any],
        load_golden_context: bool = False,
    ) -> dict[str, Any]:
        """Extracts a question-answer pair from an item."""
        qa_pair = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", "").lower()
            if isinstance(item.get("answer"), str)
            else item.get("answer"),
        }

        if load_golden_context:
            qa_pair["golden_context"] = self._get_golden_context(item)

        return qa_pair

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        auto_download: bool = True,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """Loads and processes the Musique QA dataset with optional filtering."""
        raw_corpus = self._get_raw_corpus(auto_download)

        if instance_filter is not None:
            raw_corpus = self._filter_instances(raw_corpus, instance_filter, id_key="id")

        if limit is not None and 0 < limit < len(raw_corpus):
            random.seed(seed)
            raw_corpus = random.sample(raw_corpus, limit)

        corpus_list = []
        question_answer_pairs = []

        for item in raw_corpus:
            corpus_list.extend(self._get_corpus_entries(item))
            question_answer_pairs.append(self._get_question_answer_pair(item, load_golden_context))

        return corpus_list, question_answer_pairs

    def _musique_download_file(self) -> None:
        """Downloads and unzips the Musique dataset if not present locally."""
        url = self.dataset_info["download_url"]
        zip_filename = self.dataset_info["zip_filename"]
        target_filename = self.dataset_info["filename"]

        if os.path.exists(target_filename):
            print(f"File '{target_filename}' is already present. Skipping download.")
            return

        print(f"Attempting to download from Google Drive: {url}")
        gdown.download(url=url, output=zip_filename, quiet=False, fuzzy=True)

        if os.path.exists(zip_filename):
            print(f"Unzipping {zip_filename} ...")
            with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall()
        else:
            raise FileNotFoundError(f"Failed to download the zip file: {zip_filename}")

        if not os.path.exists(target_filename):
            raise FileNotFoundError(
                f"After unzipping, '{target_filename}' not found. "
                "Check the contents of the extracted files."
            )
