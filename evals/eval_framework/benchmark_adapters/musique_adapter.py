import os
import json
import random
from typing import Optional, Union, Any, LiteralString
import zipfile

import gdown

from evals.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class MusiqueQAAdapter(BaseBenchmarkAdapter):
    """
    Adapter to load and process the Musique QA dataset from a local .jsonl file.
    Optionally downloads and unzips the dataset if it does not exist locally.
    """

    dataset_info = {
        # Name of the final file we want to load
        "filename": "data/musique_ans_v1.0_dev.jsonl",
        # A Google Drive URL (or share link) to the ZIP containing this file
        "download_url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing",
        # The name of the ZIP archive we expect after downloading
        "zip_filename": "musique_v1.0.zip",
    }

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        auto_download: bool = True,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Loads the Musique QA dataset.

        :param limit: If set, randomly sample 'limit' items.
        :param seed: Random seed for sampling.
        :param auto_download: If True, attempt to download + unzip the dataset
            from Google Drive if the .jsonl file is not present locally.
        :return: (corpus_list, question_answer_pairs)
        """
        target_filename = self.dataset_info["filename"]

        # 1. Ensure the file is locally available; optionally download if missing
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

        if limit is not None and 0 < limit < len(data):
            random.seed(seed)
            data = random.sample(data, limit)

        corpus_list = []
        question_answer_pairs = []

        for item in data:
            # Each 'paragraphs' is a list of dicts; we can concatenate their 'paragraph_text'
            paragraphs = item.get("paragraphs", [])
            combined_paragraphs = " ".join(paragraph["paragraph_text"] for paragraph in paragraphs)
            corpus_list.append(combined_paragraphs)

            question = item.get("question", "")
            answer = item.get("answer", "")

            question_answer_pairs.append(
                {
                    "id": item.get("id", ""),
                    "question": question,
                    "answer": answer.lower() if isinstance(answer, str) else answer,
                }
            )

        return corpus_list, question_answer_pairs

    def _musique_download_file(self) -> None:
        """
        Download and unzip the Musique dataset if not already present locally.
        Uses gdown for Google Drive links.
        """
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
