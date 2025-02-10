import logging
import json
from evals.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor


async def run_corpus_builder(params: dict, questions_path: str) -> None:
    if params.get("building_corpus_from_scratch"):
        logging.info("Corpus Builder started...")
        corpus_builder = CorpusBuilderExecutor(benchmark=params["benchmark"])
        questions = await corpus_builder.build_corpus(
            limit=params.get("number_of_samples_in_corpus")
        )
        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
        logging.info("Corpus Builder End...")
