import logging
import asyncio
import json
from evals.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from evals.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)

logging.basicConfig(level=logging.INFO)

eval_params = {
    # Corpus builder params
    "building_corpus_from_scratch": False,
    "number_of_samples_in_corpus": 5,
    "benchmark": "TwoWikiMultiHop",  # 'HotPotQA' or 'Dummy' or 'TwoWikiMultiHop'
    # Question answering params
    "answering_questions": True,
    "qa_engine": "cognee_graph_completion",  # 'cognee_completion' or 'cognee_graph_completion' or 'cognee_insights'
}

questions_file = "questions_output.json"


async def main():
    # Step 1: Corpus builder module
    if eval_params["building_corpus_from_scratch"]:
        logging.info("Starting Corpus Builder...")

        corpus_builder = CorpusBuilderExecutor()
        questions = await corpus_builder.build_corpus(
            limit=eval_params["number_of_samples_in_corpus"], benchmark=eval_params["benchmark"]
        )

        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        logging.info("Corpus Builder End...")

    # Step 2: Question answering module
    if eval_params["answering_questions"]:
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)

        print(f"Loaded {len(questions)} questions from {questions_file}")

        answer_generator = AnswerGeneratorExecutor()
        answers = await answer_generator.question_answering_non_parallel(
            questions=questions, qa_engine=eval_params["qa_engine"]
        )

        print(answers)

        logging.info("Question answering...")

    print()


if __name__ == "__main__":
    asyncio.run(main())
