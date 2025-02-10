import logging
import json
from evals.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)


async def run_question_answering(params: dict) -> None:
    if params.get("answering_questions"):
        logging.info("Question answering started...")
        try:
            with open(params["questions_path"], "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {params['questions_path']}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {params['questions_path']}: {e}")

        logging.info(f"Loaded {len(questions)} questions from {params['questions_path']}")
        answer_generator = AnswerGeneratorExecutor()
        answers = await answer_generator.question_answering_non_parallel(
            questions=questions, qa_engine=params["qa_engine"]
        )
        with open(params["answers_path"], "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
        logging.info("Question answering End...")
