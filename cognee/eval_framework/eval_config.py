from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class EvalConfig(BaseSettings):
    # Corpus builder params
    building_corpus_from_scratch: bool = True
    number_of_samples_in_corpus: int = 1
    benchmark: str = "Dummy"  # Options: 'HotPotQA', 'Dummy', 'TwoWikiMultiHop'
    task_getter_type: str = "Default"  # Options: 'Default', 'CascadeGraph'

    # Question answering params
    answering_questions: bool = True
    qa_engine: str = (
        "cognee_completion"  # Options: 'cognee_completion' or 'cognee_graph_completion'
    )

    # Evaluation params
    evaluating_answers: bool = True
    evaluation_engine: str = "DeepEval"
    evaluation_metrics: List[str] = ["correctness", "EM", "f1"]
    deepeval_model: str = "gpt-4o-mini"

    # Visualization
    dashboard: bool = True

    # file paths
    questions_path: str = "questions_output.json"
    answers_path: str = "answers_output.json"
    metrics_path: str = "metrics_output.json"
    dashboard_path: str = "dashboard.html"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "building_corpus_from_scratch": self.building_corpus_from_scratch,
            "number_of_samples_in_corpus": self.number_of_samples_in_corpus,
            "benchmark": self.benchmark,
            "answering_questions": self.answering_questions,
            "qa_engine": self.qa_engine,
            "evaluating_answers": self.evaluating_answers,
            "evaluation_engine": self.evaluation_engine,
            "evaluation_metrics": self.evaluation_metrics,
            "dashboard": self.dashboard,
            "questions_path": self.questions_path,
            "answers_path": self.answers_path,
            "metrics_path": self.metrics_path,
            "dashboard_path": self.dashboard_path,
            "deepeval_model": self.deepeval_model,
            "task_getter_type": self.task_getter_type,
        }


@lru_cache
def get_llm_config():
    return EvalConfig()
