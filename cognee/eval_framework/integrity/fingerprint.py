import hashlib
import json
from typing import Any

# Config keys that say where outputs go or how strictly to judge them; they never change what an
# answer or a score is. Everything else in the run config is result-determining, so a new
# EvalConfig field is fingerprinted by default.
OPERATIONAL_KEYS = frozenset(
    {
        "results_dir",
        "questions_path",
        "answers_path",
        "metrics_path",
        "aggregate_metrics_path",
        "dashboard_path",
        "dashboard",
        "resume",
        "resume_verified",
        "max_answer_failure_rate",
        "max_unscored_rate",
        "strict_integrity",
    }
)

# Result-determining for scores but not for answers: changing them must not discard answers.
EVALUATION_KEYS = frozenset(
    {
        "evaluating_answers",
        "evaluating_contexts",
        "evaluation_engine",
        "evaluation_metrics",
        "deepeval_model",
        "calculate_metrics",
        "direct_llm_system_prompt",
        "direct_llm_eval_prompt",
    }
)


def result_config(params: dict[str, Any]) -> dict[str, Any]:
    """The part of a run's params that determines its results (stored in the manifest)."""
    return {k: v for k, v in params.items() if k not in OPERATIONAL_KEYS and not k.startswith("_")}


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def answers_fingerprint(
    params: dict[str, Any],
    questions_path: str,
    system_prompt: str,
    system_prompt_text: Any,
    top_k: int | None,
    llm: dict[str, Any],
) -> str:
    """Digest of everything that determines an answer: the answering config, the exact questions
    file, the retriever's system prompt (name and text), top_k, and the LLM. A journal written
    under a different fingerprint must not be reused."""
    config = {k: v for k, v in result_config(params).items() if k not in EVALUATION_KEYS}
    return _digest(
        {
            "config": config,
            "questions": file_digest(questions_path),
            "system_prompt": system_prompt,
            "system_prompt_text": system_prompt_text,
            "top_k": top_k,
            "llm": llm,
        }
    )
