"""Stub retriever and evaluator for the integrity tests. No LLM, database or network."""

from typing import Any


class StubRetriever:
    """Answers ``"answer to <question>"``; raises in the phase named by ``fail_in`` for any
    question listed in ``fail_on``."""

    def __init__(self, fail_on: set[str] | None = None, fail_in: str = "retrieval", answer=None):
        self.fail_on = fail_on or set()
        self.fail_in = fail_in
        self.fixed_answer = answer
        self.asked: list[str] = []

    async def get_retrieved_objects(self, query: str) -> list[str]:
        self.asked.append(query)
        if query in self.fail_on and self.fail_in == "retrieval":
            raise RuntimeError("vector store unavailable")
        return [query]

    async def get_context_from_objects(self, query: str, retrieved_objects: list[str]) -> str:
        return f"context for {query}"

    async def get_completion_from_context(
        self, query: str, retrieved_objects: list[str], context: str, **kwargs: Any
    ) -> Any:
        if query in self.fail_on and self.fail_in == "llm":
            raise TimeoutError("llm timed out")
        if self.fixed_answer is not None:
            return self.fixed_answer
        return [f"answer to {query}"]


def make_questions(count: int) -> list[dict[str, str]]:
    return [{"question": f"q{i}", "answer": f"golden {i}"} for i in range(count)]


def make_answer(index: int, status: str = "ok") -> dict[str, Any]:
    return {
        "question": f"q{index}",
        "answer": f"answer to q{index}" if status == "ok" else "",
        "golden_answer": f"golden {index}",
        "retrieval_context": "ctx",
        "status": status,
    }


def make_metrics_entry(index: int, score: float | None, status: str = "ok") -> dict[str, Any]:
    return {
        **make_answer(index, status),
        "metrics": {"correctness": {"score": score, "reason": None}},
    }
