import asyncio
import json
from typing import Any, Optional

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.retrieval.base_retriever import BaseRetriever
from pydantic import BaseModel

from decompose import Decomposition, decompose_hypothesis_text
from scores import Scores, score_hypothesis, similar_decompositions


class Match(BaseModel):
    premise: str
    conclusion: str


class BlissRetrieval(BaseModel):
    decomposition: Decomposition
    matches: list[Match]


class RetrievalResult(BaseModel):
    premise: str
    conclusion: str
    feasibility: float
    novelty: float
    explain: str


class Explanation(BaseModel):
    text: str


EXPLAIN_PROMPT = (
    "Explain feasibility and novelty in two short sentences. Use the scores and evidence provided."
)


def _format_context(retrieval: BlissRetrieval) -> str:
    lines = [
        f"Candidate premise: {retrieval.decomposition.premise}",
        f"Candidate conclusion: {retrieval.decomposition.conclusion}",
        "",
        "Similar stored hypotheses:",
    ]
    for index, match in enumerate(retrieval.matches, start=1):
        lines.append(f"{index}. Premise: {match.premise}\n   Conclusion: {match.conclusion}")
    return "\n".join(lines)


async def _explain(query: str, context: str, scores: Scores) -> str:
    response = await LLMGateway.acreate_structured_output(
        text_input=(
            f"Hypothesis: {query}\n\n{context}\n\n"
            f"Feasibility: {scores.feasibility:.3f}\nNovelty: {scores.novelty:.3f}"
        ),
        system_prompt=EXPLAIN_PROMPT,
        response_model=Explanation,
    )
    return response.text


class BlissRetriever(BaseRetriever):
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    async def get_retrieved_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
    ) -> BlissRetrieval:
        hypothesis = query or query_batch
        if not hypothesis:
            raise ValueError("query is required")
        decomposition = await decompose_hypothesis_text(hypothesis)
        matches = [
            Match(premise=premise, conclusion=conclusion)
            for premise, conclusion in await similar_decompositions(
                decomposition.premise, self.top_k
            )
        ]
        return BlissRetrieval(decomposition=decomposition, matches=matches)

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> str:
        return _format_context(retrieved_objects)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> list[dict]:
        retrieval = retrieved_objects
        matches = [(match.premise, match.conclusion) for match in retrieval.matches]
        scores = await score_hypothesis(
            retrieval.decomposition.premise,
            retrieval.decomposition.conclusion,
            matches=matches,
        )
        explain = await _explain(query or query_batch or "", context, scores)
        return [
            RetrievalResult(
                premise=retrieval.decomposition.premise,
                conclusion=retrieval.decomposition.conclusion,
                feasibility=scores.feasibility,
                novelty=scores.novelty,
                explain=explain,
            ).model_dump()
        ]


async def main() -> None:
    candidate = "Problem A is solved by routing inputs through beta, then gamma, then delta."
    result = (await BlissRetriever().get_completion(candidate))[0]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
