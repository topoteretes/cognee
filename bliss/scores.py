import numpy as np
from pydantic import BaseModel

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment

PREMISE_INDEX = "HypothesisPremise_text"
DECOMPOSITION_NODE_TYPES = ["HypothesisPremise", "HypothesisConclusion", "Hypothesis"]


class Scores(BaseModel):
    feasibility: float
    novelty: float


NAN_SCORES = Scores(feasibility=float("nan"), novelty=float("nan"))


async def _embed_normalized(texts: list[str]) -> list[np.ndarray]:
    vectors = await get_vector_engine().embedding_engine.embed_text(texts)
    return [v / np.linalg.norm(v) for v in vectors]


def _hypothesis_decomposition(hypothesis) -> tuple[str, str, str] | None:
    neighbours = hypothesis.get_skeleton_neighbours()
    premise = next((n for n in neighbours if n.attributes.get("type") == "HypothesisPremise"), None)
    conclusion = next(
        (n for n in neighbours if n.attributes.get("type") == "HypothesisConclusion"), None
    )
    if not premise or not conclusion:
        return None
    prem_text, conc_text = premise.attributes.get("text"), conclusion.attributes.get("text")
    if not prem_text or not conc_text:
        return None
    return premise.id, prem_text, conc_text


async def _premise_conclusion_map() -> dict[str, tuple[str, str]]:
    fragment = await get_memory_fragment(
        properties_to_project=["text", "type"],
        memory_fragment_filter=[{"type": DECOMPOSITION_NODE_TYPES}],
    )
    decompositions = {}
    for hypothesis in fragment.nodes.values():
        if hypothesis.attributes.get("type") != "Hypothesis":
            continue
        decomposition = _hypothesis_decomposition(hypothesis)
        if decomposition:
            premise_id, prem_text, conc_text = decomposition
            decompositions[premise_id] = (prem_text, conc_text)
    return decompositions


def _feasibility(
    p: np.ndarray, c: np.ndarray, p_i: list[np.ndarray], c_i: list[np.ndarray]
) -> float:
    weighted = sum(float(np.dot(p, pi)) * ci for pi, ci in zip(p_i, c_i))
    norm = np.linalg.norm(weighted)
    if norm == 0:
        return float("nan")
    return float(np.dot(weighted / norm, c))


def _novelty(p: np.ndarray, c: np.ndarray, p_i: list[np.ndarray], c_i: list[np.ndarray]) -> float:
    distances = [
        0.5 * ((1 - float(np.dot(p, pi))) + (1 - float(np.dot(c, ci)))) for pi, ci in zip(p_i, c_i)
    ]
    return max(distances)


async def similar_decompositions(premise: str, top_k: int = 3) -> list[tuple[str, str]]:
    """Return premise/conclusion pairs from similar stored premises."""
    (p,) = await _embed_normalized([premise])
    similar_premises = await get_vector_engine().search(
        PREMISE_INDEX, query_vector=p.tolist(), limit=top_k
    )
    decompositions = await _premise_conclusion_map()
    return [
        decompositions[str(hit.id)] for hit in similar_premises if str(hit.id) in decompositions
    ]


async def score_hypothesis(
    premise: str,
    conclusion: str,
    top_k: int = 3,
    matches: list[tuple[str, str]] | None = None,
) -> Scores:
    """Score candidate premise/conclusion against top-k similar stored premises."""
    p, c = await _embed_normalized([premise, conclusion])
    if matches is None:
        matches = await similar_decompositions(premise, top_k)
    if not matches:
        return NAN_SCORES

    prem_texts, conc_texts = zip(*matches)
    # Re-embed matched texts; vectors already live in the index, but this keeps the code slightly simpler.
    p_i = await _embed_normalized(prem_texts)
    c_i = await _embed_normalized(conc_texts)

    return Scores(
        feasibility=_feasibility(p, c, p_i, c_i),
        novelty=_novelty(p, c, p_i, c_i),
    )


async def main() -> None:
    scores = await score_hypothesis(
        premise="Problem A inputs are routed through beta, then gamma, then delta.",
        conclusion="The pipeline output matches Problem A targets.",
    )
    print(scores.model_dump())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
