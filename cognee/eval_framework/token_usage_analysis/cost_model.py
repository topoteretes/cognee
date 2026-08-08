"""Cost comparison between full-context prompting and cognee persistent memory.

Pure arithmetic, no IO. The two querying strategies are modelled as objects that
each compute their own cumulative token cost over a number of queries. Every other
figure (parity, reduction milestones) is derived from those two objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMeasurement:
    """One sampled chunk run through one llm_model, in real measured tokens.

    Prompt/completion counts come from the LLM response, so the instruction
    wrapper and Pydantic schema are already included. input_tokens is a local
    token count of the chunk content, sharing a basis with corpus_tokens.
    """

    llm_model: str
    input_tokens: int
    summary_prompt_tokens: int
    summary_completion_tokens: int
    graph_prompt_tokens: int
    graph_completion_tokens: int

    @property
    def ingestion_tokens(self) -> int:
        return (
            self.summary_prompt_tokens
            + self.summary_completion_tokens
            + self.graph_prompt_tokens
            + self.graph_completion_tokens
        )

    @property
    def graph_ratio(self) -> float:
        """Graph-extraction output per content token — the density signal."""
        return self.graph_completion_tokens / self.input_tokens

    @property
    def summary_ratio(self) -> float:
        return self.summary_completion_tokens / self.input_tokens


@dataclass(frozen=True)
class FullContextQueryCost:
    """Cost of answering queries by sending the whole corpus each time."""

    corpus_tokens: int
    query_overhead_tokens: int  # instruction wrapper + question

    def tokens(self, queries: int) -> float:
        return queries * (self.corpus_tokens + self.query_overhead_tokens)


@dataclass(frozen=True)
class CogneeQueryCost:
    """Cost of answering queries via cognee: ingest once, then retrieve per query."""

    ingestion_tokens: float  # one-time cognee.remember() over the whole corpus
    retrieved_context_tokens: int  # cognee.recall() context per query (~constant at scale)

    def tokens(self, queries: int) -> float:
        return self.ingestion_tokens + queries * self.retrieved_context_tokens


def average_measurement(chunk_measurements: list[ChunkMeasurement]) -> ChunkMeasurement:
    """Return the typical chunk for one llm_model: the mean of the token fields.

    The ratios are properties, so they derive correctly from these means.
    """
    count = len(chunk_measurements)

    def mean(field: str) -> int:
        return round(sum(getattr(m, field) for m in chunk_measurements) / count)

    return ChunkMeasurement(
        llm_model=chunk_measurements[0].llm_model,
        input_tokens=mean("input_tokens"),
        summary_prompt_tokens=mean("summary_prompt_tokens"),
        summary_completion_tokens=mean("summary_completion_tokens"),
        graph_prompt_tokens=mean("graph_prompt_tokens"),
        graph_completion_tokens=mean("graph_completion_tokens"),
    )


def corpus_ingestion_tokens(average: ChunkMeasurement, corpus_tokens: int) -> float:
    """Scale one chunk's measured ingestion cost up to the whole corpus.

    The multiplier is ingestion tokens per content token; both the chunk's
    input_tokens and corpus_tokens are local counts, so they share a basis.
    """
    multiplier = average.ingestion_tokens / average.input_tokens
    return multiplier * corpus_tokens


def queries_for_reduction(
    full_context_cost: FullContextQueryCost,
    cognee_cost: CogneeQueryCost,
    factor: float,
) -> float | None:
    """Queries at which full-context costs `factor`x as much as cognee.

    factor 1.0 is parity (the cross-over). Returns None when the factor is
    unreachable, i.e. cognee's per-query cost alone already exceeds the target.
    """
    full_per_query = full_context_cost.corpus_tokens + full_context_cost.query_overhead_tokens
    denominator = full_per_query - factor * cognee_cost.retrieved_context_tokens
    if denominator <= 0:
        return None
    return factor * cognee_cost.ingestion_tokens / denominator
