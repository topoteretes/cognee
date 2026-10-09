"""Per-llm_model orchestration: turn measurements into the JSON report.

Reads as: take this llm_model's chunk measurements, size the corpus, build the
two query-cost strategies, and read the reduction milestones off them.
"""

from __future__ import annotations

from cost_model import (
    ChunkMeasurement,
    CogneeQueryCost,
    FullContextQueryCost,
    average_measurement,
    corpus_ingestion_tokens,
    queries_for_reduction,
)
from measure import count_tokens


def build_report(all_measurements: list[ChunkMeasurement], text: str, args) -> dict:
    return {
        llm_model: analyze_llm_model(llm_model, all_measurements, text, args)
        for llm_model in args.llm_models
    }


def analyze_llm_model(
    llm_model: str,
    all_measurements: list[ChunkMeasurement],
    text: str,
    args,
) -> dict:
    chunk_measurements = [
        measurement for measurement in all_measurements if measurement.llm_model == llm_model
    ]
    average = average_measurement(chunk_measurements)
    corpus_tokens = args.corpus_tokens or count_tokens(text, llm_model)

    full_context_cost = FullContextQueryCost(corpus_tokens, args.query_overhead)
    cognee_cost = CogneeQueryCost(
        corpus_ingestion_tokens(average, corpus_tokens),
        args.retrieved_context,
    )
    reduction_milestones = {
        factor: queries_for_reduction(full_context_cost, cognee_cost, factor)
        for factor in args.reduction_factors
    }
    return assemble(
        chunk_measurements, average, full_context_cost, cognee_cost, reduction_milestones
    )


def assemble(
    chunk_measurements: list[ChunkMeasurement],
    average: ChunkMeasurement,
    full_context_cost: FullContextQueryCost,
    cognee_cost: CogneeQueryCost,
    reduction_milestones: dict[float, float | None],
) -> dict:
    return {
        "ingestion_multiplier": round(
            cognee_cost.ingestion_tokens / full_context_cost.corpus_tokens, 3
        ),
        "full_context": {
            "corpus_tokens": full_context_cost.corpus_tokens,
            "query_overhead_tokens": full_context_cost.query_overhead_tokens,
            "per_query_tokens": full_context_cost.corpus_tokens
            + full_context_cost.query_overhead_tokens,
        },
        "cognee": {
            "ingestion_tokens": round(cognee_cost.ingestion_tokens),
            "retrieved_context_tokens": cognee_cost.retrieved_context_tokens,
            "per_query_tokens": cognee_cost.retrieved_context_tokens,
        },
        "reduction_milestones": {
            str(factor): _round_or_none(queries) for factor, queries in reduction_milestones.items()
        },
        "average_chunk": _measurement_dict(average),
        "chunk_measurements": [_measurement_dict(m) for m in chunk_measurements],
    }


def _measurement_dict(measurement: ChunkMeasurement) -> dict:
    return {
        "llm_model": measurement.llm_model,
        "input_tokens": measurement.input_tokens,
        "summary_prompt_tokens": measurement.summary_prompt_tokens,
        "summary_completion_tokens": measurement.summary_completion_tokens,
        "graph_prompt_tokens": measurement.graph_prompt_tokens,
        "graph_completion_tokens": measurement.graph_completion_tokens,
        "ingestion_tokens": measurement.ingestion_tokens,
        "summary_ratio": round(measurement.summary_ratio, 4),
        "graph_ratio": round(measurement.graph_ratio, 4),
    }


def _round_or_none(queries: float | None) -> float | None:
    return None if queries is None else round(queries, 1)
