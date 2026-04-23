from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from cognee.eval_framework.answer_generation.beam_router import BEAMRouter
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_completion_decomposition_retriever import (
    GraphCompletionDecompositionRetriever,
)
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)

StrategyMode = Literal["fixed_retriever", "router"]


@dataclass(frozen=True)
class AnsweringStrategySpec:
    name: str
    mode: StrategyMode
    cls: type
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, **overrides: Any) -> Any:
        return self.cls(**{**self.default_kwargs, **overrides})


ANSWERING_STRATEGIES: dict[str, AnsweringStrategySpec] = {
    "cognee_graph_completion": AnsweringStrategySpec(
        name="cognee_graph_completion",
        mode="fixed_retriever",
        cls=GraphCompletionRetriever,
    ),
    "cognee_graph_completion_cot": AnsweringStrategySpec(
        name="cognee_graph_completion_cot",
        mode="fixed_retriever",
        cls=GraphCompletionCotRetriever,
    ),
    "cognee_graph_completion_context_extension": AnsweringStrategySpec(
        name="cognee_graph_completion_context_extension",
        mode="fixed_retriever",
        cls=GraphCompletionContextExtensionRetriever,
    ),
    "cognee_completion": AnsweringStrategySpec(
        name="cognee_completion",
        mode="fixed_retriever",
        cls=CompletionRetriever,
    ),
    "graph_summary_completion": AnsweringStrategySpec(
        name="graph_summary_completion",
        mode="fixed_retriever",
        cls=GraphSummaryCompletionRetriever,
    ),
    "graph_completion_decomposition": AnsweringStrategySpec(
        name="graph_completion_decomposition",
        mode="fixed_retriever",
        cls=GraphCompletionDecompositionRetriever,
    ),
    "beam_router": AnsweringStrategySpec(
        name="beam_router",
        mode="router",
        cls=BEAMRouter,
    ),
}


def get_answering_strategy_spec(name: str) -> AnsweringStrategySpec:
    try:
        return ANSWERING_STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(ANSWERING_STRATEGIES))
        raise ValueError(f"Unsupported qa_engine '{name}'. Available: {available}") from exc


def build_answering_strategy(name: str, **overrides: Any) -> Any:
    return get_answering_strategy_spec(name).build(**overrides)


def get_fixed_retriever_options() -> dict[str, type]:
    return {
        name: spec.cls
        for name, spec in ANSWERING_STRATEGIES.items()
        if spec.mode == "fixed_retriever"
    }


def build_sweep_config(
    strategy_name: str,
    *,
    config_name: Optional[str] = None,
    strategy_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    spec = get_answering_strategy_spec(strategy_name)
    kwargs = dict(strategy_kwargs or {})
    config = {
        "name": config_name or strategy_name,
        "mode": spec.mode,
    }

    if spec.mode == "router":
        config["router_cls"] = spec.cls
        config["router_kwargs"] = kwargs
    else:
        config["retriever_cls"] = spec.cls
        config["retriever_kwargs"] = kwargs

    return config
