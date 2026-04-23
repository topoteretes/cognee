from dataclasses import dataclass, field
from typing import Any, Optional

from cognee.eval_framework.answer_generation.registry import build_sweep_config

BEAM_DEFAULT_SPLIT = "100K"
BEAM_SUPPORTED_SPLITS = ("100K", "500K", "1M", "10M")
BEAM_10M_ALL_PLANS = [f"plan-{index}" for index in range(1, 11)]


@dataclass(frozen=True)
class BeamPreset:
    split: str
    router_kwargs: dict[str, Any] = field(default_factory=dict)
    chunks_per_batch: Optional[int] = None
    plans: Optional[list[str]] = None


BEAM_PRESETS: dict[str, BeamPreset] = {
    "100K": BeamPreset(split="100K"),
    "500K": BeamPreset(split="500K"),
    "1M": BeamPreset(
        split="1M",
        router_kwargs={
            "top_k_overrides": {
                "summarization": 80,
                "DEFAULT": 30,
            },
            "context_extension_rounds": 6,
            "wide_search_top_k": 150,
            "triplet_distance_penalty": 5.0,
        },
    ),
    "10M": BeamPreset(
        split="10M",
        router_kwargs={
            "top_k_overrides": {
                "summarization": 150,
                "DEFAULT": 50,
            },
            "context_extension_rounds": 8,
            "wide_search_top_k": 300,
            "triplet_distance_penalty": 4.0,
        },
        chunks_per_batch=40,
        plans=BEAM_10M_ALL_PLANS,
    ),
}


def get_beam_preset(split: str) -> BeamPreset:
    try:
        return BEAM_PRESETS[split]
    except KeyError as exc:
        available = ", ".join(BEAM_SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported BEAM split '{split}'. Available: {available}") from exc


def get_beam_router_kwargs(split: str) -> dict[str, Any]:
    return dict(get_beam_preset(split).router_kwargs)


def get_default_beam_sweep_configs(split: str) -> list[dict[str, Any]]:
    return [
        build_sweep_config(
            "cognee_graph_completion",
            config_name="graph_completion",
            strategy_kwargs={"top_k": 20},
        ),
        build_sweep_config(
            "graph_completion_decomposition",
            config_name="graph_completion_decomposition",
            strategy_kwargs={"top_k": 20},
        ),
        build_sweep_config(
            "cognee_graph_completion_context_extension",
            config_name="context_extension",
            strategy_kwargs={"top_k": 20, "context_extension_rounds": 4},
        ),
        build_sweep_config(
            "cognee_completion",
            config_name="raw_chunks",
            strategy_kwargs={"top_k": 30},
        ),
    ]


def apply_beam_prompt_policy(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enable BEAM per-question-type prompts for fixed-retriever sweep configs."""
    updated_configs = []
    for config in configs:
        updated_config = dict(config)
        if updated_config.get("mode") == "fixed_retriever":
            updated_config["use_beam_question_type_prompts"] = True
        updated_configs.append(updated_config)
    return updated_configs
