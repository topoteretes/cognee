"""BEAM / BEAM-10M dataset loading, reworked to load once per run.

``cognee.eval_framework.benchmark_adapters.beam_adapter``/``beam_10m_adapter`` reload the
HuggingFace dataset on every call (``load_beam_row``/``load_beam_10m_row``), which is fine for
loading a single conversation for an eval run but wasteful for preprocessing — a run there
touches every conversation in a split. This module splits "load the dataset" from "get one row"
so ``preprocess.py`` loads once per split and indexes into the already-loaded dataset per
conversation.

``get_beam_10m_plan_names``/``collect_beam_10m_plan_batches`` are pure row-extraction helpers
that don't touch dataset loading at all — copied over unchanged from ``beam_10m_adapter.py``.
"""

from __future__ import annotations

from typing import Any, Optional


def _datasets_load_dataset():
    try:
        import datasets as _datasets_lib
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for BEAM preprocessing. "
            "Install it with: pip install datasets"
        )
    return _datasets_lib.load_dataset


def load_beam_dataset(split: str):
    """Load the BEAM dataset for ``split`` once; index the result per conversation."""
    return _datasets_load_dataset()("Mohammadta/BEAM", split=split)


def load_beam_10m_dataset():
    """Load the BEAM-10M dataset once; index the result per conversation."""
    return _datasets_load_dataset()("Mohammadta/BEAM-10M", split="10M")


def get_beam_row(dataset: Any, conversation_index: int, *, dataset_label: str) -> dict[str, Any]:
    if conversation_index >= len(dataset):
        raise IndexError(
            f"conversation_index={conversation_index} out of range "
            f"({dataset_label} has {len(dataset)} conversations)"
        )
    return dataset[conversation_index]


def get_beam_10m_plan_names(
    chat_items: list[dict[str, Any]],
    plans: Optional[list[str]] = None,
) -> list[str]:
    if plans is not None:
        return plans

    plan_names = {
        plan_name
        for chat_item in chat_items
        for plan_name, batches in chat_item.items()
        if plan_name.startswith("plan-") and batches
    }
    return sorted(plan_names, key=lambda value: int(value.split("-")[1]))


def collect_beam_10m_plan_batches(
    chat_items: list[dict[str, Any]],
    *,
    plans: Optional[list[str]] = None,
    max_batches_per_plan: Optional[int] = None,
) -> dict[str, list[dict[str, Any]]]:
    plan_keys = get_beam_10m_plan_names(chat_items, plans=plans)
    plan_batches: dict[str, list[dict[str, Any]]] = {plan_key: [] for plan_key in plan_keys}

    for chat_item in chat_items:
        for plan_key in plan_keys:
            if (
                max_batches_per_plan is not None
                and len(plan_batches[plan_key]) >= max_batches_per_plan
            ):
                continue

            batches = chat_item.get(plan_key, []) or []
            if max_batches_per_plan is not None:
                remaining = max_batches_per_plan - len(plan_batches[plan_key])
                batches = batches[:remaining]
            plan_batches[plan_key].extend(batches)

    return plan_batches
