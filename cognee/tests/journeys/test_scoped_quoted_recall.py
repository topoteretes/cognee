"""Scoped quoted recall preserves document and DLT retrieval without an LLM answer."""

import importlib
from unittest.mock import AsyncMock

import pytest

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules import preflight
from cognee.modules.recall.config import RecallConfig
from cognee.modules.search.types import SearchType


@pytest.mark.journey
@pytest.mark.asyncio
@pytest.mark.parametrize("llm_available", [True, False])
@pytest.mark.parametrize("data_kind", ["text", "csv", "mixed"])
async def test_scoped_quoted_recall_returns_context_without_completion(
    clean_env, default_user, tmp_path, monkeypatch, llm_available, data_kind
):
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    dataset = f"journey_scoped_quote_{data_kind}_{llm_available}"
    expected = []

    if data_kind in ("csv", "mixed"):
        pytest.importorskip("dlt")
        csv = tmp_path / "cars.csv"
        csv.write_text("id,model\n1,Volvo C70\n2,Volvo V70\n")
        await cognee.add(str(csv), dataset_name=dataset, node_set=["garage"])
        expected.extend(["c70", "v70"])

    if data_kind in ("text", "mixed"):
        document = tmp_path / "maintenance.txt"
        document.write_text("Volvo C70 maintenance includes an annual inspection.")
        await cognee.add(str(document), dataset_name=dataset, node_set=["garage"])
        expected.append("annual inspection")

    if data_kind == "csv":
        outside = tmp_path / "other_cars.csv"
        outside.write_text("id,model\n3,Saab 9000 radiator replacement\n")
    else:
        outside = tmp_path / "other_garage.txt"
        outside.write_text("Saab 9000 maintenance includes a radiator replacement.")
    await cognee.add(str(outside), dataset_name=dataset, node_set=["other_garage"])
    await cognee.cognify(datasets=[dataset])

    monkeypatch.setattr(preflight, "llm_available", lambda _config: llm_available)
    config_module = importlib.import_module("cognee.modules.recall.config")
    config = RecallConfig(_env_file=None, recall_warmup_shortcircuit=False)
    monkeypatch.setattr(config_module, "get_recall_config", lambda: config)
    completion = AsyncMock(side_effect=AssertionError("Scoped quoted recall called an LLM"))
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", completion)

    results = await cognee.recall(
        query_text='"Volvo"',
        datasets=[dataset],
        node_name=["garage"],
        user=default_user,
    )

    assert results
    assert all(result.search_type == SearchType.HYBRID_COMPLETION for result in results)
    assert all(result.system_prompt for result in results)
    context = " ".join(result.text for result in results).lower()
    for fact in expected:
        assert fact in context
    assert "saab" not in context
    assert "radiator replacement" not in context
    completion.assert_not_awaited()
