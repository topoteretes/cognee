"""The optional ingestion changes of the LoCoMo benchmark: resolved dates, prompts, index."""

import asyncio
import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.eval_framework.benchmark_adapters.locomo_adapter import (
    LocomoConversation,
    LocomoSession,
    LocomoTurn,
)
from cognee.eval_framework.locomo import dates
from cognee.eval_framework.locomo.preprocess import build_conversation_windows

LOCOMO_DIR = Path(dates.__file__).resolve().parent
WINDOW = (
    "Session 1 of the conversation between Calvin and Dave, which took place at 11:53 am on "
    "23 March, 2023 (part 2 of 3).\n"
    "Dave: When are you leaving?\n"
    "Calvin: I'm heading there next month. Been a few days since we talked."
)
ANNOTATED = WINDOW.replace(
    "next month.", "next month [= the month after 23 March 2023, i.e. April 2023]."
).replace("a few days since", "a few days[= a few days before 23 March 2023] since")


def test_is_faithful_accepts_added_annotations_only():
    assert dates.strip_annotations(ANNOTATED) == WINDOW
    assert dates.is_faithful(WINDOW, ANNOTATED)
    assert dates.is_faithful(WINDOW, WINDOW)
    assert dates.is_faithful(WINDOW, ANNOTATED.replace("\n", "\n\n"))  # whitespace is free
    assert not dates.is_faithful(WINDOW, ANNOTATED.replace("Dave:", "Dave said:"))
    assert not dates.is_faithful(WINDOW, "")


def test_render_prompt_fills_the_session_date():
    when = dates.session_datetime("11:53 am on 23 March, 2023")
    assert when.strftime("%Y-%m-%d %H:%M") == "2023-03-23 11:53"
    prompt = dates.render_prompt(WINDOW, when)
    assert "took place on Thursday, 23 March 2023." in prompt
    assert "[= the Saturday before 23 March 2023, i.e. <day month year>]" in prompt
    assert prompt.endswith("Excerpt:\n" + WINDOW)
    assert not re.search(r"\{(weekday|date|text)\}", prompt)


def _conversation() -> LocomoConversation:
    lines = ["When are you leaving?", "I'm heading there next month.", "Nice.", "Thanks!"]
    turns = [
        LocomoTurn(speaker="Dave" if i % 2 else "Calvin", dia_id=f"D1:{i}", text=text)
        for i, text in enumerate(lines, start=1)
    ]
    session = LocomoSession(index=1, date_time="11:53 am on 23 March, 2023", turns=turns)
    return LocomoConversation(
        conversation_index=0,
        sample_id="conv-1",
        speaker_a="Calvin",
        speaker_b="Dave",
        sessions=[session],
        qa=[],
    )


def test_resolve_window_dates_annotates_rejects_and_reuses(tmp_path, monkeypatch):
    conversation = _conversation()
    windows = build_conversation_windows(conversation, window_turns=2)
    assert len(windows) == 2

    async def fake_complete(prompt, *, model, api_key):
        excerpt = prompt.split("Excerpt:\n", 1)[1]
        if "next month" in excerpt:
            return excerpt.replace(
                "next month.", "next month [= the month after 23 March 2023, i.e. April 2023]."
            )
        return excerpt.replace("Nice.", "Nice!")  # the text changed: must be rejected

    monkeypatch.setattr(dates, "complete", fake_complete)
    cache = tmp_path / "date_annotations.json"
    resolved, stats = asyncio.run(
        dates.resolve_window_dates(conversation, windows, model="m", api_key="k", cache_path=cache)
    )
    assert "[= the month after 23 March 2023, i.e. April 2023]" in resolved[0].text
    assert resolved[0].word_count == len(resolved[0].text.split())
    assert resolved[1].text == windows[1].text
    assert {k: stats[k] for k in ("annotated", "rejected", "unchanged", "error", "reused")} == {
        "annotated": 1,
        "rejected": 1,
        "unchanged": 0,
        "error": 0,
        "reused": 0,
    }
    assert stats["annotations"] == 1
    saved = json.loads(cache.read_text())
    assert [row["status"] for row in saved["windows"]] == ["annotated", "rejected: text changed"]

    async def no_calls(prompt, **kwargs):
        raise AssertionError("cached windows must not be sent to the model again")

    monkeypatch.setattr(dates, "complete", no_calls)
    again, stats = asyncio.run(
        dates.resolve_window_dates(conversation, windows, model="m", api_key="k", cache_path=cache)
    )
    assert [w.text for w in again] == [w.text for w in resolved]
    assert stats["reused"] == 2


def test_annotate_text_keeps_the_window_after_repeated_errors(monkeypatch):
    async def boom(prompt, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(dates, "complete", boom)
    monkeypatch.setattr(dates, "RETRIES", 1)
    when = dates.session_datetime("11:53 am on 23 March, 2023")
    text, status = asyncio.run(
        dates.annotate_text(WINDOW, when, model="m", api_key=None, semaphore=asyncio.Semaphore(1))
    )
    assert text == WINDOW
    assert status.startswith("error: RuntimeError")


def test_summary_prompt_override_reaches_cognee_and_is_undone(monkeypatch):
    from cognee.eval_framework.locomo.ingest import (
        SUMMARY_MODULE,
        SUMMARY_PROMPT_NAME,
        summary_prompt_override,
    )

    module = importlib.import_module(SUMMARY_MODULE)
    seen = []

    def fake_read(name, *args, **kwargs):
        seen.append(name)
        return f"default:{name}"

    captured = {}

    async def fake_structured(text_input, system_prompt, response_model, *args, **kwargs):
        captured["system_prompt"] = system_prompt
        return response_model(summary="s")

    class Summary(BaseModel):
        summary: str

    monkeypatch.setattr(module, "read_query_prompt", fake_read)
    monkeypatch.setattr(module.LLMGateway, "acreate_structured_output", fake_structured)
    with summary_prompt_override("OUR PROMPT") as override:
        assert module.read_query_prompt("other.txt") == "default:other.txt"
        asyncio.run(module.extract_summary("chunk text", Summary))
    assert captured["system_prompt"] == "OUR PROMPT"
    assert override.calls == 1
    assert seen == ["other.txt"]
    assert module.read_query_prompt is fake_read  # restored on exit
    assert module.read_query_prompt(SUMMARY_PROMPT_NAME) == f"default:{SUMMARY_PROMPT_NAME}"

    monkeypatch.delattr(module, "read_query_prompt")
    with pytest.raises(RuntimeError, match="no longer loads"), summary_prompt_override("x"):
        pass


def test_ingest_args_passes_the_ingestion_options():
    from cognee.eval_framework.locomo.run_locomo_eval import ingest_args

    common = {
        "data_path": "locomo10.json",
        "window_turns": 6,
        "chunk_size": 4096,
        "max_sessions": None,
    }
    settings = SimpleNamespace(
        **common,
        resolve_dates=True,
        extraction_prompt=Path("/p/extract.txt"),
        summary_prompt=Path("/p/summary.txt"),
        global_context_index=True,
    )
    cli = ingest_args(settings, 3, Path("run/conv_03"))
    assert "--resolve-dates" in cli and "--global-context-index" in cli
    assert cli[cli.index("--extraction-prompt") + 1] == str(Path("/p/extract.txt"))
    assert cli[cli.index("--summary-prompt") + 1] == str(Path("/p/summary.txt"))

    plain = SimpleNamespace(
        **common,
        resolve_dates=False,
        extraction_prompt=None,
        summary_prompt=None,
        global_context_index=False,
    )
    options = {
        "--resolve-dates",
        "--extraction-prompt",
        "--summary-prompt",
        "--global-context-index",
    }
    assert not options & set(ingest_args(plain, 3, Path("run/conv_03")))


def test_packaged_prompts_and_config():
    prompts = LOCOMO_DIR / "prompts"
    assert "{weekday}" in (prompts / "resolve_dates.txt").read_text()
    assert "# 4. Conversations" in (prompts / "graph_extraction_conversations.txt").read_text()
    assert "Questions:" in (prompts / "summarize_content_questions.txt").read_text()
    config = json.loads((LOCOMO_DIR / "configs" / "locomo_hybrid_5_5_5_gci.json").read_text())
    kwargs = config["retrievers"][0]["kwargs"]
    assert (kwargs["chunks_top_k"], kwargs["entities_top_k"], kwargs["facts_top_k"]) == (5, 5, 5)
    assert kwargs["include_global_context_index"] is True
