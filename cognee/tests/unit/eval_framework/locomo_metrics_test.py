import asyncio
from types import SimpleNamespace

import pytest

from cognee.eval_framework.locomo import aggregate
from cognee.eval_framework.locomo.eval_adapter import LocomoEvalAdapter
from cognee.eval_framework.locomo.metrics import llm_judge


def test_parse_judge_output_variants():
    assert llm_judge.parse_judge_output('{"label": "CORRECT", "reason": "same date"}') == {
        "label": "CORRECT",
        "reason": "same date",
    }
    assert (
        llm_judge.parse_judge_output('```json\n{"label":"wrong","reason":"r"}\n```')["label"]
        == "WRONG"
    )
    assert llm_judge.parse_judge_output("Verdict: WRONG because ...")["label"] == "WRONG"
    assert llm_judge.parse_judge_output("")["label"] is None
    assert llm_judge.parse_judge_output("no idea")["label"] is None


def test_render_judge_prompt_adversarial_mentions_distractor():
    prompt = llm_judge.render_judge_prompt(
        question="What did Melanie realize?",
        gold_answer="The conversation does not contain this information.",
        model_answer="self-care is important",
        question_type="adversarial",
        adversarial_answer="self-care is important",
    )
    assert "UNANSWERABLE" in prompt
    assert '"self-care is important"' in prompt
    assert "{gold_block}" not in prompt

    factual = llm_judge.render_judge_prompt(
        question="Q",
        gold_answer="7 May 2023",
        model_answer="May 7th 2023",
        question_type="temporal",
    )
    assert "Gold answer: 7 May 2023" in factual


def test_render_judge_prompt_ignores_abstention_distractor():
    prompt = llm_judge.render_judge_prompt(
        question="Q?",
        gold_answer="Not mentioned in the conversation",
        model_answer="Not mentioned in the conversation.",
        question_type="adversarial",
        adversarial_answer="Not mentioned",
    )
    assert "UNANSWERABLE" in prompt
    assert "distractor" not in prompt


def test_call_judge_grades_with_reasoning(monkeypatch):
    import litellm

    seen = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content='{"label": "CORRECT", "reason": "ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_judge, "_judge_api_key", lambda: "test-key")
    raw = asyncio.run(llm_judge.call_judge("prompt", model="openai/gpt-5.1"))
    assert llm_judge.parse_judge_output(raw)["label"] == "CORRECT"
    assert seen["reasoning_effort"] == "medium"
    assert seen["drop_params"] is True
    assert seen["model"] == "openai/gpt-5.1"


def test_eval_adapter_runs_the_judge(monkeypatch):
    prompts = []

    async def fake_call_judge(prompt, *, model=None):
        prompts.append(prompt)
        correct = "Gold answer: yes" in prompt or "Not mentioned" in prompt.split("Response")[-1]
        return '{"label": "CORRECT", "reason": "ok"}' if correct else '{"label": "WRONG"}'

    monkeypatch.setattr(llm_judge, "call_judge", fake_call_judge)
    adapter = LocomoEvalAdapter(max_concurrent_evaluations=2)
    answers = [
        {"question": "q1", "answer": "yes", "golden_answer": "yes", "question_type": "single_hop"},
        {"question": "q2", "answer": "maybe", "golden_answer": "no", "question_type": "multi_hop"},
        {
            "question": "q3",
            "answer": "Not mentioned in the conversation ",
            "golden_answer": "Not mentioned in the conversation",
            "question_type": "adversarial",
            "adversarial_answer": "a necklace",
        },
    ]
    results = asyncio.run(adapter.evaluate_answers(answers, ["llm_judge"]))
    assert [r["metrics"]["llm_judge"]["score"] for r in results] == [1.0, 0.0, 1.0]
    # The distractor passed through with the answer reaches the judge's prompt.
    assert any('distractor "a necklace"' in prompt for prompt in prompts)

    with pytest.raises(ValueError):
        asyncio.run(adapter.evaluate_answers(answers, ["f1"]))


def _metrics_entry(qt, judge, answer="a", question_idx=None):
    entry = {
        "question": "q",
        "answer": answer,
        "golden_answer": "g",
        "question_type": qt,
        "metrics": {"llm_judge": {"score": judge}},
    }
    if question_idx is not None:
        entry["question_idx"] = question_idx
    return entry


def test_aggregate_run_dir(tmp_path):
    qa = tmp_path / "conv_00" / "qa"
    qa.mkdir(parents=True)
    entry = _metrics_entry
    run0 = [
        entry("single_hop", 1.0),
        entry("adversarial", 0.0),
        entry("temporal", None),
        entry("open_domain", 0.0),
    ]
    run1 = [entry("single_hop", 1.0, answer="ERROR: boom"), entry("adversarial", 1.0)]
    aggregate.write_json(str(qa / "locomo_metrics_conv0_hybrid_completion_20_20_run0.json"), run0)
    aggregate.write_json(str(qa / "locomo_metrics_conv0_hybrid_completion_20_20_run1.json"), run1)

    summary = aggregate.aggregate_run_dir(tmp_path)
    block = summary["retrievers"]["hybrid_completion_20_20"]
    judge = block["metrics"]["llm_judge"]
    # The total leaves out open-domain; it is still reported per category.
    assert judge["slices"]["all_except_open_domain"]["n"] == 4
    assert judge["slices"]["all_except_open_domain"]["mean"] == pytest.approx(0.75)
    assert set(judge["by_question_type"]) == {"single_hop", "adversarial", "open_domain"}
    assert judge["by_question_type"]["open_domain"]["mean"] == pytest.approx(0.0)
    assert len(judge["run_means"]) == 2
    assert judge["unscored"] == 1
    assert block["answer_errors"] == 1
    assert "gold_variants" not in block

    markdown = aggregate.render_markdown(summary)
    assert "## hybrid_completion_20_20" in markdown
    assert "| single_hop |" in markdown


def test_aggregate_gold_variants(tmp_path):
    qa = tmp_path / "conv_03" / "qa"
    qa.mkdir(parents=True)
    rows = [
        _metrics_entry("single_hop", 1.0, question_idx=0),
        _metrics_entry("single_hop", 0.0, question_idx=1),
        _metrics_entry("open_domain", 0.0, question_idx=2),
        _metrics_entry("adversarial", 1.0, question_idx=3),
    ]
    aggregate.write_json(str(qa / "locomo_metrics_conv3_hybrid_run0.json"), rows)
    exclusions = tmp_path / "gold_exclusions.json"
    exclusions.write_text(
        '{"sets": {"external": {"questions": [{"conversation_index": 3, "question_idx": 1}]},'
        ' "ours": {"questions": [{"conversation_index": 3, "question_idx": 2},'
        ' {"conversation_index": 9, "question_idx": 0}]}},'
        ' "variants": {"without_external": ["external"], "without_both": ["external", "ours"]}}'
    )

    variants = aggregate.load_gold_exclusions(exclusions)
    assert variants["without_both"] == {(3, 1), (3, 2), (9, 0)}
    summary = aggregate.aggregate_and_write(tmp_path, exclusions)
    block = summary["retrievers"]["hybrid"]
    total = block["metrics"]["llm_judge"]["slices"]["all_except_open_domain"]
    assert (total["n"], total["mean"]) == (3, pytest.approx(2 / 3))
    external = block["gold_variants"]["without_external"]
    assert external["excluded_questions"] == 1
    external_total = external["metrics"]["llm_judge"]["slices"]["all_except_open_domain"]
    assert external_total["mean"] == pytest.approx(1.0)
    both = block["gold_variants"]["without_both"]
    assert both["excluded_questions"] == 2  # (9, 0) is not in this run
    assert "open_domain" not in both["metrics"]["llm_judge"]["by_question_type"]
    # The Markdown reports only the numbers without the corrupted questions.
    markdown = (tmp_path / "locomo_summary.md").read_text()
    assert "left out (`without_both`): 2" in markdown
    assert "| total (open-domain left out) | 1.000 (n=2) |" in markdown
    assert "(n=3)" not in markdown


def test_driver_helpers():
    from cognee.eval_framework.locomo.answer import select_questions
    from cognee.eval_framework.locomo.run_locomo_eval import parse_conversations

    assert parse_conversations("0") == [0]
    assert parse_conversations("0-2,5, 7") == [0, 1, 2, 5, 7]
    assert parse_conversations("3,1-2,2") == [1, 2, 3]

    questions = [
        {"question_idx": i, "question_type": qt}
        for i, qt in enumerate(["single_hop"] * 4 + ["temporal"] * 2 + ["adversarial"])
    ]
    picked = select_questions(questions, question_types=None, max_questions=4)
    assert len(picked) == 4
    # round-robin keeps the category mix instead of taking the first four single-hops
    assert {q["question_type"] for q in picked} == {"single_hop", "temporal", "adversarial"}
    assert [q["question_idx"] for q in picked] == sorted(q["question_idx"] for q in picked)
    only_temporal = select_questions(questions, question_types=["temporal"], max_questions=None)
    assert [q["question_type"] for q in only_temporal] == ["temporal", "temporal"]


def test_with_answer_prompt_uses_one_prompt_for_every_question():
    from cognee.eval_framework.locomo.answer import QA_PROMPT_PATH, with_answer_prompt

    configs = [{"name": "r", "qa_prompt_paths": {"DEFAULT": "x.txt", "temporal": "t.txt"}}]
    resolved = with_answer_prompt(configs)
    assert resolved[0]["qa_prompt_paths"] == {"DEFAULT": str(QA_PROMPT_PATH)}
    assert QA_PROMPT_PATH.exists()
    assert configs[0]["qa_prompt_paths"]["DEFAULT"] == "x.txt"  # input not mutated
