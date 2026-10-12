import asyncio
from types import SimpleNamespace

import pytest

from cognee.eval_framework.locomo import aggregate
from cognee.eval_framework.locomo.eval_adapter import LocomoEvalAdapter
from cognee.eval_framework.locomo.metrics import llm_judge
from cognee.eval_framework.locomo.metrics.f1 import LocomoF1Metric, normalize_answer, token_f1
from cognee.eval_framework.locomo.model_registry import ensure_model_registered


def test_normalize_answer_strips_articles_punctuation_case():
    assert normalize_answer("The Adoption Agencies!") == "adoption agencies"
    assert normalize_answer(None) == ""


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("Adoption agencies", "adoption agencies", 1.0),
        ("She researched adoption agencies.", "Adoption agencies", 2 / 3),
        ("", "", 1.0),
        ("", "something", 0.0),
        ("cats", "dogs", 0.0),
    ],
)
def test_token_f1(prediction, gold, expected):
    f1, _, _ = token_f1(prediction, gold)
    assert f1 == pytest.approx(expected)


def test_f1_metric_protocol():
    metric = LocomoF1Metric()
    case = SimpleNamespace(actual_output="7 May 2023", expected_output="7 May 2023")
    assert metric.measure(case) == 1.0
    assert metric.score == 1.0
    assert "F1: 1.00" in metric.reason


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


def test_eval_adapter_runs_f1_and_judge(monkeypatch):
    async def fake_call_judge(prompt, *, model=None):
        return (
            '{"label": "CORRECT", "reason": "ok"}'
            if "Gold answer: yes" in prompt
            else '{"label": "WRONG", "reason": "no"}'
        )

    monkeypatch.setattr(llm_judge, "call_judge", fake_call_judge)
    adapter = LocomoEvalAdapter(max_concurrent_evaluations=2)
    answers = [
        {"question": "q1", "answer": "yes", "golden_answer": "yes", "question_type": "single_hop"},
        {"question": "q2", "answer": "maybe", "golden_answer": "no", "question_type": "multi_hop"},
    ]
    results = asyncio.run(adapter.evaluate_answers(answers, ["f1", "llm_judge"]))
    assert results[0]["metrics"]["f1"]["score"] == 1.0
    assert results[0]["metrics"]["llm_judge"]["score"] == 1.0
    assert results[1]["metrics"]["f1"]["score"] == 0.0
    assert results[1]["metrics"]["llm_judge"]["score"] == 0.0

    with pytest.raises(ValueError):
        asyncio.run(adapter.evaluate_answers(answers, ["rubric"]))


def test_aggregate_run_dir(tmp_path):
    qa = tmp_path / "conv_00" / "qa"
    qa.mkdir(parents=True)

    def entry(qt, f1, judge, answer="a"):
        return {
            "question": "q",
            "answer": answer,
            "golden_answer": "g",
            "question_type": qt,
            "metrics": {"f1": {"score": f1}, "llm_judge": {"score": judge}},
        }

    run0 = [
        entry("single_hop", 1.0, 1.0),
        entry("adversarial", 0.0, 0.0),
        entry("temporal", 0.5, None),
    ]
    run1 = [entry("single_hop", 0.0, 1.0, answer="ERROR: boom"), entry("adversarial", 1.0, 1.0)]
    aggregate.write_json(str(qa / "locomo_metrics_conv0_hybrid_completion_20_20_run0.json"), run0)
    aggregate.write_json(str(qa / "locomo_metrics_conv0_hybrid_completion_20_20_run1.json"), run1)

    summary = aggregate.aggregate_run_dir(tmp_path)
    block = summary["retrievers"]["hybrid_completion_20_20"]
    f1 = block["metrics"]["f1"]
    assert f1["overall"]["n"] == 5
    assert f1["overall"]["mean"] == pytest.approx(0.5)
    assert f1["mem0_comparable"]["n"] == 3  # adversarial excluded
    assert set(f1["by_question_type"]) == {"single_hop", "adversarial", "temporal"}
    assert len(f1["run_means"]) == 2
    assert block["metrics"]["llm_judge"]["unscored"] == 1
    assert block["answer_errors"] == 1

    markdown = aggregate.render_markdown(summary)
    assert "## hybrid_completion_20_20" in markdown
    assert "| single_hop |" in markdown


def test_ensure_model_registered_copies_capabilities_for_unknown_openai_model():
    import litellm

    name = "openai/gpt-5-test-unknown-mini"
    assert name not in litellm.model_cost
    row = ensure_model_registered(name)
    assert row is not None
    assert litellm.model_cost[name]["litellm_provider"] == "openai"
    assert "gpt-5-test-unknown-mini" in litellm.model_cost
    assert ensure_model_registered(name) is None  # idempotent
    assert ensure_model_registered("anthropic/claude-something") is None


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
