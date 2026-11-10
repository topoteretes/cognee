import argparse
import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from cognee.eval_framework.locomo import aggregate
from cognee.eval_framework.locomo.eval_adapter import LocomoEvalAdapter
from cognee.eval_framework.locomo.metrics import llm_judge
from cognee.eval_framework.locomo.scope import DEFAULT_GOLD_EXCLUSIONS, EvaluationScope


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


def test_find_metric_files_reads_one_run_only(tmp_path):
    rows = [_metrics_entry("single_hop", 1.0)]
    layout = {"qa": 0, "conv_01/qa": 1, "other_run/qa": 2, "other_run/conv_03/qa": 3}
    for folder, conv in layout.items():
        name = f"locomo_metrics_conv{conv}_hybrid_run0.json"
        aggregate.write_json(str(tmp_path / folder / name), rows)
    found = aggregate.find_metric_files(tmp_path)
    # the top-level qa/ and conv_NN/qa/ belong to this run; nested folders are other runs
    assert sorted(f["conversation_index"] for f in found) == [0, 1]


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


def _write_exclusions(path, sets):
    """A gold-exclusions file with ``sets = {name: [(conversation_index, question_idx), ...]}``."""
    document = {
        "sets": {
            name: {"questions": [{"conversation_index": c, "question_idx": q} for c, q in keys]}
            for name, keys in sets.items()
        },
        "variants": {"without_corrupted_golds": list(sets)},
    }
    path.write_text(json.dumps(document))
    return path


def test_scope_excludes_open_domain_and_audited_questions(tmp_path):
    path = _write_exclusions(
        tmp_path / "x.json", {"external": [(0, 1), (0, 2)], "ours": [(0, 2), (0, 3)]}
    )
    scope = EvaluationScope.load(path)

    assert scope.excluded_questions == {(0, 1), (0, 2), (0, 3)}  # both audits, overlap once
    assert not scope.includes(0, 1, "single_hop")
    assert not scope.includes(0, 3, "adversarial")
    assert not scope.includes(0, 4, "open_domain")
    assert scope.includes(0, 4, "adversarial")  # abstention stays in scope
    assert scope.includes(1, 1, "single_hop")  # the same position in another conversation
    assert scope.metadata() == {
        "gold_exclusions": str(path.resolve()),
        "gold_exclusions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gold_exclusion_variant": "without_corrupted_golds",
        "excluded_question_types": ["open_domain"],
    }


def test_default_scope_reads_the_published_manifest():
    assert len(EvaluationScope.load().excluded_questions) == 143


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"sets": {}}',
        '{"sets": {}, "variants": {"without_external": []}}',
        '{"sets": {}, "variants": {"without_corrupted_golds": ["unknown_set"]}}',
        (
            '{"sets": {"ours": {"questions": [{"conversation_index": "0", "question_idx": 1}]}},'
            ' "variants": {"without_corrupted_golds": ["ours"]}}'
        ),
    ],
)
def test_scope_rejects_a_malformed_manifest(tmp_path, content):
    path = tmp_path / "x.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        EvaluationScope.load(path)


def test_scope_rejects_a_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        EvaluationScope.load(tmp_path / "missing.json")


def _fake_answer_pipeline(monkeypatch, questions):
    """Stand-ins for the dataset and the sweep; returns the questions each sweep received."""
    from cognee.eval_framework.locomo import answer

    class FakeAdapter:
        def __init__(self, **_):
            pass

        def load_conversation(self, index):
            return SimpleNamespace(sample_id=f"conv-{index}")

        def questions_for(self, conversation, load_golden_context):
            return questions

    swept = []

    async def fake_sweep(*, questions, **_):
        swept.append(questions)
        return []

    monkeypatch.setattr(answer, "LocomoAdapter", FakeAdapter)
    monkeypatch.setattr(answer, "load_beam_sweep_payload_from_file", lambda path: {})
    monkeypatch.setattr(answer, "build_registry_base_configs", dict)
    monkeypatch.setattr(answer, "resolve_beam_sweep_config", lambda payload, base: [{"name": "r"}])
    monkeypatch.setattr(answer, "run_retriever_sweep_for_questions", fake_sweep)
    return swept


def _conversation_2_questions():
    types = ["single_hop", "single_hop", "open_domain", "temporal", "adversarial", "multi_hop"]
    return [
        {"conversation_index": 2, "question_idx": i, "question_type": qt, "question": f"q{i}"}
        for i, qt in enumerate(types)
    ]


def _answer_args(tmp_path, gold_exclusions, **overrides):
    return argparse.Namespace(
        **{
            "conversation_index": 2,
            "run_dir": tmp_path / "conv_02",
            "config_json_path": tmp_path / "config.json",
            "data_path": None,
            "gold_exclusions": gold_exclusions,
            "max_sessions": None,
            "max_questions": None,
            "question_types": None,
            "num_runs": 1,
            "max_concurrent_questions": 1,
            **overrides,
        }
    )


def test_answer_conversation_answers_only_in_scope_questions(tmp_path, monkeypatch):
    from cognee.eval_framework.locomo.answer import answer_conversation

    swept = _fake_answer_pipeline(monkeypatch, _conversation_2_questions())
    exclusions = _write_exclusions(tmp_path / "x.json", {"ours": [(2, 0), (5, 1)]})

    summary = asyncio.run(answer_conversation(_answer_args(tmp_path, exclusions, max_questions=3)))

    # (2, 0) is audited and question 2 is open-domain; (5, 1) is another conversation's.
    # The scope applies before --max-questions, so all three picked questions are in scope.
    assert [q["question_idx"] for q in swept[0]] == [1, 3, 4]
    written = json.loads((tmp_path / "conv_02" / "qa" / "locomo_questions_conv2.json").read_text())
    assert [q["question_idx"] for q in written] == [1, 3, 4]
    assert summary["question_count"] == 3
    assert summary["scope"]["questions_before_scope"] == 6
    assert summary["scope"]["questions_in_scope"] == 4
    assert summary["scope"]["gold_exclusions"] == str(exclusions.resolve())


def test_answer_conversation_never_answers_excluded_questions(tmp_path, monkeypatch):
    from cognee.eval_framework.locomo.answer import answer_conversation

    swept = _fake_answer_pipeline(monkeypatch, _conversation_2_questions())
    exclusions = _write_exclusions(tmp_path / "x.json", {"ours": [(2, 0), (2, 1)]})
    args = _answer_args(tmp_path, exclusions, question_types=["single_hop", "open_domain"])

    with pytest.raises(RuntimeError, match="No questions selected"):
        asyncio.run(answer_conversation(args))
    assert swept == []


def test_answer_conversation_checks_the_manifest_before_the_dataset(tmp_path, monkeypatch):
    from cognee.eval_framework.locomo import answer

    def no_dataset(**_):
        raise AssertionError("dataset loaded before the manifest was checked")

    monkeypatch.setattr(answer, "LocomoAdapter", no_dataset)
    with pytest.raises(FileNotFoundError):
        asyncio.run(answer.answer_conversation(_answer_args(tmp_path, tmp_path / "missing.json")))


def test_answer_cli_defaults_to_the_published_manifest(monkeypatch):
    from cognee.eval_framework.locomo import answer

    argv = ["answer", "--conversation-index", "0", "--run-dir", "r", "--config-json-path", "c.json"]
    monkeypatch.setattr(sys, "argv", argv)
    assert answer._parse_args().gold_exclusions == DEFAULT_GOLD_EXCLUSIONS


def test_driver_forwards_the_manifest_to_every_answer_child(tmp_path, monkeypatch):
    from cognee.eval_framework.locomo import run_locomo_eval as driver

    def forwarded(*extra):
        monkeypatch.setattr(sys, "argv", ["run_locomo_eval", "--run-dir", str(tmp_path), *extra])
        cli = driver.answer_args(driver._parse_args(), 0, tmp_path / "conv_00")
        return cli[cli.index("--gold-exclusions") + 1]

    assert forwarded() == str(DEFAULT_GOLD_EXCLUSIONS)
    custom = _write_exclusions(tmp_path / "custom.json", {"ours": [(0, 1)]})
    assert forwarded("--gold-exclusions", str(custom)) == str(custom.resolve())


def test_driver_rejects_a_bad_manifest_before_running_anything(tmp_path, monkeypatch):
    from cognee.eval_framework.locomo import run_locomo_eval as driver

    bad = tmp_path / "bad.json"
    bad.write_text('{"sets": {}, "variants": {}}')
    run_dir = tmp_path / "run"
    monkeypatch.setattr(driver, "run_child", lambda *_: pytest.fail("child process launched"))
    argv = ["run_locomo_eval", "--run-dir", str(run_dir), "--gold-exclusions", str(bad)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        driver.main()
    assert exit_info.value.code == 2
    assert not run_dir.exists()


def test_out_of_scope_metric_files(tmp_path):
    from cognee.eval_framework.locomo.run_locomo_eval import out_of_scope_metric_files

    scope = EvaluationScope.load(_write_exclusions(tmp_path / "x.json", {"ours": [(3, 1)]}))
    run_dir = tmp_path / "run"

    def write(folder, conversation_index, row):
        path = run_dir / folder / f"locomo_metrics_conv{conversation_index}_hybrid_run0.json"
        aggregate.write_json(str(path), [row])
        return path

    # Both layouts: an artifacts folder's qa/ and a driver run's conv_NN/qa/.
    write("qa", 3, _metrics_entry("adversarial", 0.0, question_idx=0))
    write("conv_02/qa", 2, _metrics_entry("single_hop", 1.0, question_idx=1))
    assert out_of_scope_metric_files(run_dir, scope) == []

    out_of_scope = [
        # Fresh rows carry no conversation index: (3, 1) is audited only by its file name.
        write("conv_03/qa", 3, _metrics_entry("single_hop", 1.0, question_idx=1)),
        write("conv_04/qa", 4, _metrics_entry("open_domain", 1.0, question_idx=0)),
        write("conv_05/qa", 5, _metrics_entry("single_hop", 1.0)),  # no question_idx
    ]
    assert out_of_scope_metric_files(run_dir, scope) == sorted(out_of_scope)


def test_driver_aggregates_only_in_scope_runs(tmp_path, monkeypatch):
    from cognee.eval_framework.locomo import run_locomo_eval as driver

    exclusions = _write_exclusions(tmp_path / "x.json", {"ours": [(0, 1)]})
    run_dir = tmp_path / "run"
    metrics = run_dir / "conv_00" / "qa" / "locomo_metrics_conv0_hybrid_run0.json"
    argv = ["run_locomo_eval", "--stage", "aggregate", "--run-dir", str(run_dir)]
    monkeypatch.setattr(sys, "argv", [*argv, "--gold-exclusions", str(exclusions)])
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    aggregate.write_json(str(metrics), [_metrics_entry("single_hop", 1.0, question_idx=0)])
    driver.main()
    summary = json.loads((run_dir / "locomo_summary.json").read_text())
    assert "gold_exclusions" not in summary
    assert "gold_variants" not in summary["retrievers"]["hybrid"]

    aggregate.write_json(str(metrics), [_metrics_entry("single_hop", 1.0, question_idx=1)])
    with pytest.raises(SystemExit, match="out-of-scope"):
        driver.main()
