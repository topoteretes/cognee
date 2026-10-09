import json

import pytest

from cognee.eval_framework.benchmark_adapters.locomo_adapter import (
    ADVERSARIAL_GOLD_ANSWER,
    LocomoAdapter,
    build_question_records,
    category_name,
    flatten_conversation,
    parse_conversation,
    parse_evidence,
)
from cognee.eval_framework.locomo.preprocess import (
    build_conversation_bundle,
    build_session_windows,
    conversation_overview_text,
    session_id_for,
    write_conversation_files,
)


def make_record(sample_id="conv-1", turns_per_session=(4, 3)):
    conversation = {"speaker_a": "Ana", "speaker_b": "Ben"}
    for session_index, count in enumerate(turns_per_session, start=1):
        conversation[f"session_{session_index}_date_time"] = f"1:00 pm on {session_index} May, 2023"
        conversation[f"session_{session_index}"] = [
            {
                "speaker": "Ana" if i % 2 == 0 else "Ben",
                "dia_id": f"D{session_index}:{i + 1}",
                "text": f"turn {i + 1} of session {session_index}",
                **({"blip_caption": "a cat on a sofa", "img_url": "['x']"} if i == 1 else {}),
            }
            for i in range(count)
        ]
    qa = [
        {"question": "Q single?", "answer": "A single", "evidence": "['D1:1']", "category": "4"},
        {
            "question": "Q multi?",
            "answer": "A multi",
            "evidence": "['D1:2', 'D2:1']",
            "category": 1,
        },
        {"question": "Q when?", "answer": "1 May 2023", "evidence": "['D1:3']", "category": "2"},
        {
            "question": "Q adversarial?",
            "adversarial_answer": "distractor",
            "evidence": "['D2:3']",
            "category": "5",
        },
    ]
    return {"sample_id": sample_id, "conversation": conversation, "qa": qa}


def test_parse_evidence_accepts_stringified_lists_and_lists():
    assert parse_evidence("['D1:2', 'D2:1']") == ["D1:2", "D2:1"]
    assert parse_evidence(["D1:2"]) == ["D1:2"]
    assert parse_evidence("") == []
    assert parse_evidence(None) == []


def test_category_names():
    assert category_name("1") == "multi_hop"
    assert category_name(2) == "temporal"
    assert category_name(3) == "open_domain"
    assert category_name(4) == "single_hop"
    assert category_name(5) == "adversarial"
    assert category_name("x") == "category_x"


def test_parse_conversation_and_flatten():
    conversation = parse_conversation(make_record(), 0)
    assert conversation.speaker_a == "Ana"
    assert [s.index for s in conversation.sessions] == [1, 2]
    assert conversation.turn_count == 7
    transcript = flatten_conversation(conversation)
    assert "--- Session 1 — 1:00 pm on 1 May, 2023 ---" in transcript
    assert "[shares a photo: a cat on a sofa]" in transcript


def test_build_question_records_adversarial_and_golden_context():
    conversation = parse_conversation(make_record(), 3)
    records = build_question_records(conversation, load_golden_context=True)
    assert [r["question_type"] for r in records] == [
        "single_hop",
        "multi_hop",
        "temporal",
        "adversarial",
    ]
    assert all(r["conversation_index"] == 3 for r in records)
    assert records[0]["question_idx"] == 0
    assert "Session 1, 1:00 pm on 1 May, 2023" in records[0]["golden_context"]
    adversarial = records[-1]
    assert adversarial["answer"] == ADVERSARIAL_GOLD_ANSWER
    assert adversarial["adversarial_answer"] == "distractor"


def test_build_question_records_drops_beyond_truncation_and_adversarial():
    conversation = parse_conversation(make_record(), 0)
    records = build_question_records(conversation, include_adversarial=False, max_session_index=1)
    # multi-hop needs D2:1 -> dropped; adversarial dropped by flag
    assert [r["question_type"] for r in records] == ["single_hop", "temporal"]


def test_adapter_load_corpus_from_file(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([make_record("conv-1"), make_record("conv-2")]), encoding="utf-8")

    adapter = LocomoAdapter(data_path=str(path))
    corpus, questions = adapter.load_corpus()
    assert len(corpus) == 2
    assert len(questions) == 8
    assert {q["conversation_id"] for q in questions} == {"conv-1", "conv-2"}

    single = LocomoAdapter(conversation_index=1, max_sessions=1, data_path=str(path))
    corpus, questions = single.load_corpus(limit=2)
    assert len(corpus) == 1
    assert len(questions) == 2
    assert all(q["conversation_index"] == 1 for q in questions)

    with pytest.raises(IndexError):
        single.load_conversation(5)


def test_session_windows_have_dated_headers_and_fold_small_tail():
    conversation = parse_conversation(make_record(turns_per_session=(7,)), 0)
    session = conversation.sessions[0]
    windows = build_session_windows(conversation, session, window_turns=3)
    # 7 turns / 3 -> [3, 3, 1]; the 1-turn tail folds into the previous window -> 2 windows
    assert [w.parts for w in windows] == [2, 2]
    assert windows[0].text.startswith(
        "Session 1 of the conversation between Ana and Ben, which took place at 1:00 pm on 1 May, 2023"
    )
    assert windows[1].last_dia_id == "D1:7"
    assert "Ben: turn 2 of session 1 [shares a photo: a cat on a sofa]" in windows[0].text
    assert build_session_windows(conversation, session, window_turns=100)[0].parts == 1
    with pytest.raises(ValueError):
        build_session_windows(conversation, session, window_turns=0)


def test_bundle_and_files(tmp_path):
    conversation = parse_conversation(make_record(), 0)
    bundle = build_conversation_bundle(conversation, window_turns=2)
    assert bundle["dataset_name"] == "locomo_conv_1"
    assert bundle["sessions"][0]["session_id"] == session_id_for(
        conversation, conversation.sessions[0]
    )
    assert bundle["sessions"][0]["session_id"] == "locomo_conv_1_s01"
    assert "Session 2 took place at 1:00 pm on 2 May, 2023" in conversation_overview_text(
        conversation
    )

    folder = write_conversation_files(bundle, tmp_path)
    files = sorted(p.name for p in folder.iterdir())
    assert files == ["manifest.json", "overview.txt", "session_01.json", "session_02.json"]
    items = json.loads((folder / "session_01.json").read_text(encoding="utf-8"))
    assert isinstance(items, list) and len(items) == 2
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["window_count"] == 2
